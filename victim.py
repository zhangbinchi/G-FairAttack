import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from dgl.nn.pytorch import GATConv, GraphConv
from deeprobust.graph.defense.pgd import PGD, prox_operators


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid,nclass)

    def forward(self, g, x):
        x = self.body(g,x)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        # self.gc2 = GraphConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = self.dropout(x)
        # x = self.gc2(g, x)
        # x = self.dropout(x)
        return x


class GAT_body(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT_body, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)

        return logits


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()

        self.body = GAT_body(num_layers, in_dim, num_hidden, heads, feat_drop, attn_drop, negative_slope, residual)
        self.fc = nn.Linear(num_hidden,num_classes)

    def forward(self, g, inputs):

        logits = self.body(g,inputs)
        logits = self.fc(logits)

        return logits

def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    else:
        print("Model not implement")
        return

    return model

class FairGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(FairGNN,self).__init__()

        nhid = args.num_hidden
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,1)
        self.adv = nn.Linear(nhid,1)

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def forward(self,g,x):
        z = self.GNN(g,x)
        y = self.classifier(z)
        return y
    
    def optimize(self, g, x, labels, idx_train, idx_val, sens):
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        h = self.GNN(g, x)
        y = self.classifier(h)

        s_g = self.adv(h)

        s_score = sens.float().view(-1, 1)
        # s_score = (s_score > 0.5).float()
        y_score = torch.sigmoid(y)
        self.cov = torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))

        self.cls_loss = self.criterion(y[idx_train], labels[idx_train].unsqueeze(1).float())
        self.adv_loss = self.criterion(s_g, s_score)

        self.G_loss = self.cls_loss + self.args.alpha * self.cov - self.args.beta * self.adv_loss
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g, s_score)
        self.A_loss.backward()
        self.optimizer_A.step()
        G_loss_val = self.G_loss
        return G_loss_val


class EDITS(nn.Module):

    def __init__(self, args, nfeat, node_num, nclass, nfeat_out, adj_lambda, layer_threshold=2, dropout=0.1):
        super(EDITS, self).__init__()
        self.x_debaising = X_debaising(nfeat)
        self.layer_threshold = layer_threshold
        self.adj_renew = Adj_renew(node_num, nfeat, nfeat_out, adj_lambda)
        self.fc = nn.Linear(nfeat * (layer_threshold + 1), 1)
        self.lr = args.lr
        self.optimizer_feature_l1 = PGD(self.x_debaising.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=self.lr, alphas=[5e-6])
        self.dropout = nn.Dropout(dropout)
        G_params = list(self.x_debaising.parameters())
        self.optimizer_G = torch.optim.RMSprop(G_params, lr=self.lr, eps=1e-04, weight_decay=args.weight_decay)
        self.optimizer_A = torch.optim.RMSprop(self.fc.parameters(), lr=self.lr, eps=1e-04, weight_decay=args.weight_decay)

    def propagation_cat_new_filter(self, X_de, A_norm, layer_threshold):
        A_norm = A_norm.half()
        X_agg = X_de.half()
        for i in range(layer_threshold):
            X_de = A_norm.mm(X_de)
            X_agg = torch.cat((X_agg, X_de), dim=1)

        return X_agg.half()

    def forward(self, A, X):
        X_de = self.x_debaising(X)
        adj_new = self.adj_renew()
        agg_con = self.propagation_cat_new_filter(X_de.half(), adj_new.half(), layer_threshold=self.layer_threshold).half()  # A_de or A
        D_pre = self.fc(agg_con)
        D_pre = self.dropout(D_pre)
        return adj_new, X_de, D_pre, D_pre, agg_con

    def optimize(self, adj, features, idx_train, sens, epoch, lr):
        self.lr = lr
        for param_group in self.optimizer_G.param_groups:
            param_group["lr"] = lr
        for param_group in self.optimizer_A.param_groups:
            param_group["lr"] = lr

        # optimize attribute debiasing module
        # *************************  attribute debiasing  *************************
        self.train()
        self.optimizer_G.zero_grad()
        self.fc.requires_grad_(False)

        if epoch == 0:
            self.adj_renew.fit(adj, self.lr)

        _, X_debiased, predictor_sens, show, agg_con = self.forward(adj, features)
        positive_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
        negative_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)
        adv_loss = - (torch.mean(positive_eles) - torch.mean(negative_eles))
        loss_train = 5e-2 * (X_debiased - features).norm(2) + 0.1 * adv_loss  # credit
        # loss_train = 3e-2 * (X_debiased - features).norm(2) + adv_loss  # bail
        # loss_train = 30e-2 * (X_debiased - features).norm(2) + 0.8 * adv_loss  # german
        # loss_train = 0.01 * (X_debiased - features).norm(2) + 0.8 * adv_loss  # facebook
        # loss_train = 3e-2 * (X_debiased - features).norm(2) + 70 * adv_loss
        # loss_train = 3e-2 * (X_debiased - features).norm(2) + 70 * adv_loss

        loss_train.backward()
        self.optimizer_G.step()
        self.optimizer_feature_l1.zero_grad()
        self.optimizer_feature_l1.step()

        # optimize structural debiasing module
        # *************************  structural debiasing  *************************
        _, X_debiased, predictor_sens, show, agg_con = self.forward(adj, features)

        positive_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
        negative_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)

        adv_loss = - (torch.mean(positive_eles) - torch.mean(negative_eles))
        self.adj_renew.train_adj(X_debiased, adj, adv_loss, epoch, lr)

        # *************************  PGD  *************************
        param = self.state_dict()
        zero = torch.zeros_like(param["x_debaising.s"])
        one = torch.ones_like(param["x_debaising.s"])
        param["x_debaising.s"] = torch.where(param["x_debaising.s"] > 1, one, param["x_debaising.s"])
        param["x_debaising.s"] = torch.where(param["x_debaising.s"] < 0, zero, param["x_debaising.s"])
        # param["x_debaising.s"] = torch.clamp(param["x_debaising.s"], min=0, max=1)
        self.load_state_dict(param)

        # optimize WD approximator
        # *************************  optimize WD approximator  *************************
        for i in range(8):
            self.fc.requires_grad_(True)
            self.optimizer_A.zero_grad()
            _, X_debiased, predictor_sens, show, agg_con = self.forward(adj, features)

            positive_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] > 0)
            negative_eles = torch.masked_select(predictor_sens[idx_train].squeeze(), sens[idx_train] <= 0)

            loss_train = torch.mean(positive_eles) - torch.mean(negative_eles)
            loss_train.backward()
            self.optimizer_A.step()
            for p in self.fc.parameters():
                p.data.clamp_(-0.02, 0.02)
            # print("loss_train:  " + str(loss_train.item()))

        return 0


class EstimateAdj(nn.Module):

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n), requires_grad=True)
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            adj = adj.to_dense()
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

class Adj_renew(nn.Module):

    def __init__(self, node_num, nfeat, nfeat_out, adj_lambda):
        super(Adj_renew, self).__init__()
        self.node_num = node_num
        self.nfeat = nfeat
        self.nfeat_out = nfeat_out
        self.adj_lambda = adj_lambda

        self.reset_parameters()

    def fit(self, adj, lr):
        estimator = EstimateAdj(adj, symmetric=False, device='cuda').to('cuda').half()
        # estimator = EstimateAdj(adj, symmetric=False, device='cuda').half()
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                              momentum=0.9, lr=lr)   # 0.005

        self.optimizer_l1 = PGD(estimator.parameters(),
                        proxs=[prox_operators.prox_l1],
                        lr=lr, alphas=[5e-4])  # 5e-4
        self.optimizer_nuclear = PGD(estimator.parameters(),
                  proxs=[prox_operators.prox_nuclear],
                  lr=lr, alphas=[1.5])

    def reset_parameters(self):
        pass

    def the_norm(self):
        return self.estimator._normalize(self.estimator.estimated_adj)

    def forward(self):
        return self.estimator.estimated_adj

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


    def train_adj(self, features, adj, adv_loss, epoch, lr):
        for param_group in self.optimizer_adj.param_groups:
            param_group["lr"] = lr

        estimator = self.estimator
        estimator.train()
        self.optimizer_adj.zero_grad()

        delta = estimator.estimated_adj - adj
        loss_fro = torch.sum(delta.mul(delta))
        loss_diffiential = 1 * loss_fro + 15 * adv_loss
        # loss_diffiential = 1 * loss_fro + 20 * adv_loss  # German
        # loss_diffiential = 0.1 * loss_fro + 20 * adv_loss  # Facebook
        loss_diffiential.backward()
        self.optimizer_adj.step()
        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        estimator.estimated_adj.data.copy_(torch.clamp(
                  estimator.estimated_adj.data, min=0, max=1))
        estimator.estimated_adj.data.copy_((estimator.estimated_adj.data + estimator.estimated_adj.data.transpose(0, 1)) / 2)

        return estimator.estimated_adj


class X_debaising(nn.Module):

    def __init__(self, in_features):
        super(X_debaising, self).__init__()
        self.in_features = in_features
        self.s = Parameter(torch.FloatTensor(in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.s.data.uniform_(1, 1)

    def forward(self, feature):
        return torch.mm(feature, torch.diag(self.s))
