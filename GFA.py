import os.path
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.utils import convert
from tqdm import tqdm
import utils
import math
import numpy as np
import scipy.sparse as sp
from surrogate import SGC
from gcn import GCN
from kernel_estimator import KernelEstimator
import time
import pdb
from numba import njit, prange
import numba
import pynvml


@njit(nopython=True)
def edge_loop(edge_set, edge_coo, edge_ptr, AXT, XT, degree, Zlast, y, train_idx, test_idx, sens):
    loss = np.zeros(edge_set.shape[0])
    fair = np.zeros(edge_set.shape[0])
    train_filt = np.zeros(AXT.shape[0]).astype(np.bool_)
    test_filt = np.zeros(AXT.shape[0]).astype(np.bool_)
    train_filt[train_idx] = True
    test_filt[test_idx] = True
    for j in range(edge_set.shape[0]):
        e = edge_set[j]
        N0_set_rm01 = set(edge_coo[1, edge_ptr[e[0]]:edge_ptr[e[0]+1]]) - set(e.astype(np.int_))
        N1_set_rm01 = set(edge_coo[1, edge_ptr[e[1]]:edge_ptr[e[1]+1]]) - {e[0], e[1]}
        neighbor_0 = list(N0_set_rm01 - N1_set_rm01)
        neighbor_1 = list(N1_set_rm01 - N0_set_rm01)
        neighbor_both = list(N0_set_rm01 & N1_set_rm01)
        indicator = 0
        if e[1] in edge_coo[1, edge_ptr[e[0]]:edge_ptr[e[0]+1]]:
            indicator = 1
        Z = Zlast.copy()

        if indicator == 0:
            Z[e[0]] = (Zlast[e[0]] - AXT[e[0]] / np.square(degree[e[0]])) * degree[e[0]] / (degree[e[0]]+1) + (AXT[e[0]]+XT[e[1]]) / np.square(degree[e[0]]+1) + (AXT[e[1]]+XT[e[0]]) / (degree[e[0]]+1) / (degree[e[1]]+1)
        else:
            Z[e[0]] = (Zlast[e[0]] - AXT[e[0]] / np.square(degree[e[0]]) - AXT[e[1]] / degree[e[0]] / degree[e[1]]) * degree[e[0]] / (degree[e[0]]-1) + (AXT[e[0]]-XT[e[1]]) / np.square(degree[e[0]]-1)

        if indicator == 0:
            Z[e[1]] = (Zlast[e[1]] - AXT[e[1]] / np.square(degree[e[1]])) * degree[e[1]] / (degree[e[1]]+1) + (AXT[e[1]]+XT[e[0]]) / np.square(degree[e[1]]+1) + (AXT[e[0]]+XT[e[1]]) / (degree[e[1]]+1) / (degree[e[0]]+1)
        else:
            Z[e[1]] = (Zlast[e[1]] - AXT[e[1]] / np.square(degree[e[1]]) - AXT[e[0]] / degree[e[1]] / degree[e[0]]) * degree[e[1]] / (degree[e[1]]-1) + (AXT[e[1]]-XT[e[0]]) / np.square(degree[e[1]]-1)
        
        for u in neighbor_0:
            Z[u] = Zlast[u] - AXT[e[0]] / degree[u] / degree[e[0]] + (AXT[e[0]] + (1 - 2*indicator)*XT[e[1]]) / degree[u] / (degree[e[0]] + 1 - 2*indicator)
        for u in neighbor_1:
            Z[u] = Zlast[u] - AXT[e[1]] / degree[u] / degree[e[1]] + (AXT[e[1]] + (1 - 2*indicator)*XT[e[0]]) / degree[u] / (degree[e[1]] + 1 - 2*indicator)
        for u in neighbor_both:
            Z[u] = Zlast[u] - AXT[e[0]] / degree[u] / degree[e[0]] - AXT[e[1]] / degree[u] / degree[e[1]] + (AXT[e[0]] + (1 - 2*indicator)*XT[e[1]]) / degree[u] / (degree[e[0]] + 1 - 2*indicator) + (AXT[e[1]] + (1 - 2*indicator)*XT[e[0]]) / degree[u] / (degree[e[1]] + 1 - 2*indicator)
            
        Z_sm = 1 / (1 + np.exp(-Z))
        loss[j] = -(y[train_filt] * np.log(Z_sm[train_filt]) + (1 - y[train_filt]) * np.log(1 - Z_sm[train_filt])).mean()
        pred = Z[test_filt] >= 0
        fair[j] = np.abs(pred[sens[test_filt] == 0].mean() - pred[sens[test_filt] == 1].mean())

    return loss, fair


@njit(nopython=True)
def neighbor_dist(Zlast, edge_coo, edge_ptr):
    dist = np.abs(Zlast)
    dist = dist.max() - dist
    neighbor_dist = np.zeros(Zlast.shape[0])
    for i in range(Zlast.shape[0]):
        neighbor_dist[i] = dist[edge_coo[1, edge_ptr[i]:edge_ptr[i+1]]].sum()
    return neighbor_dist


def choose_edge_approx(edge_set, Zlast, edge_coo, edge_ptr, prop):
    dist = neighbor_dist(Zlast, edge_coo, edge_ptr)
    score = dist[edge_set[:, 0]] + dist[edge_set[:, 1]]
    num = int(edge_set.shape[0] * prop)
    rank_num_ind = np.argpartition(score, -num)[-num:]
    return edge_set[rank_num_ind]


@njit(nopython=True)
def rank_score(edge_set, Zlast, edge_coo, edge_ptr):
    dist = np.abs(Zlast)
    dist = dist.max() - dist
    neighbor_dist = np.zeros(edge_set.shape[0])
    for i in range(edge_set.shape[0]):
        neighbor_union = set(edge_coo[1, edge_ptr[edge_set[i, 0]]:edge_ptr[edge_set[i, 0]+1]]) | set(edge_coo[1, edge_ptr[edge_set[i, 1]]:edge_ptr[edge_set[i, 1]+1]])
        neighbor_union = np.array(list(neighbor_union))
        neighbor_dist[i] = dist[neighbor_union].sum()
    return neighbor_dist


def choose_edge(edge_set, Zlast, edge_coo, edge_ptr, prop):
    neighbor_dist = rank_score(edge_set, Zlast, edge_coo, edge_ptr)
    num = int(edge_set.shape[0] * prop)
    rank_num_ind = np.argpartition(neighbor_dist, -num)[-num:]
    return edge_set[rank_num_ind]


@njit(nopython=True)
def mutual_neighbor(edge_set, Zlast, edge_coo, edge_ptr):
    neighbor = np.zeros((Zlast.shape[0], Zlast.shape[0]))
    for i in range(Zlast.shape[0]):
        neighbor_i = list(edge_coo[1, edge_ptr[edge_set[i, 0]]:edge_ptr[edge_set[i, 0] + 1]])
        for j in range(len(neighbor_i)):
            for k in range(j + 1, len(neighbor_i)):
                neighbor[neighbor_i[j], neighbor_i[k]] += 1
    return neighbor[edge_set[:, 0], edge_set[:, 1]]


class GFA:
    def __init__(self, adj, x, y, sensitive_attr, train_idx, val_idx, test_idx, dropout, lr, wd, train_iters, device, with_bias, dataset_name):
        super(GFA, self).__init__()
        """
        x is the node attribute matrix, a ndarray with shape [N, d]
        y is the ground truth vector, a ndarray with shape [N]
        adj is the adjacency matrix, a sparse matrix with shape [N, N]
        sens is the sensitive attribute vector, a ndarray with shape [N]
        """

        assert (adj != adj.T).sum() == 0, "Adjacency matrix is not symmetric!"
        assert (adj.diagonal() != 0).sum() == 0, "Adjacency matrix contains self-loops!"
        assert (adj.data != 1).sum() == 0, "Adjacency matrix is not binary!"
        self.adj_atk = adj
        self.adj_ori = self.adj_atk.copy()
        self.adj_self_connection = None

        self.x = x
        self.y = y
        self.sens = sensitive_attr
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.N = self.x.shape[0]
        self.nfeat = self.x.shape[1]
        self.nclass = self.y.max()
        self.with_bias = with_bias

        self.surrogate_model = SGC(nfeat=self.nfeat, nclass=self.nclass, dropout=dropout, with_bias=with_bias).to(device)
        self.surrogate_optimizer = optim.Adam(self.surrogate_model.parameters(), lr=lr, weight_decay=wd)
        self.train_iters = train_iters
        self.device = device
        self.dataset_name = dataset_name
        self.trained_weight = None
        self.trained_bias = None
    

    def _initialize(self):
        stdv = 1. / math.sqrt(self.nclass)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            self.bias.data.uniform_(-stdv, stdv)


    def train_surrogate(self, alpha, h, int_num):
        print('===== training surrogate model')
        self.surrogate_model.initialize()

        adj_norm = utils.normalize_adj(self.adj_atk)
        adj_norm_square = adj_norm @ adj_norm
        adj_norm_square = utils.sparse_mx_to_torch_sparse_tensor(adj_norm_square).to(self.device)
        input = torch.FloatTensor(self.x).to(self.device)
        labels = torch.LongTensor(self.y).to(self.device)
        idx_train = torch.LongTensor(self.train_idx).to(self.device)
        idx_val = torch.LongTensor(self.val_idx).to(self.device)
        idx_test = torch.LongTensor(self.test_idx).to(self.device)
        output_return = None
        min_val_loss = 1e10
        best_result = {}
        for i in range(self.train_iters):
            self.surrogate_model.train()
            self.surrogate_optimizer.zero_grad()
            output = self.surrogate_model(torch.spmm(adj_norm_square, input))
            den0 = torch.sigmoid(output.view(-1))[self.sens == 0]
            den1 = torch.sigmoid(output.view(-1))[self.sens == 1]
            integral_x = torch.arange(0, 1, 1 / int_num).to(self.device)
            reg = torch.abs(KernelEstimator(integral_x, den0, h) - KernelEstimator(integral_x, den1, h)).mean()
            loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float()) + alpha * reg
            loss_train.backward()
            self.surrogate_optimizer.step()

            self.surrogate_model.eval()
            output = self.surrogate_model(torch.spmm(adj_norm_square, input))
            den0 = torch.sigmoid(output.view(-1))[self.sens == 0]
            den1 = torch.sigmoid(output.view(-1))[self.sens == 1]
            integral_x = torch.arange(0, 1, 1 / int_num).to(self.device)
            reg_val = torch.abs(KernelEstimator(integral_x, den0, h) - KernelEstimator(integral_x, den1, h)).mean()
            loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float()) + alpha * reg_val
            if loss_val < min_val_loss:
                min_val_loss = loss_val
                best_result['epoch'] = i
                best_result['pred0'] = (output[self.test_idx] < 0).sum()
                best_result['pred1'] = (output[self.test_idx] >= 0).sum()
                best_result['acc'] = utils.accuracy1(output[idx_test], labels[idx_test].cpu())
                best_result['parity'], best_result['equality'] = utils.fair_metric1(output[idx_test], labels[idx_test].cpu(), self.sens[self.test_idx])
                self.trained_weight = self.surrogate_model.W.weight.detach().cpu().numpy().T
                if self.with_bias:
                    self.trained_bias = self.surrogate_model.W.bias.detach().cpu().numpy()
                output_return = output.detach().cpu().numpy()

            interval = 10
            if self.train_iters >= 500:
                interval = 100
            
            if i % interval == 0:
                test_acc = utils.accuracy1(output[idx_test], labels[idx_test].cpu())
                parity, equality = utils.fair_metric1(output[idx_test], labels[idx_test].cpu(), self.sens[self.test_idx])
                print(f"iteration {i}, loss: {loss_train.item():.4f}, loss_reg: {alpha * reg.item():.4f}, val_loss: {loss_val.item():.4f}, test_acc: {test_acc:.4f}, parity: {parity:.4f}, equality: {equality:.4f}, train_acc: {utils.accuracy1(output[idx_train], labels[idx_train].cpu()):.4f}")
        print(f"best epoch: {best_result['epoch']}, best acc: {best_result['acc']:.4f}, parity: {best_result['parity']:.4f}, equality: {best_result['equality']:.4f} best pred 0: {best_result['pred0']}, best pred 1: {best_result['pred1']}")
        return output_return


    def evasion_attack(self, p, loss_p, alpha, h, int_num, prop=1):
        budget = int(self.adj_ori.sum() * p / 2)
        print(budget)

        self.adj_atk = self.adj_ori.copy()
        _ = self.train_surrogate(alpha, h, int_num)

        Theta = self.trained_weight.copy()
        B = 0
        if self.with_bias:
            B = self.trained_bias.copy()
        adj_self_loop = self.adj_ori.tolil()
        adj_self_loop[np.arange(self.N), np.arange(self.N)] = 1
        neighbour = list()
        for i in range(self.N):
            neighbour.append(set(np.where(adj_self_loop[i].todense().A.squeeze()==1)[0]))
        adj_norm = utils.normalize_adj(self.adj_ori)
        Zlast = adj_norm.dot(adj_norm) @ self.x @ Theta + B
        Zlast = Zlast.squeeze()
        AXT = adj_self_loop @ self.x @ Theta
        AXT = AXT.squeeze()
        XT = self.x @ Theta
        XT = XT.squeeze()
        degree = adj_self_loop.sum(1).A.squeeze()
        edge_set = np.vstack([np.hstack([i * np.ones((self.N-i-1, 1)), np.arange(i+1, self.N).reshape(-1, 1)]) for i in range(self.N-1)]).astype(np.int_)
        adj_self_loop = adj_self_loop.tocoo()
        edge_coo = np.vstack([adj_self_loop.row, adj_self_loop.col]).astype(np.int_)
        edge_ptr = [0]
        for i in range(self.N):
            edge_ptr.append(edge_ptr[-1] + degree[i])
        edge_ptr = np.array(edge_ptr, dtype=np.int_)
        pred = Zlast[self.test_idx] >= 0
        fair_last = np.abs(pred[self.sens[self.test_idx] == 0].mean() - pred[self.sens[self.test_idx] == 1].mean())
        Zlast_sm = 1 / (1 + np.exp(-Zlast))
        loss_init = -(self.y[self.train_idx] * np.log(Zlast_sm[self.train_idx]) + (1 - self.y[self.train_idx]) * np.log(1 - Zlast_sm[self.train_idx])).mean()
        best_adj = adj_self_loop.tocsr()
        best_fair = fair_last
        adj_self_loop = adj_self_loop.tolil()

        for epoch in range(budget):
            print(f"Total budget: {budget}, attacking edge No.{epoch + 1}...")
            start = time.time()
            if prop != 1:
                edge_set_opt = choose_edge_approx(edge_set, Zlast, edge_coo, edge_ptr, prop)
            else:
                edge_set_opt = edge_set
            print(f"Loop number: {edge_set_opt.shape[0]}")
            loss, fair = edge_loop(edge_set_opt, edge_coo, edge_ptr, AXT, XT, degree, Zlast, self.y, self.train_idx, self.test_idx, self.sens)
            end = time.time()
            print(f"Finish looping, time: {end - start}")

            delta_fair = fair - fair_last
            delta_loss = loss - loss_init
            score = delta_fair - np.abs(np.sum(delta_fair * delta_loss) / np.sum(delta_loss * delta_loss) * delta_loss)
            flip_edge_ind = score.argmax()

            flip_edge = edge_set_opt[flip_edge_ind]
            if adj_self_loop[flip_edge[0], flip_edge[1]] == 0:
                adj_self_loop[flip_edge[0], flip_edge[1]] = 1
                adj_self_loop[flip_edge[1], flip_edge[0]] = 1
                degree[flip_edge[0]] += 1
                degree[flip_edge[1]] += 1
                AXT[flip_edge[0]] += XT[flip_edge[1]]
                AXT[flip_edge[1]] += XT[flip_edge[0]]
                neighbour[flip_edge[0]].add(flip_edge[1])
                neighbour[flip_edge[1]].add(flip_edge[0])
                edge_coo = np.hstack([edge_coo[:, :edge_ptr[flip_edge[0]+1]], flip_edge.reshape(2, 1), edge_coo[:, edge_ptr[flip_edge[0]+1]:]])
                edge_ptr[flip_edge[0]+1:] += 1
                edge_coo = np.hstack([edge_coo[:, :edge_ptr[flip_edge[1]+1]], np.array([flip_edge[1], flip_edge[0]]).reshape(2, 1), edge_coo[:, edge_ptr[flip_edge[1]+1]:]])
                edge_ptr[flip_edge[1]+1:] += 1
            else:
                adj_self_loop[flip_edge[0], flip_edge[1]] = 0
                adj_self_loop[flip_edge[1], flip_edge[0]] = 0
                degree[flip_edge[0]] -= 1
                degree[flip_edge[1]] -= 1
                AXT[flip_edge[0]] -= XT[flip_edge[1]]
                AXT[flip_edge[1]] -= XT[flip_edge[0]]
                neighbour[flip_edge[0]].remove(flip_edge[1])
                neighbour[flip_edge[1]].remove(flip_edge[0])
                edge_ind = np.where(edge_coo[1, edge_ptr[flip_edge[0]]:edge_ptr[flip_edge[0]+1]] == flip_edge[1])[0][0] + edge_ptr[flip_edge[0]]
                assert edge_coo[0, edge_ind] == flip_edge[0] and edge_coo[1, edge_ind] == flip_edge[1]
                edge_coo = np.delete(edge_coo, edge_ind, axis=1)
                edge_ptr[flip_edge[0]+1:] -= 1
                edge_ind = np.where(edge_coo[1, edge_ptr[flip_edge[1]]:edge_ptr[flip_edge[1]+1]] == flip_edge[0])[0][0] + edge_ptr[flip_edge[1]]
                assert edge_coo[0, edge_ind] == flip_edge[1] and edge_coo[1, edge_ind] == flip_edge[0]
                edge_coo = np.delete(edge_coo, edge_ind, axis=1)
                edge_ptr[flip_edge[1]+1:] -= 1

            neighbor_set = list(neighbour[flip_edge[0]].union(neighbour[flip_edge[1]]))
            Zt = list()
            for u in tqdm(neighbor_set):
                neighbor_u = list(neighbour[u])
                Z = (AXT[neighbor_u] / degree[u] / degree[neighbor_u]).sum() + B
                Zt.append(Z)
            Zt = np.array(Zt).reshape(-1)
            Zlast[neighbor_set] = Zt
            del_ind = np.where((edge_set == flip_edge).all(1))[0]
            edge_set = np.delete(edge_set, del_ind, axis=0)

            edge_ptr_check = [0]
            for i in range(self.N):
                edge_ptr_check.append(edge_ptr_check[-1] + degree[i])
            edge_ptr_check = np.array(edge_ptr_check)
            assert (edge_ptr_check != edge_ptr).sum() == 0
            for i in range(self.N):
                assert (edge_coo[0, edge_ptr[i]:edge_ptr[i+1]] != i).sum() == 0
            
            pred = Zlast[self.test_idx] >= 0
            fair_cur = np.abs(pred[self.sens[self.test_idx] == 0].mean() - pred[self.sens[self.test_idx] == 1].mean())
            fair_last = fair_cur
            print(f"edge change: {((adj_self_loop - np.eye(self.N))!=self.adj_ori).sum() / 2}, Delta SP: {fair_cur}")

            if fair_cur >= best_fair and np.abs(loss[flip_edge_ind] - loss_init) <= (loss_p * loss_init):
                best_fair = fair_cur
                best_adj = adj_self_loop.tocsr()

        self.adj_atk = best_adj - sp.eye(self.N)
        if not os.path.exists('evasion_structure'):
            os.mkdir('evasion_structure')
        sp.save_npz('evasion_structure/'+self.dataset_name+'_p_'+str(p)+'_lp_'+str(loss_p)+'_prop_'+str(prop)+'_alpha_'+str(alpha)+'.npz', self.adj_atk)
        return best_fair
    

    def poisoning_attack(self, p, loss_p, alpha, h, int_num, prop=1):
        budget = int(self.adj_ori.sum() * p / 2)
        self.adj_atk = self.adj_ori.tolil()
        print(f"Start training 0/{budget}...")
        start = time.time()
        _ = self.train_surrogate(alpha, h, int_num)
        end = time.time()
        print(f"Finish training, time: {end - start}")
        Theta = self.trained_weight.copy()
        Theta_init = self.trained_weight.copy()
        B = 0
        bias_init = 0
        if self.with_bias:
            B = self.trained_bias.copy()
            bias_init = self.trained_bias.copy()
        adj_self_loop = self.adj_ori.tolil()
        adj_self_loop[np.arange(self.N), np.arange(self.N)] = 1
        neighbour = list()
        for i in range(self.N):
            neighbour.append(set(np.where(adj_self_loop[i].todense().A.squeeze()==1)[0]))
        adj_norm = utils.normalize_adj(self.adj_ori)
        X_last = adj_norm.dot(adj_norm) @ self.x
        AX = adj_self_loop @ self.x
        X = self.x.copy()
        AXT = (AX @ Theta).reshape(-1)
        XT = (X @ Theta).reshape(-1)
        degree = adj_self_loop.sum(1).A.squeeze()
        edge_set = np.vstack([np.hstack([i * np.ones((self.N-i-1, 1)), np.arange(i+1, self.N).reshape(-1, 1)]) for i in range(self.N-1)]).astype(int)
        adj_self_loop = adj_self_loop.tocoo()
        edge_coo = np.vstack([adj_self_loop.row, adj_self_loop.col]).astype(np.int_)
        edge_ptr = [0]
        for i in range(self.N):
            edge_ptr.append(edge_ptr[-1] + degree[i])
        edge_ptr = np.array(edge_ptr, dtype=np.int_)
        Z_last = (X_last @ Theta + B).reshape(-1)
        pred = Z_last[self.test_idx] >= 0
        fair_last = np.abs(pred[self.sens[self.test_idx] == 0].mean() - pred[self.sens[self.test_idx] == 1].mean())
        Z_last_sm = 1 / (1 + np.exp(-Z_last))
        loss_init = -(self.y[self.train_idx] * np.log(Z_last_sm[self.train_idx]) + (1 - self.y[self.train_idx]) * np.log(1 - Z_last_sm[self.train_idx])).mean()
        loss_last = loss_init
        best_adj = adj_self_loop.tocsr()
        adj_self_loop = adj_self_loop.tolil()

        for epoch in range(budget):
            print(f"Total budget: {budget}, attacking edge No.{epoch + 1}...")
            start = time.time()
            if prop != 1:
                edge_set_opt = choose_edge_approx(edge_set, Z_last, edge_coo, edge_ptr, prop)
            else:
                edge_set_opt = edge_set
            print(f"Loop number: {edge_set_opt.shape[0]}")
            loss, fair = edge_loop(edge_set_opt, edge_coo, edge_ptr, AXT, XT, degree, Z_last, self.y, self.train_idx, self.test_idx, self.sens)
            end = time.time()
            print(f"Finish looping, time: {end - start}")

            delta_fair = fair - fair_last
            delta_loss = loss - loss_last
            score = delta_fair - np.abs(np.sum(delta_fair * delta_loss) / np.sum(delta_loss * delta_loss) * delta_loss)
            flip_edge_ind = score.argmax()

            flip_edge = edge_set_opt[flip_edge_ind]
            if adj_self_loop[flip_edge[0], flip_edge[1]] == 0:
                adj_self_loop[flip_edge[0], flip_edge[1]] = 1
                adj_self_loop[flip_edge[1], flip_edge[0]] = 1
                self.adj_atk[flip_edge[0], flip_edge[1]] = 1
                self.adj_atk[flip_edge[1], flip_edge[0]] = 1
                degree[flip_edge[0]] += 1
                degree[flip_edge[1]] += 1
                AX[flip_edge[0]] += X[flip_edge[1]]
                AX[flip_edge[1]] += X[flip_edge[0]]
                neighbour[flip_edge[0]].add(flip_edge[1])
                neighbour[flip_edge[1]].add(flip_edge[0])
                edge_coo = np.hstack([edge_coo[:, :edge_ptr[flip_edge[0]+1]], flip_edge.reshape(2, 1), edge_coo[:, edge_ptr[flip_edge[0]+1]:]])
                edge_ptr[flip_edge[0]+1:] += 1
                edge_coo = np.hstack([edge_coo[:, :edge_ptr[flip_edge[1]+1]], np.array([flip_edge[1], flip_edge[0]]).reshape(2, 1), edge_coo[:, edge_ptr[flip_edge[1]+1]:]])
                edge_ptr[flip_edge[1]+1:] += 1
            else:
                adj_self_loop[flip_edge[0], flip_edge[1]] = 0
                adj_self_loop[flip_edge[1], flip_edge[0]] = 0
                self.adj_atk[flip_edge[0], flip_edge[1]] = 0
                self.adj_atk[flip_edge[1], flip_edge[0]] = 0
                degree[flip_edge[0]] -= 1
                degree[flip_edge[1]] -= 1
                AX[flip_edge[0]] -= X[flip_edge[1]]
                AX[flip_edge[1]] -= X[flip_edge[0]]
                neighbour[flip_edge[0]].remove(flip_edge[1])
                neighbour[flip_edge[1]].remove(flip_edge[0])
                edge_ind = np.where(edge_coo[1, edge_ptr[flip_edge[0]]:edge_ptr[flip_edge[0]+1]] == flip_edge[1])[0][0] + edge_ptr[flip_edge[0]]
                assert edge_coo[0, edge_ind] == flip_edge[0] and edge_coo[1, edge_ind] == flip_edge[1]
                edge_coo = np.delete(edge_coo, edge_ind, axis=1)
                edge_ptr[flip_edge[0]+1:] -= 1
                edge_ind = np.where(edge_coo[1, edge_ptr[flip_edge[1]]:edge_ptr[flip_edge[1]+1]] == flip_edge[0])[0][0] + edge_ptr[flip_edge[1]]
                assert edge_coo[0, edge_ind] == flip_edge[1] and edge_coo[1, edge_ind] == flip_edge[0]
                edge_coo = np.delete(edge_coo, edge_ind, axis=1)
                edge_ptr[flip_edge[1]+1:] -= 1
            neighbor_set = list(neighbour[flip_edge[0]].union(neighbour[flip_edge[1]]))
            X_t = list()
            for u in neighbor_set:
                neighbor_u = list(neighbour[u])
                X_prime = (AX[neighbor_u] / degree[u] / degree[neighbor_u].reshape(-1, 1)).sum(0)
                X_t.append(X_prime)
            X_t = np.array(X_t)
            X_last[neighbor_set] = X_t
            del_filter = np.bitwise_or(edge_set[:, 0]!=flip_edge[0], edge_set[:, 1]!=flip_edge[1])
            edge_set = edge_set[del_filter]

            print(f"Start training {epoch + 1}/{budget}...")
            start = time.time()
            _ = self.train_surrogate(alpha, h, int_num)
            end = time.time()
            print(f"Finish training, time: {end - start}")
            Theta = self.trained_weight
            B = 0
            if self.with_bias:
                B = self.trained_bias
            
            Z_last = (X_last @ Theta + B).reshape(-1)
            AXT = (AX @ Theta).reshape(-1)
            XT = (X @ Theta).reshape(-1)     
            pred = Z_last[self.test_idx] >= 0
            fair_last = np.abs(pred[self.sens[self.test_idx] == 0].mean() - pred[self.sens[self.test_idx] == 1].mean())

            Z_comp = (X_last @ Theta_init + bias_init).reshape(-1)
            pred = Z_comp[self.test_idx] >= 0
            fair_comp = np.abs(pred[self.sens[self.test_idx] == 0].mean() - pred[self.sens[self.test_idx] == 1].mean())
            Z_comp_sm = 1 / (1 + np.exp(-Z_comp))
            loss_comp = -(self.y[self.train_idx] * np.log(Z_comp_sm[self.train_idx]) + (1 - self.y[self.train_idx]) * np.log(1 - Z_comp_sm[self.train_idx])).mean()
            print(f"edge change: {((adj_self_loop - sp.eye(self.N))!=self.adj_ori).sum() / 2}, Delta SP init: {fair_comp}")
            if np.abs(loss_comp - loss_init) <= (loss_p * loss_init):
                best_adj = adj_self_loop.copy()

        self.adj_atk = best_adj - sp.eye(self.N)
        if not os.path.exists('poisoning_structure'):
            os.mkdir('poisoning_structure')
        sp.save_npz('poisoning_structure/'+self.dataset_name+'_p_'+str(p)+'_lp_'+str(loss_p)+'_prop_'+str(prop)+'_alpha_'+str(alpha)+'.npz', self.adj_atk)

