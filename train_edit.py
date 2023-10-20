import time
import argparse
import numpy as np
import torch
from victim import EDITS
# from utils import load_bail, load_credit, load_german, sparse_mx_to_torch_sparse_tensor, normalize_scipy, feature_norm
import scipy.sparse as sp
from tqdm import tqdm
from utils import metric_wd, sparse_mx_to_torch_sparse_tensor, normalize_scipy, feature_norm
from dataloading import load_data
import warnings
warnings.filterwarnings('ignore')


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--cuda_device', type=int, default=0,
                    help='cuda device running on.')
parser.add_argument('--dataset', type=str, default='bail',
                    help='a dataset from credit, german and bail.')
parser.add_argument('--epochs', type=int, default=100,  # bail: 100; others: 500.
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.003,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-7,
                    help='Weight decay (L2 loss on parameters).')
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

def binarize(A_debiased, adj_ori, threshold_proportion):

    the_con1 = (A_debiased - adj_ori).A
    the_con1 = np.where(the_con1 > np.max(the_con1) * threshold_proportion, 1 + the_con1 * 0, the_con1)
    the_con1 = np.where(the_con1 < np.min(the_con1) * threshold_proportion, -1 + the_con1 * 0, the_con1)
    the_con1 = np.where(np.abs(the_con1) == 1, the_con1, the_con1 * 0)
    A_debiased = adj_ori + sp.coo_matrix(the_con1)
    assert A_debiased.max() == 1
    assert A_debiased.min() == 0
    A_debiased = normalize_scipy(A_debiased)
    return A_debiased


dataset = args.dataset
adj, features, labels, idx_train, idx_val, idx_test, sens = load_data(dataset, False)
adj_poissoned = sp.load_npz('Path')
features = torch.FloatTensor(features)
labels = torch.LongTensor(labels)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
sens = torch.LongTensor(sens)

features_preserve = features.clone()
features = features / features.norm(dim=0)
adj_preserve = adj
adj = sparse_mx_to_torch_sparse_tensor(adj)
adj_preserve_poissoned = adj_poissoned
adj_poissoned = sparse_mx_to_torch_sparse_tensor(adj_poissoned)
model = EDITS(args, nfeat=features.shape[1], node_num=features.shape[0], nfeat_out=int(features.shape[0]/10), adj_lambda=1, nclass=2, layer_threshold=2, dropout=0.2)  # 3-nba


if args.cuda:
    torch.cuda.set_device(args.cuda_device)
    model.cuda().half()
    adj = adj.cuda().half()
    adj_gfa = adj_poissoned.cuda().half()
    features = features.cuda().half()
    features_preserve = features_preserve.cuda().half()
    labels = labels.cuda().half()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sens = sens.cuda()

A_debiased, X_debiased = adj, features
A_debiased_poissoned = adj_poissoned
val_adv = []
test_adv = []
for epoch in tqdm(range(args.epochs)):
    model.train()
    model.optimize(adj, features, idx_train, sens, epoch, args.lr)
    A_debiased, X_debiased, predictor_sens, show, _ = model(adj, features)
    A_debiased_poissoned = model(adj_poissoned, features)[0]
    positive_eles = torch.masked_select(predictor_sens[idx_val].squeeze(), sens[idx_val] > 0)
    negative_eles = torch.masked_select(predictor_sens[idx_val].squeeze(), sens[idx_val] <= 0)
    loss_val = -(torch.mean(positive_eles) - torch.mean(negative_eles))
    val_adv.append(loss_val.data)

param = model.state_dict()

indices = torch.argsort(param["x_debaising.s"])[:4]
for i in indices:
    features_preserve[:, i] = torch.zeros_like(features_preserve[:, i])
X_debiased = features_preserve
adj1 = sp.csr_matrix(A_debiased.detach().cpu().numpy())
sp.save_npz('pre_processed/'+dataset+'_A_debiased.npz', adj1)
torch.save(X_debiased, "pre_processed/"+dataset+"_X_debiased.pt")
adj_poissoned = sp.csr_matrix(A_debiased_poissoned.detach().cpu().numpy())
sp.save_npz('pre_processed/'+dataset+'_A_debiased_poissoned.npz', adj_poissoned)
print("Preprocessed datasets saved.")

# python train_edit.py --dataset facebook --epochs 1000 --lr 5e-3 --weight_decay 1e-5