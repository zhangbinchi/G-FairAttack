import numpy as np
import scipy.sparse as sp
import torch
import argparse
from torch import optim
from torch.nn import functional as F
from GFA import GFA
from dataloading import load_data
from surrogate import GCN
from FAGNN import Fair_Attack
import random
import pdb
import os


parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--poisoning', action='store_true', default=False,
                    help='Conduct poisoning attack.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--p', type=float, default=0.05,
                    help='Proportion of the attack budget.')
parser.add_argument('--lp', type=float, default=0.05,
                    help='Proportion of the utility variation budget.')
parser.add_argument('--alpha', type=float, default=1, help='Weight of the fairness loss term.')
parser.add_argument('--h', type=float, default=0.1, help='Bandwidth of the kernel function.')
parser.add_argument('--prop', type=float, default=1, help='Threshold for fast computation.')
parser.add_argument('--int_num', type=float, default=10000, help='Number of itervals of integral.')
parser.add_argument('--dropout', type=float, default=.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='facebook')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')

""" Attack """
adj, features, labels, idx_train, idx_val, idx_test, sens = load_data(args.dataset, False)
sens = sens.reshape(-1)
print(f"nnode: {features.shape[0]}, nfeat: {features.shape[1]}, nclass: {labels.max() + 1}")
adj = adj.tocsr()
gfa = GFA(adj, features, labels, sens, idx_train, idx_val, idx_test, dropout=args.dropout, lr=args.lr, wd=args.weight_decay, train_iters=args.epochs, device=device, with_bias=True, dataset_name=args.dataset)
if not args.poisoning:
    gfa.evasion_attack(p=args.p, loss_p=args.lp, alpha=args.alpha, h=args.h, int_num=args.int_num, prop=args.prop)
else:
    gfa.poisoning_attack(p=args.p, loss_p=args.lp, alpha=args.alpha, h=args.h, int_num=args.int_num, prop=args.prop)
