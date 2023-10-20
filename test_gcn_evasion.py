import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
import utils
import argparse
from dataloading import load_data
from surrogate import GCN
from sklearn.metrics import roc_auc_score, f1_score
import random
import pdb
import os


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nhid', type=int, default=128,
                    help='Number of hidden units of the sensitive attribute estimator')
parser.add_argument('--dropout', type=float, default=.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='facebook')
parser.add_argument('--poisoning', action='store_true', default=False,
                    help='Evaluate poisoning attack.')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')

""" Test """
adj, features, labels, idx_train, idx_val, idx_test, sens = load_data(args.dataset, False)
adj = adj.tocsr()
nfeat = features.shape[1]
nclass = labels.max()
victim_model = GCN(nfeat, args.nhid, nclass, args.dropout)
victim_optimizer = optim.Adam(victim_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

features = torch.FloatTensor(features).to(device)
labels = torch.LongTensor(labels).to(device)
victim_model = victim_model.to(device)
adj = adj.tocoo()
edge_clean = np.vstack([adj.row, adj.col])
edge_clean = torch.LongTensor(edge_clean).to(device)

adj_poissoned = sp.load_npz('Path')
print(f"G-FairAttack poisoned edge: {(adj!=adj_poissoned).sum() / 2}")
adj_poissoned = adj_poissoned.tocoo()
edge_poissoned = np.vstack([adj_poissoned.row, adj_poissoned.col])
edge_poissoned = torch.LongTensor(edge_poissoned).to(device)

min_val_loss = 1e5
best_result = {}
best_result_poissoned = {}
best_epoch = 0
for i in range(args.epochs):
    victim_model.train()
    victim_optimizer.zero_grad()
    output = victim_model(features, edge_clean)
    loss = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))
    loss.backward()
    victim_optimizer.step()

    victim_model.eval()
    output = victim_model(features, edge_clean)
    loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float().to(device))
    val_acc = utils.accuracy1(output[idx_test], labels[idx_test])

    if loss_val < min_val_loss:
        min_val_loss = loss_val
        victim_model.eval()
        best_epoch = i

        pred = (output[idx_test] >= 0).float().cpu().numpy()
        best_result['acc'] = utils.accuracy1(output[idx_test], labels[idx_test])
        best_result['roc'] = roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
        best_result['f1'] = f1_score(labels.cpu().numpy()[idx_test], pred)
        best_result['parity'], best_result['equality'] = utils.fair_metric1(output[idx_test], labels.cpu()[idx_test], sens[idx_test])
        best_result['loss'] = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device)).item()

        output_poissoned = victim_model(features, edge_poissoned)
        pred_poissoned = (output_poissoned[idx_test] >= 0).float().cpu().numpy()
        best_result_poissoned['acc'] = utils.accuracy1(output_poissoned[idx_test], labels[idx_test])
        best_result_poissoned['roc'] = test_roc = roc_auc_score(labels.cpu().numpy()[idx_test], output_poissoned.detach().cpu().numpy()[idx_test])
        best_result_poissoned['f1'] = f1_score(labels.cpu().numpy()[idx_test], pred_poissoned)
        best_result_poissoned['parity'], best_result_poissoned['equality'] = utils.fair_metric1(output_poissoned[idx_test], labels.cpu()[idx_test], sens[idx_test])
        best_result_poissoned['loss'] = F.binary_cross_entropy_with_logits(output_poissoned[idx_train], labels[idx_train].unsqueeze(1).float().to(device)).item()

    if i % 100 == 0:
        parity, equality = utils.fair_metric1(output[idx_test], labels.cpu()[idx_test], sens[idx_test])
        print(f"iteration {i}, loss: {loss.item():.4f}, test_acc: {utils.accuracy1(output[idx_test], labels[idx_test]):.4f}, test_roc_auc: {roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test]):.4f}, test_parity: {parity:.4f}, test_equality: {equality:.4f}")

print(f"Best epoch: {best_epoch}")
print("Test Clean:",
            "accuracy: {:.4f}".format(best_result['acc']),
            "roc: {:.4f}".format(best_result['roc']),
            "f1: {:.4f}".format(best_result['f1']),
            "parity: {:.4f}".format(best_result['parity']),
            "equality: {:.4f}".format(best_result['equality']),
            "loss: {:.4f}".format(best_result['loss']))
print("Test GFA:",
            "accuracy: {:.4f}".format(best_result_poissoned['acc']),
            "roc: {:.4f}".format(best_result_poissoned['roc']),
            "f1: {:.4f}".format(best_result_poissoned['f1']),
            "parity: {:.4f}".format(best_result_poissoned['parity']),
            "equality: {:.4f}".format(best_result_poissoned['equality']),
            "loss: {:.4f}".format(best_result_poissoned['loss']))
