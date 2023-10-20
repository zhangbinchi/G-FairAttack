import numpy as np
import scipy.sparse as sp
import torch
import argparse
from torch import optim
from torch.nn import functional as F
import utils
from dataloading import load_data
from surrogate import GCN
import random
import pdb
import os
from sklearn.metrics import roc_auc_score, f1_score



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

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')

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
print(f"G-FairAttack poisoned edge: {(adj != adj_poissoned).sum() / 2}")
adj_poissoned = adj_poissoned.tocoo()
edge_poissoned = np.vstack([adj_poissoned.row, adj_poissoned.col])
edge_poissoned = torch.LongTensor(edge_poissoned).to(device)


def test(victim_model, victim_optimizer, type="Clean"):
    min_val_loss = 1e10
    best_result = {}
    best_epoch = 0
    if type == "Clean":
        edge = edge_clean
    elif type == "Poisson":
        edge = edge_poissoned
    for i in range(args.epochs):
        victim_model.train()
        victim_optimizer.zero_grad()
        output = victim_model(features, edge)
        loss = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))
        loss.backward()
        victim_optimizer.step()

        victim_model.eval()
        output = victim_model(features, edge)
        pred = (output[idx_test] >= 0).float().cpu().numpy()
        best_result['acc'] = utils.accuracy1(output[idx_test], labels[idx_test])
        best_result['roc'] = roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
        best_result['f1'] = f1_score(labels.cpu().numpy()[idx_test], pred)
        best_result['parity'], best_result['equality'] = utils.fair_metric1(output[idx_test], labels.cpu()[idx_test], sens[idx_test])

        if i % 100 == 0:
            parity, equality = utils.fair_metric1(output[idx_test], labels.cpu()[idx_test], sens[idx_test])
            print(f"iteration {i}, loss: {loss.item():.4f}, val_acc: {utils.accuracy1(output[idx_val], labels[idx_val]):.4f}, parity: {parity:.4f}, equality: {equality:.4f}, train_acc: {utils.accuracy1(output[idx_train], labels[idx_train]):.4f}, train_roc_auc: {roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train]):.4f}")

    return best_epoch, best_result


def main():
    best_epoch = dict()
    best_result = dict()
    victim_model = GCN(nfeat, args.nhid, nclass, args.dropout)
    torch.save(victim_model.state_dict(), 'pre_processed/poisoning_gcn_' + args.dataset + '.pt')
    for type in ['Clean', 'Poisson']:
        victim_model.load_state_dict(torch.load('pre_processed/poisoning_gcn_' + args.dataset + '.pt'))
        victim_model = victim_model.to(device)
        victim_optimizer = optim.Adam(victim_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_epoch_type, best_result_type = test(victim_model, victim_optimizer, type)
        best_epoch[type] = best_epoch_type
        best_result[type] = best_result_type
    for type in ['Clean', 'Poisson']:
        print("Test " + type + ":",
              "accuracy: {:.4f}".format(best_result[type]['acc']),
              "roc: {:.4f}".format(best_result[type]['roc']),
              "f1: {:.4f}".format(best_result[type]['f1']),
              "parity: {:.4f}".format(best_result[type]['parity']),
              "equality: {:.4f}".format(best_result[type]['equality']))


if __name__ == '__main__':
    main()
    