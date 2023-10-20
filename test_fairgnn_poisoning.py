import time
import argparse
import numpy as np
import scipy.sparse as sp
import dgl
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataloading import load_data
from victim import FairGNN
from utils import accuracy, accuracy1
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score


def fair_metric(output, idx):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity, equality


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units of the sensitive attribute estimator')
parser.add_argument('--dropout', type=float, default=.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=4,
                    help='The hyperparameter of alpha')
parser.add_argument('--beta', type=float, default=0.01,
                    help='The hyperparameter of beta')
parser.add_argument('--model', type=str, default="GAT",
                    help='the type of model GCN/GAT')
parser.add_argument('--dataset', type=str, default='pokec_n')
parser.add_argument('--num-hidden', type=int, default=64,
                    help='Number of hidden units of classifier.')
parser.add_argument("--num-heads", type=int, default=1,
                    help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--attn-drop", type=float, default=.0,
                    help="attention dropout")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--acc', type=float, default=0.688,
                    help='the selected FairGNN accuracy on val would be at least this high')
parser.add_argument('--roc', type=float, default=0.745,
                    help='the selected FairGNN ROC score on val would be at least this high')
parser.add_argument('--sens_number', type=int, default=200,
                    help="the number of sensitive attributes")
parser.add_argument('--label_number', type=int, default=500,
                    help="the number of labels")

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

dataset = args.dataset
adj, features, labels, idx_train, idx_val, idx_test, sens = load_data(dataset, False)
idx_sens_train = idx_train
adj = adj.tocsr()
adj_poissoned = sp.load_npz('Path')
G = dgl.from_scipy(adj)
G = dgl.add_self_loop(G)
G_poissoned = dgl.from_scipy(adj_poissoned)
G_poissoned = dgl.add_self_loop(G_poissoned)

features = torch.FloatTensor(features)
labels = torch.LongTensor(labels)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
sens = torch.LongTensor(sens).squeeze()
idx_sens_train = torch.LongTensor(idx_sens_train)


model = FairGNN(nfeat=features.shape[1], args=args)
torch.save(model.state_dict(), 'pre_processed/poisoning_fairgnn_'+dataset+'.pt')
if args.cuda:
    model.cuda()
    G = G.to(torch.device('cuda'))
    G_poissoned = G_poissoned.to(torch.device('cuda'))
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sens = sens.cuda()
    idx_sens_train = idx_sens_train.cuda()


def test(victim_model, type):
    t_total = time.time()
    best_result = {}
    best_fair = 100
    best_val = 0
    best_epoch = 0
    if type == "Clean":
        graph = G
    elif type == "Poisson":
        graph = G_poissoned
    for epoch in range(args.epochs):
        victim_model.train()
        _ = victim_model.optimize(graph, features, labels, idx_train, idx_val, sens)
        cov = victim_model.cov
        cls_loss = victim_model.cls_loss
        adv_loss = victim_model.adv_loss
        victim_model.eval()
        output = victim_model(graph, features)
        acc_val = accuracy1(output[idx_val], labels[idx_val])
        acc_test = accuracy1(output[idx_test], labels[idx_test])
        roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
        parity,equality = fair_metric(output, idx_test)
        acc_train = accuracy1(output[idx_train], labels[idx_train])

        best_epoch = epoch
        best_result['acc'] = acc_test.item()
        best_result['roc'] = roc_test
        best_result['parity'] = parity
        best_result['equality'] = equality
        
        if epoch % 100 == 0:
            print("=================================")
            print('Epoch: {:04d}'.format(epoch+1),
                    'cov: {:.4f}'.format(cov.item()),
                    'cls: {:.4f}'.format(cls_loss.item()),
                    'adv: {:.4f}'.format(adv_loss.item()),
                    'acc_val: {:.4f}'.format(acc_val.item()))
            print("Test:",
                    "accuracy: {:.4f}".format(acc_test.item()),
                    "roc: {:.4f}".format(roc_test),
                    "parity: {:.4f}".format(parity),
                    "equality: {:.4f}".format(equality))
            print("Train:", "accuracy: {:.4f}".format(acc_train.item()))

    return best_epoch, best_result


def main():
    best_epoch = dict()
    best_result = dict()
    for type in ['Clean', 'Poisson']:
        model.load_state_dict(torch.load('pre_processed/poisoning_fairgnn_'+dataset+'.pt'))
        best_epoch_type, best_result_type = test(model, type)
        best_epoch[type] = best_epoch_type
        best_result[type] = best_result_type
    for type in ['Clean', 'Poisson']:
        print("Test "+type+":",
                "accuracy: {:.4f}".format(best_result[type]['acc']),
                "roc: {:.4f}".format(best_result[type]['roc']),
                "parity: {:.4f}".format(best_result[type]['parity']),
                "equality: {:.4f}".format(best_result[type]['equality']))


if __name__ == '__main__':
    main()

# python test_fairgnn_poisoning.py --dataset german --hidden 16 --model GCN --num-hidden 16 --lr 1e-4 --seed 0 --alpha 1 --beta 10 --epochs 5000