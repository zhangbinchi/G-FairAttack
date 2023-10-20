import time
import argparse
import numpy as np

import torch
print(torch.__version__)

import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
import scipy.sparse as sp
# from utils import load_bail, load_credit, load_german, feature_norm, normalize_scipy
from utils import metric_wd, sparse_mx_to_torch_sparse_tensor, normalize_scipy, feature_norm
from dataloading import load_data
from surrogate import GCN
from torch_geometric.utils import dropout_adj, convert
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import ctypes
# ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--cuda_device', type=int, default=0,
                    help='cuda device running on.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--preprocessed_using', type=int, default=0,
                    help='1 and 0 represent utilizing and not utilizing the preprocessed results.')
parser.add_argument('--dataset', type=str, default='bail',
                    help='a dataset from credit, german and bail.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed of the model.')
parser.add_argument('--th', type=float, default=0.1,
                    help='Debiasing threshold.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataset = args.dataset
adj, features, labels, idx_train, idx_val, idx_test, sens = load_data(dataset, False)
print(f'train: {idx_train.shape}, val: {idx_val.shape}, test: {idx_test.shape}')
features = torch.FloatTensor(features)
labels = torch.LongTensor(labels)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
sens = torch.LongTensor(sens)

adj_ori = adj
adj = normalize_scipy(adj)
adj_poissoned = sp.load_npz('Path')

if args.preprocessed_using:
    A_debiased, features = sp.load_npz('pre_processed/'+dataset+'_A_debiased_poisoning.npz'), torch.load("pre_processed/"+dataset+"_X_debiased_poisoning.pt", map_location=torch.device('cpu')).cpu().float()
    threshold_proportion = args.th
    the_con1 = (A_debiased - adj_ori).A
    the_con1 = np.where(the_con1 > np.max(the_con1) * threshold_proportion, 1 + the_con1 * 0, the_con1)
    the_con1 = np.where(the_con1 < np.min(the_con1) * threshold_proportion, -1 + the_con1 * 0, the_con1)
    the_con1 = np.where(np.abs(the_con1) == 1, the_con1, the_con1 * 0)
    A_debiased = adj_ori + sp.coo_matrix(the_con1)
    assert A_debiased.max() == 1
    assert A_debiased.min() == 0
    features = features[:, torch.nonzero(features.sum(axis=0)).squeeze()].detach()
    A_debiased = normalize_scipy(A_debiased)

    A_debiased_poissoned = sp.load_npz('pre_processed/' + dataset + '_A_debiased_poissoned_poisoning.npz')
    the_con1_poissoned = (A_debiased_poissoned - adj_poissoned).A
    the_con1_poissoned = np.where(the_con1_poissoned > np.max(the_con1_poissoned) * threshold_proportion,
                               1 + the_con1_poissoned * 0, the_con1_poissoned)
    the_con1_poissoned = np.where(the_con1_poissoned < np.min(the_con1_poissoned) * threshold_proportion,
                               -1 + the_con1_poissoned * 0, the_con1_poissoned)
    the_con1_poissoned = np.where(np.abs(the_con1_poissoned) == 1, the_con1_poissoned, the_con1_poissoned * 0)
    A_debiased_poissoned = adj_poissoned + sp.coo_matrix(the_con1_poissoned)
    assert A_debiased_poissoned.max() == 1
    assert A_debiased_poissoned.min() == 0
    A_debiased_poissoned = normalize_scipy(A_debiased_poissoned)

if args.preprocessed_using:
    print("****************************After debiasing****************************")
    metric_wd(features, A_debiased, sens, 0.9, 0)
    metric_wd(features, A_debiased, sens, 0.9, 2)
    print("****************************************************************************")
    X_debiased = features.float()
    edge_index = convert.from_scipy_sparse_matrix(A_debiased)[0].cuda()
    edge_index_poissoned = convert.from_scipy_sparse_matrix(A_debiased_poissoned)[0].cuda()
else:
    print("****************************Before debiasing****************************")
    metric_wd(features, adj, sens, 0.9, 0)
    metric_wd(features, adj, sens, 0.9, 2)
    print("****************************************************************************")
    X_debiased = features.float()
    edge_index = convert.from_scipy_sparse_matrix(adj)[0].cuda()
    edge_index_poissoned = convert.from_scipy_sparse_matrix(adj_poissoned)[0].cuda()


# Model and optimizer
model = GCN(nfeat=X_debiased.shape[1], nhid=args.hidden, nclass=labels.max().item(), dropout=args.dropout).float()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
torch.save(model.state_dict(), 'pre_processed/poisoning_edit_'+dataset+'.pt')

if args.cuda:
    torch.cuda.set_device(args.cuda_device)
    model.cuda()
    X_debiased = X_debiased.cuda()
    labels = labels.cuda()
    idx_train = idx_train
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch, result, val_loss, best_epoch, best_val, type):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    edge = edge_index
    if type == "Clean":
        edge = edge_index
    elif type == "Poisson":
        edge = edge_index_poissoned
    output = model(x=X_debiased, edge_index=torch.LongTensor(edge.cpu()).cuda())
    preds = (output.squeeze() > 0).type_as(labels)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())

    loss_train.backward()
    optimizer.step()
    _, _ = fair_metric(preds[idx_train.cpu().numpy()].cpu().numpy(), labels[idx_train.cpu().numpy()].cpu().numpy(), sens[idx_train.cpu().numpy()].cpu().numpy())

    model.eval()
    output = model(x=X_debiased, edge_index=torch.LongTensor(edge.cpu()).cuda())
    preds = (output.squeeze() > 0).type_as(labels)
    loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
    
    if loss_val <= best_val:
        best_val = loss_val
        best_epoch = epoch
    result = test(edge)
    if epoch % 100 == 0:
        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val.cpu().numpy()], output.detach().cpu().numpy()[idx_val.cpu().numpy()])
        f1_val = f1_score(labels[idx_val.cpu().numpy()].cpu().numpy(), preds[idx_val.cpu().numpy()].cpu().numpy())
        accuracy = (preds[idx_train] == labels[idx_train]).sum().item() / idx_train.shape[0]
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(accuracy),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'F1_val: {:.4f}'.format(f1_val),
              'AUC_val: {:.4f}'.format(auc_roc_val),
              'time: {:.4f}s'.format(time.time() - t))

    return result, val_loss, best_epoch, best_val


def test(edge):
    model.eval()
    output = model(x=X_debiased, edge_index=torch.LongTensor(edge.cpu()).cuda())
    preds = (output.squeeze() > 0).type_as(labels)
    auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu().numpy()], output.detach().cpu().numpy()[idx_test.cpu().numpy()])
    f1_test = f1_score(labels[idx_test.cpu().numpy()].cpu().numpy(), preds[idx_test.cpu().numpy()].cpu().numpy())
    accuracy = (preds[idx_test] == labels[idx_test]).sum().item() / idx_test.shape[0]
    parity_test, equality_test = fair_metric(preds[idx_test.cpu().numpy()].cpu().numpy(),
                                               labels[idx_test.cpu().numpy()].cpu().numpy(),
                                               sens[idx_test.cpu().numpy()].cpu().numpy())
    result = dict()
    result['pa'] = parity_test
    result['eq'] = equality_test
    result['f1'] = f1_test
    result['auc'] = auc_roc_test
    result['acc'] = accuracy

    return result


# Train model
t_total = time.time()
val_loss = 1e5
best_result = dict()
type_list = ['Clean', 'Poisson']
best_epoch = 0
epochs = args.epochs
for i, type in enumerate(type_list):
    result = dict()
    model.load_state_dict(torch.load('pre_processed/poisoning_edit_'+dataset+'.pt'))
    best_val = 1e5
    for epoch in tqdm(range(epochs)):
        result, val_loss, best_epoch, best_val = train(epoch, result, val_loss, best_epoch, best_val, type)
    best_result[type] = result
    
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print(f"best epoch: {best_epoch}")
for type in type_list:
    print("Test "+type+":",
            "accuracy: {:.4f}".format(best_result[type]['acc']),
            "roc: {:.4f}".format(best_result[type]['auc']),
            "f1: {:.4f}".format(best_result[type]['f1']),
            "parity: {:.4f}".format(best_result[type]['pa']),
            "equality: {:.4f}".format(best_result[type]['eq']))

# python test_edit_poisoning.py --preprocessed_using 1 --dataset german --hidden 16 --epochs 2000 --lr 1e-3 --th 0.001 --seed 0