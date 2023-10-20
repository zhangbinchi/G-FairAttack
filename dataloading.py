import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
import os
import dgl
import networkx as nx
from typing import Dict, Tuple
import pickle
import random
from os.path import join, dirname, realpath
import csv
import pickle as pkl
import pdb


def feature_norm(features):
    min_values = features.min(0)
    max_values = features.max(0)
    return 2 * (features - min_values) / (max_values - min_values) - 1


def mx_to_torch_sparse_tensor(sparse_mx, is_sparse=False, return_tensor_sparse=True):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not is_sparse:
        sparse_mx=sp.coo_matrix(sparse_mx)
    else:
        sparse_mx=sparse_mx.tocoo()
    if not return_tensor_sparse:
        return sparse_mx

    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def process_facebook(return_tensor_sparse=True):
    edges_file=open('./dataset/facebook/facebook/107.edges')
    edges=[]
    for line in edges_file:
        edges.append([int(one) for one in line.strip('\n').split(' ')])

    feat_file=open('./dataset/facebook/facebook/107.feat')
    feats=[]
    for line in feat_file:
        feats.append([int(one) for one in line.strip('\n').split(' ')])

    feat_name_file = open('./dataset/facebook/facebook/107.featnames')
    feat_name = []
    for line in feat_name_file:
        feat_name.append(line.strip('\n').split(' '))
    names={}
    for name in feat_name:
        if name[1] not in names:
            names[name[1]]=name[1]
        if 'gender' in name[1]:
            print(name)

    print(feat_name)
    feats=np.array(feats)

    node_mapping={}
    for j in range(feats.shape[0]):
        node_mapping[feats[j][0]]=j

    feats=feats[:,1:]

    print(feats.shape)
    #for i in range(len(feat_name)):
    #    print(i, feat_name[i], feats[:,i].sum())

    sens=feats[:,264]
    labels=feats[:,220]

    feats=np.concatenate([feats[:,:264],feats[:,266:]],-1)

    feats=np.concatenate([feats[:,:220],feats[:,221:]],-1)

    edges=np.array(edges)
    #edges=torch.tensor(edges)
    #edges=torch.stack([torch.tensor(one) for one in edges],0)
    print(len(edges))

    node_num=feats.shape[0]
    adj=np.zeros([node_num,node_num])


    for j in range(edges.shape[0]):
        adj[node_mapping[edges[j][0]],node_mapping[edges[j][1]]]=1

    # idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
    # idx_val=list(set(list(range(node_num)))-set(idx_train))
    # idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
    # idx_val=list(set(idx_val)-set(idx_test))

    import random
    random.seed(2022)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:int(0.5 * len(label_idx_0))],
                          label_idx_1[:int(0.5 * len(label_idx_1))])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.7 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.7 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.7 * len(label_idx_0)):], label_idx_1[int(0.7 * len(label_idx_1)):])
    
    # idx_train=np.random.choice(list(range(node_num)),int(p_train*node_num),replace=False)
    # idx_test=list(set(list(range(node_num)))-set(idx_train))

    if return_tensor_sparse:
        features = torch.FloatTensor(feats)
        sens = torch.FloatTensor(sens)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)
        features=torch.cat([features,sens.unsqueeze(-1)],-1)
    else:
        features = np.hstack([feats, sens.reshape(-1, 1)])
    adj=mx_to_torch_sparse_tensor(adj,return_tensor_sparse=return_tensor_sparse)
    return adj, features, labels, idx_train, idx_val, idx_test, sens


def process_german_bail_credit(dataset_name, return_tensor_sparse=True):
    def feature_norm(features):
        min_values = features.min(0)
        max_values = features.max(0)
        return 2 * (features - min_values) / (max_values - min_values) - 1

    def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/",
                    label_number=6000):
        from scipy.spatial import distance_matrix

        # print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('Single')

        # build relationship
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=int).reshape(edges_unordered.shape)

        print(len(edges))

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = adj + sp.eye(adj.shape[0])

        features = np.array(features.todense())
        # features = torch.FloatTensor(np.array(features.todense()))
        # labels = torch.LongTensor(labels)

        import random
        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.7 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.7 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.7 * len(label_idx_0)):], label_idx_1[int(0.7 * len(label_idx_1)):])

        sens = idx_features_labels[sens_attr].values.astype(int)
        # sens = torch.FloatTensor(sens)
        # idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        # idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test

    def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="./dataset/bail/", label_number=100):
        # print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)


        # build relationship

        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')


        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=int).reshape(edges_unordered.shape)
        print(len(edges))

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        # adj = adj + sp.eye(adj.shape[0])
        features = np.array(features.todense())
        # features = torch.FloatTensor(np.array(features.todense()))
        # labels = torch.LongTensor(labels)

        import random
        random.seed(20)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.7 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.7 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.7 * len(label_idx_0)):], label_idx_1[int(0.7 * len(label_idx_1)):])

        sens = idx_features_labels[sens_attr].values.astype(int)
        # sens = torch.FloatTensor(sens)
        # idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        # idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test

    def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="./dataset/german/",
                    label_number=100):
        # print('Loading {} dataset from {}'.format(dataset, path))
        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        header.remove('OtherLoansAtStore')
        header.remove('PurposeOfLoan')

        # Sensitive Attribute
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values
        labels[labels == -1] = 0
        print(f"num_label: {labels.shape}")

        idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=int).reshape(edges_unordered.shape)

        print(len(edges))

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # adj = adj + sp.eye(adj.shape[0])

        features = np.array(features.todense())
        # features = torch.FloatTensor(np.array(features.todense()))
        # labels = torch.LongTensor(labels)

        import random
        random.seed(2022)
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.7 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.7 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.7 * len(label_idx_0)):], label_idx_1[int(0.7 * len(label_idx_1)):])
        # idx_train = np.append(label_idx_0[:int(p_train * len(label_idx_0))], label_idx_1[:int(p_train * len(label_idx_1))])
        # idx_test = np.append(label_idx_0[int(p_train * len(label_idx_0)):], label_idx_1[int(p_train * len(label_idx_1)):])

        sens = idx_features_labels[sens_attr].values.astype(int)
        # sens = torch.FloatTensor(sens)
        # idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        # idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, edges, sens, idx_train, idx_val, idx_test

    if dataset_name=='german':
        adj, features, labels, edges, sens, idx_train, idx_val, idx_test =load_german('german')
    elif dataset_name=='recidivism':
        adj, features, labels, edges, sens, idx_train, idx_val, idx_test =load_bail('bail')
    elif dataset_name=='credit':
        adj, features, labels, edges, sens, idx_train, idx_val, idx_test =load_credit('credit')
    else:
        raise NotImplementedError
    #node_num=features.shape[0]
    #adj=adj.todense()
    #idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
    #idx_val=list(set(list(range(node_num)))-set(idx_train))
    #idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
    #idx_val=list(set(idx_val)-set(idx_test))

    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    # labels = torch.LongTensor(labels)
    features=feature_norm(features)
    # adj=mx_to_torch_sparse_tensor(adj, is_sparse=True, return_tensor_sparse=return_tensor_sparse)
    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_pokec(dataset, label_number=1000):  # 1000
    """Load data"""

    if dataset == 1:
        edges = np.load('dataset/pokec_dataset/region_job_1_edges.npy')
        features = np.load('dataset/pokec_dataset/region_job_1_features.npy')
        labels = np.load('dataset/pokec_dataset/region_job_1_labels.npy')
        sens = np.load('dataset/pokec_dataset/region_job_1_sens.npy')
    else:
        edges = np.load('dataset/pokec_dataset/region_job_2_2_edges.npy')
        features = np.load('dataset/pokec_dataset/region_job_2_2_features.npy')
        labels = np.load('dataset/pokec_dataset/region_job_2_2_labels.npy')
        sens = np.load('dataset/pokec_dataset/region_job_2_2_sens.npy')

    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0, :], edges[1, :])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    print(adj.sum())
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])
    print(adj.sum())

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    print("Ratio observations:")
    print(label_idx_0.shape)
    print(label_idx_1.shape)
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    # features = torch.FloatTensor(np.array(features))
    # labels = torch.LongTensor(labels)
    # sens = torch.LongTensor(sens)
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_data(dataset_name, return_tensor_sparse=True):

    if dataset_name=='facebook':
        return process_facebook(return_tensor_sparse)
    elif dataset_name == 'pokec_z':
        return load_pokec(1, label_number=1000)
    elif dataset_name == 'pokec_n':
        return load_pokec(2, label_number=1000)
    elif dataset_name=='german' or dataset_name=='recidivism' or dataset_name=='credit':
        return process_german_bail_credit(dataset_name, return_tensor_sparse)
