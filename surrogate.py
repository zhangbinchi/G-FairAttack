import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, dropout=0.5, with_bias=True):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass, bias=with_bias)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        return self.W(x)

    def initialize(self):
        self.W.reset_parameters()


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m=None):
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)
        self.body.gc1.reset_parameters()
        self.body.gc2.reset_parameters()
        # self.body.gc3.reset_parameters()
        # if isinstance(m, nn.Linear):
        #     torch.nn.init.xavier_uniform_(m.weight.data)
        #     if m.bias is not None:
        #         m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.dropout = dropout
        self.gc1 = GCNConv(nfeat, nhid, normalize=True)
        self.gc2 = GCNConv(nhid, nhid, normalize=True)
        # self.gc3 = GCNConv(nhid, nhid, normalize=True)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index).relu()
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.gc3(x, edge_index).relu()
        return x
