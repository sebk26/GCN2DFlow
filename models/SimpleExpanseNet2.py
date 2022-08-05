import os, sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.norm import BatchNorm

sys.path.append("..")
from torch_geometric.nn import GCNConv
from util import gnn_model_summary


class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_dim, in_dim * 2)
        self.conv2 = GCNConv(in_dim * 2, in_dim * 4)
        self.conv3 = GCNConv(in_dim * 4, in_dim * 8)
        self.conv4 = GCNConv(in_dim * 8, in_dim * 16)
        self.conv5 = GCNConv(in_dim * 16, in_dim * 32)
        self.conv6 = GCNConv(in_dim * 32, in_dim * 64)
        self.conv7 = GCNConv(in_dim * 64 + in_dim, in_dim * 32)
        self.conv8 = GCNConv(in_dim * 32, in_dim * 16)
        self.conv9 = GCNConv(in_dim * 16, in_dim * 8)
        self.conv10 = GCNConv(in_dim * 8, in_dim * 4)
        self.conv11 = GCNConv(in_dim * 4, in_dim * 2)
        self.conv12 = GCNConv(in_dim * 2, out_dim)

    def forward(self, x, edge_index, batch):

        h = (self.conv1(x, edge_index)).tanh()
        h = (self.conv2(h, edge_index)).tanh()
        h = (self.conv3(h, edge_index)).tanh()
        h = (self.conv4(h, edge_index)).tanh()
        h = (self.conv5(h, edge_index)).tanh()
        h = (self.conv6(h, edge_index)).tanh()
        h = torch.cat((h, x), dim=1)  # Shortcut
        h = (self.conv7(h, edge_index)).tanh()
        h = (self.conv8(h, edge_index)).tanh()
        h = (self.conv9(h, edge_index)).tanh()
        h = (self.conv10(h, edge_index)).tanh()
        h = (self.conv11(h, edge_index)).tanh()
        h = self.conv12(h, edge_index)

        return h
