import os, sys
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.norm import BatchNorm

sys.path.append("..")
from torch_geometric.nn import GCNConv
from util import gnn_model_summary


class GCN6(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN6, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_dim, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv6 = GCNConv(128, 256)
        self.conv7 = GCNConv(256 + in_dim, 512)
        self.conv7a = GCNConv(512, 1024)
        self.conv7b = GCNConv(1024, 2048)
        self.conv8 = GCNConv(2048 + in_dim, 1024)
        self.conv8a = GCNConv(1024, 512)
        self.conv8b = GCNConv(512, 256)
        self.conv9 = GCNConv(256, 128)
        self.conv10 = GCNConv(128, 64)
        self.conv11 = GCNConv(64, 32)
        self.conv12 = GCNConv(32, out_dim)

    def forward(self, x, edge_index, batch):

        h = (self.conv1(x, edge_index)).tanh()
        h = (self.conv2(h, edge_index)).tanh()
        h = (self.conv3(h, edge_index)).tanh()
        h = (self.conv6(h, edge_index)).tanh()
        h = torch.cat((h, x), dim=1)  # Residual Connection
        h = (self.conv7(h, edge_index)).tanh()
        h = (self.conv7a(h, edge_index)).tanh()
        h = (self.conv7b(h, edge_index)).tanh()
        h = torch.cat((h, x), dim=1)  # Residual Connection
        h = (self.conv8(h, edge_index)).tanh()
        h = (self.conv8a(h, edge_index)).tanh()
        h = (self.conv8b(h, edge_index)).tanh()
        h = (self.conv9(h, edge_index)).tanh()
        h = (self.conv10(h, edge_index)).tanh()
        h = (self.conv11(h, edge_index)).tanh()
        h = self.conv12(h, edge_index)

        return h
