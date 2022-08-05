"""
Graph Convolutional Network with LSTM-like layers
"""
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool


class GCN_2LSTM(torch.nn.Module):
    """Model with 2 LSTM cells"""

    def __init__(self, in_dim, out_dim, name="2LSTM"):
        super(GCN_2LSTM, self).__init__()

        self.LSTM1 = GCN_LSTM(in_dim, 64, 32)
        self.LSTM2 = GCN_LSTM(64, 192, 64)

        self.conv1 = GCNConv(192 + in_dim, 256, 1)
        self.conv2 = GCNConv(256, 128, 1)
        self.conv3 = GCNConv(128, 64, 1)
        self.conv4 = GCNConv(64, 32, 1)

        self.convOut = GCNConv(32, out_dim, 1)

    def forward(self, x, adj_t, batch=None):
        h = self.LSTM1(x, adj_t, batch)
        h = self.LSTM2(h, adj_t, batch)

        h = torch.cat([x, h], 1)

        h = self.conv1(h, adj_t).tanh()
        h = self.conv2(h, adj_t).tanh()
        h = self.conv3(h, adj_t).tanh()
        h = self.conv4(h, adj_t).tanh()

        return self.convOut(h, adj_t)


class GCN_LSTM(torch.nn.Module):
    def __init__(self, in_dim, out_dim, opdim):
        super(GCN_LSTM, self).__init__()

        self.conv1 = GCNConv(in_dim, opdim)
        self.inc1 = InceptionL(opdim, opdim)
        self.inc2 = InceptionL(opdim, opdim)

        self.inc3 = InceptionL(opdim, opdim)
        self.inc4 = InceptionL(opdim, opdim)

        self.inc5 = InceptionL(opdim, opdim)
        self.inc6 = InceptionL(opdim, opdim)

        self.conv2 = GCNConv(opdim, out_dim)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)  # x to out_dim shape

        # Forget Gate
        f = self.inc1(x, edge_index, batch)
        f = self.inc2(f, edge_index, batch)
        f = torch.mul(x, f)

        # Input Gate
        i1 = self.inc3(x, edge_index, batch)
        i1 = (self.inc4(i1, edge_index, batch)).sigmoid()

        i2 = self.inc5(x, edge_index, batch)
        i2 = (self.inc6(i2, edge_index, batch)).tanh()

        i = torch.mul(i1, i2)
        i = torch.add(i, f)

        return self.conv2(i, edge_index).tanh()


class InceptionL(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InceptionL, self).__init__()

        self.conv1 = GCNConv(in_dim, in_dim * 2)
        self.conv2 = GCNConv(in_dim * 2, in_dim * 4)

        self.conv3 = GCNConv(in_dim, in_dim * 2)
        self.conv4 = GCNConv(in_dim * 2, in_dim * 4)

        self.conv7 = GCNConv(9 * in_dim, out_dim)

    def forward(self, x, edge_index, batch):
        n_pts = x.size()[0]

        i1 = self.conv1(x, edge_index).tanh()
        i1 = self.conv2(i1, edge_index).tanh()

        i1 = global_max_pool(i1, batch)  # output: B x channels
        i1 = torch.max(i1, 0, keepdim=True)[0]  # output: 1 x channels
        i1 = i1.expand(n_pts, -1)  # expand to all nPts

        i2 = self.conv3(x, edge_index).tanh()
        i2 = self.conv4(i2, edge_index).tanh()

        c = torch.cat((i1, i2, x), dim=1)  # concat

        return self.conv7(c, edge_index).tanh()
