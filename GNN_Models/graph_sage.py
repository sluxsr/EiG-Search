''' 
**********************************
The file is modified based on the implemetation in PyG library.
https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/graph_sage.py

**********************************
'''

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import JumpingKnowledge, global_mean_pool
from GNN_Models.sage_conv import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def fwd(self, x, edge_index, de=None, epsilon=None, edge_weight=None):
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device)
        if de is not None:
            if type(de) == int:
                de = [de]
            for e in de:
                edge_weight[e]=epsilon
                edl, edr = edge_index[0,e], edge_index[1,e]
                rev_de = int((torch.logical_and(edge_index[0]==edr, edge_index[1]==edl)==True).nonzero()[0])
                edge_weight[rev_de]=epsilon
        x = F.relu(self.conv1(x, edge_index[:,edge_weight.bool()]))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index[:,edge_weight.bool()]))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def fwd_base(self, x, edge_index):
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGEWithJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super().__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__