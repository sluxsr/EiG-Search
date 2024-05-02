''' 
**********************************
The file is modified based on the implemetation in PyG library.
https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gin.py

**********************************
'''

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import global_mean_pool, global_add_pool
from GNN_Models.gin_conv import GINConv

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(inplace=False),
                Linear(hidden, hidden),
                ReLU(inplace=False),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(inplace=False),
                        Linear(hidden, hidden),
                        ReLU(inplace=False),
                        BN(hidden),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        # x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def get_gemb(self,data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

    def fwd_weight(self, x, edge_index, edge_weight=None):
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).float().to(edge_index.device)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


    def fwd(self, x, edge_index, de=None, epsilon=None, edge_weight=None): 
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).float().to(edge_index.device)
        if de is not None:
            edge_weight[de]=epsilon
            edl, edr = edge_index[0,de], edge_index[1,de]
            rev_de = int((torch.logical_and(edge_index[0]==edr, edge_index[1]==edl)==True).nonzero()[0])
            edge_weight[rev_de]=epsilon
        x = self.conv1(x.float(), edge_index, edge_weight=edge_weight)
        for o, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # return x
        return F.log_softmax(x, dim=-1)

    def fwd_cam(self, data, edge_weight):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
        x = global_mean_pool(x, batch)
        # x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # return F.softmax(x, dim=-1)
        return x

    def fwd_base(self, x, edge_index):
        x, edge_index = x.float(), edge_index
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 

        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
    def fwd_base_other(self, x, edge_index, ie, value):
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        edge_weight = torch.ones(edge_index.shape[1]).float().to(edge_index.device)
        edge_weight[ie]=value
        
        x = self.conv1(x.float(), edge_index, edge_weight=edge_weight)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GIN_NC(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(inplace=False),
                Linear(hidden, hidden),
                ReLU(inplace=False),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(inplace=False),
                        Linear(hidden, hidden),
                        ReLU(inplace=False),
                        BN(hidden),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
        return F.softmax(x, dim=-1)

    def fwd_eval(self, x, edge_index):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.softmax(x, dim=-1)

    def fwd_cam(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
    def fwd(self, x, edge_index, de=None, epsilon=None):
        edge_weight = torch.ones(edge_index.shape[1]).float().to(edge_index.device)
        if de is not None:
            edge_weight[de]=epsilon
            edl, edr = edge_index[0,de], edge_index[1,de]
            rev_de = int((torch.logical_and(edge_index[0]==edr, edge_index[1]==edl)==True).nonzero()[0])
            edge_weight[rev_de]=epsilon
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__



