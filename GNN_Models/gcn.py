''' 
**********************************
The file is modified based on the implemetation in PyG library.
https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/gcn.py

**********************************
'''

import torch
import torch.nn.functional as F
from torch.nn import Linear

from GNN_Models.gcn_conv import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_adj

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden, normalize=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden, normalize=False))
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
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index))
        x = global_mean_pool(x, batch)
        # x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def get_emb(self, x, edge_index):
        batch = torch.ones(x.shape[0]).to(x.device).type(torch.int64) 
        nodes = list(set(edge_index[0].cpu().tolist()+edge_index[1].cpu().tolist()))
        batch[nodes]=0
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin1(x)[0]
        return x

    def fwd_cam(self, data, edge_weight):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        x = global_mean_pool(x, batch)
        # x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # return F.softmax(x, dim=-1)
        return x

    def fwd_fea(self, x, edge_index, idn=None, idfea=None):
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64)
        w_m = self.conv1.state_dict()['lin.weight'][:,idfea].view(-1,1)
        bias = self.conv1.state_dict()['bias']

        A = to_dense_adj(edge_index)[0]

        mult = max(F.normalize(x, p=2.0, dim=-1)[0])/max(x[0])
        # tmp_x = self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index)
        # offset = tmp_x*((tmp_x<=0).float())*(-1.0/tmp_x.shape[-1])
        _x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index))
        x = F.linear(A[:,idn].view(-1,1)@(x[idn,idfea]*mult).view(1,-1), w_m)
        # x = F.linear(A[:,idn].view(-1,1)@(x[idn,idfea]*mult).view(1,-1), w_m, bias)
        # x=torch.mul(x,(~(torch.min(_x<=0, x<=0))).float())
        x=torch.mul(x,(_x>0).float())
        # x = x+offset

        # print("x", x[20])
        # # print("x+offset", (x+offset)[20])
        # print("tmp_x", tmp_x[20])
        # print("_x", _x[20])
        # # print("offset", offset[20])
        # exit(0)

        for conv in self.convs:
            mult = max(F.normalize(_x, p=2.0, dim=-1)[0])/max(_x[0])
            _x = F.relu(conv(F.normalize(_x, p=2.0, dim=-1), edge_index))
            x = conv(x*mult, edge_index) - conv.bias
            # x=torch.mul(x,(~(torch.min(_x<=0, x<=0))).float())
            x=torch.mul(x,(_x>0).float())
        _x = global_mean_pool(_x, batch)
        _x = F.relu(self.lin1(_x))

        x = global_mean_pool(x, batch)
        x = self.lin1(x)-self.lin1.bias
        # x = torch.mul(x,(~(torch.min(_x<=0, x<=0))).float())
        x=torch.mul(x,(_x>0).float())
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)-self.lin2.bias
        return F.log_softmax(x, dim=-1)

    def fwd_weight(self, x, edge_index, edge_weight=None):
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device)
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        x = global_mean_pool(x, batch)
        # x = global_add_pool(x, batch)
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
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
        # return x

    def fwd_single(self, x, edge_index, de=None, epsilon=1.0):
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        edge_weight = torch.zeros(edge_index.shape[1]).type(torch.float).to(edge_index.device)
        if de is not None:
            edge_weight[de]+=epsilon
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def fwd_base(self, x, edge_index):
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__

class GCN_NC(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden, normalize=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden, normalize=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.softmax(x, dim=-1)
    
    def fwd_eval(self, x, edge_index):
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.softmax(x, dim=-1)

    def fwd_cam(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.softmax(x, dim=-1)

    def fwd(self, x, edge_index, de=None, epsilon=None):
        edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device)
        if de is not None:
            if type(de) == int:
                de = [de]
            for e in de:
                edge_weight[e]=epsilon
                edl, edr = edge_index[0,e], edge_index[1,e]
                rev_de = int((torch.logical_and(edge_index[0]==edr, edge_index[1]==edl)==True).nonzero()[0])
                edge_weight[rev_de]=epsilon
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index, edge_weight=edge_weight))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.softmax(x, dim=-1)

    def fwd_base(self, x, edge_index):
        x = F.relu(self.conv1(F.normalize(x, p=2.0, dim=-1), edge_index))
        for conv in self.convs:
            x = F.relu(conv(F.normalize(x, p=2.0, dim=-1), edge_index))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__