import os.path as osp
import pickle
import glob
import os, json
import numpy as np

import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.utils import degree, dense_to_sparse
from torch_geometric.data import Data, Dataset, InMemoryDataset

from Utils.mol_dataset import MoleculeDataset
from Utils.nlp_dataset import *

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

class NC_Data(Dataset):
    def __init__(self, n_class, dataname):
        path = './data/'+dataname+'.pkl'
        with open(path,'rb') as fin:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pickle.load(fin)
        self.adj = torch.from_numpy(adj)
        self.edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        self.x = torch.from_numpy(features).float()
        self.y = torch.from_numpy(y_train+y_val+y_test)
        self.train_mask = torch.from_numpy(train_mask)
        self.val_mask = torch.from_numpy(val_mask)
        self.test_mask = torch.from_numpy(test_mask)
        self.num_classes = n_class

    def to(self, device):
        self.adj = self.adj.to(device)
        self.edge_index = self.edge_index.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)

    def __getitem__(self, index):
        edge_index = self.edge_index
        x = self.x
        y = self.y
        data = Data(x=x, y=y, edge_index=edge_index)
        return data

    def __len__(self):
        return 1

class BA_Motifs(Dataset):
    def __init__(self, path='./data/ba_2motifs.pkl'):
        with open(path,'rb') as fin:
            adjs,features,labels = pickle.load(fin)
        self.adjs = adjs
        self.features = features
        self.labels = labels
        self.num_classes = 2

    def __getitem__(self, index):
        adj = self.adjs[index]
        edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        x = torch.from_numpy(self.features[index])
        y = torch.argmax(torch.from_numpy(self.labels[index]))
        data = Data(x=x, y=y, edge_index=edge_index)
        return data

    def __len__(self):
        return len(self.labels)

def get_dataset(name, sparse=True, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    if name in ["MUTAG",  "Mutagenicity", "NCI1"]:
        dataset = TUDataset(path, name, cleaned=cleaned)
    elif name == "ba_2motifs":
        dataset = BA_Motifs()
        print("Before detection", sum(dataset.labels))
        return dataset
    elif name == "ba_shape":
        dataset = NC_Data(n_class=4, dataname=name)
        print("Before detection", sum(dataset.y))
        return dataset
    elif name == "tree_grid":
        dataset = NC_Data(n_class=2, dataname=name)
        print("Before detection", sum(dataset.y))
        return dataset
    elif name == "ba_community":
        dataset = NC_Data(n_class=8, dataname=name)
        print("Before detection", sum(dataset.y))
        return dataset
    else: 
        print("Invalid dataset name.")
        exit(0)

    print(dataset)
    dataset.data.edge_attr = None
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset.copy(torch.tensor(indices))

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    print("Before detection", sum(dataset.data.y))
    if name == "Mutagenicity_nh2no2":
        dataset = detect_no2nh2(dataset)
        print("After detection", sum(dataset.data.y))
    return dataset


def get_graph_data(dataset):
    pri = './data/'+dataset+'/'+dataset+'/'+'raw/'+dataset+'_'

    file_edges = pri+'A.txt'
    # file_edge_labels = pri+'edge_labels.txt'
    file_edge_labels = pri+'edge_gt.txt'
    file_graph_indicator = pri+'graph_indicator.txt'
    file_graph_labels = pri+'graph_labels.txt'
    file_node_labels = pri+'node_labels.txt'

    edges = np.loadtxt( file_edges,delimiter=',').astype(np.int32)
    try:
        edge_labels = np.loadtxt(file_edge_labels,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use edge label 0')
        edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

    graph_indicator = np.loadtxt(file_graph_indicator,delimiter=',').astype(np.int32)
    graph_labels = np.loadtxt(file_graph_labels,delimiter=',').astype(np.int32)

    try:
        node_labels = np.loadtxt(file_node_labels,delimiter=',').astype(np.int32)
    except Exception as e:
        print(e)
        print('use node label 0')
        node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

    graph_id = 1
    starts = [1]
    node2graph = {}
    for i in range(len(graph_indicator)):
        if graph_indicator[i]!=graph_id:
            graph_id = graph_indicator[i]
            starts.append(i+1)
        node2graph[i+1]=len(starts)-1
    # print(starts)
    # print(node2graph)
    graphid  = 0
    edge_lists = []
    edge_label_lists = []
    edge_list = []
    edge_label_list = []
    for (s,t),l in list(zip(edges,edge_labels)):
        sgid = node2graph[s]
        tgid = node2graph[t]
        if sgid!=tgid:
            print('edges connecting different graphs, error here, please check.')
            print(s,t,'graph id',sgid,tgid)
            exit(1)
        gid = sgid
        if gid !=  graphid:
            edge_lists.append(edge_list)
            edge_label_lists.append(edge_label_list)
            edge_list = []
            edge_label_list = []
            graphid = gid
        start = starts[gid]
        edge_list.append((s-start,t-start))
        edge_label_list.append(l)

    edge_lists.append(edge_list)
    edge_label_lists.append(edge_label_list)

    # node labels
    node_label_lists = []
    graphid = 0
    node_label_list = []
    for i in range(len(node_labels)):
        nid = i+1
        gid = node2graph[nid]
        # start = starts[gid]
        if gid!=graphid:
            node_label_lists.append(node_label_list)
            graphid = gid
            node_label_list = []
        node_label_list.append(node_labels[i])
    node_label_lists.append(node_label_list)

    return edge_lists, graph_labels, edge_label_lists, node_label_lists


def detect_no2nh2(dataset):

    for i in range(len(dataset)):
        x, y, edge_index = dataset[i].x, dataset[i].y, dataset[i].edge_index
        edges = analyze_graph(x, edge_index)

        n_edges = torch.where(((edges[0]==4) & (edges[1]==1))>0)[0]
        no_edge_index = edge_index[:,n_edges]
        _, counts = torch.unique(no_edge_index[0], return_counts=True, dim=-1)

        # NH2
        nh2_edges = torch.where(((edges[0]==4) & (edges[1]==3))>0)[0]
        nh2_edge_index = edge_index[:,nh2_edges]
        _, counts_nh2 = torch.unique(nh2_edge_index[0], return_counts=True, dim=-1)
        
        if (len(counts)>0 and max(counts)>=2) or (len(counts_nh2)>0 and max(counts_nh2)>=2):
            dataset.data.y[i]=1
        else:
            dataset.data.y[i]=0

    return dataset

def analyze_graph(x, edge_index):

    el = []
    er = []

    for e in edge_index[0]:
        el.append(int(torch.argmax(x[e])))

    for e in edge_index[1]:
        er.append(int(torch.argmax(x[e])))

    return torch.LongTensor([el,er])