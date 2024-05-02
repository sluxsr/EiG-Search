
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from GNN_Models.gin import GIN, GIN_NC
from GNN_Models.gcn import GCN, GCN_NC
from GNN_Models.graph_sage import GraphSAGE
from torch.utils.data import random_split

def check_task(dataname):
    if dataname in ["ba_shape", "tree_grid", "ba_community"]:
        return "NC"
    else: return "GC"

def load_model(dataname, gnn, n_fea, n_cls):
    if dataname in ["ba_shape"]:
        if gnn == "gin":
            model = GIN_NC(n_fea, n_cls, 3, 32).cuda()
        elif gnn == "gcn":
            model = GCN_NC(n_fea, n_cls, 3, 32).cuda()
    elif dataname in ["tree_grid", "ba_community"]:
        model = GIN_NC(n_fea, n_cls, 3, 64).cuda()
    elif dataname in ["ba_2motifs"]:
        if gnn == "gin":
            model = GIN(n_fea, n_cls, 3, 32).cuda()
        elif gnn == "gcn":
            model = GCN(n_fea, n_cls, 3, 32).cuda()
        elif gnn == "sage":
            model = GraphSAGE(n_fea, n_cls, 3, 32).cuda()
    elif dataname in ["MUTAG", "bbbp"]:
        if gnn == "gin":
            model = GIN(n_fea, n_cls, 3, 128).cuda()
        elif gnn == "gcn":
            model = GCN(n_fea, n_cls, 3, 128).cuda()
    elif dataname in ["Mutagenicity_nh2no2"]:
        model = GIN(n_fea, n_cls, 2, 32).cuda()
    elif dataname in ["Mutagenicity"]:
        if gnn == "gin":
            model = GIN(n_fea, n_cls, 3, 64).cuda()
        elif gnn == "gcn":
            model = GCN(n_fea, n_cls, 3, 64).cuda()
        elif gnn == "sage":
            model = GraphSAGE(n_fea, n_cls, 3, 64).cuda()
    elif dataname in ["NCI1"]:
        if gnn == "gin":
            model = GIN(n_fea, n_cls, 3, 64).cuda()
        elif gnn == "gcn":
            model = GCN(n_fea, n_cls, 3, 64).cuda()
        elif gnn == "sage":
            model = GraphSAGE(n_fea, n_cls, 3, 64).cuda()
    elif dataname in ["Graph-SST2"]:
        model = GCN(n_fea, n_cls, 3, 128).cuda()

    model.load_state_dict(torch.load("saved_models/"+gnn+'_'+dataname+".model"))
    return model

def detect_exp_setting(dataname, dataset):
    if dataname in ["ba_2motifs"]:
        return range(len(dataset))
    elif dataname in ["ba_shape"]:
        return range(300, dataset.x.shape[0])
    elif dataname in ["ba_community"]:
        return list(range(300, 700))+list(range(1000, 1400))
    elif dataname in ["tree_grid"]:
        return range(511, dataset.x.shape[0])
        # return range(511, 800)
    elif dataname in ["MUTAG", "Mutagenicity_nh2no2", "NCI1"]:
        return torch.where(dataset.data.y==1)[0].tolist()
    elif dataname in ["bbbp", "Mutagenicity"]:
        return torch.where(dataset.data.y==0)[0].tolist()
    elif dataname == "Graph-SST2":
        # print(dir(dataset))
        # print(dataset._indices)
        # exit(0)
        num_train = int(0.8 * len(dataset))
        num_eval = int(0.1 * len(dataset))
        num_test = len(dataset) - num_train - num_eval
        _,_, test_dataset = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                            generator=torch.Generator().manual_seed(1234))
        return list(set(test_dataset.indices).intersection(set(torch.where(dataset.data.y==1)[0].tolist())))[:500]
        

def detect_motif_nodes(dataname):
    if dataname in ["ba_shape", "ba_2motifs", "ba_community"]:
        return 5
    elif dataname in ["tree_grid"]:
        return 9
    return None

def show():
    plt.show()
    plt.clf()

def GC_vis_graph(x, edge_index, Hedges=None, good_nodes=None, datasetname=None, edge_color='red'):
    colors = [  'orange',   'red',      'lime',         'green',
                'blue',     'orchid',   'darksalmon',   'darkslategray',
                'gold',     'bisque',   'tan',          'lightseagreen',
                'indigo',   'navy',     'aliceblue',     'violet', 
                'palegreen', 'lightsalmon', 'olive', 'peru',
                'cyan']
    edges = np.transpose(np.asarray(edge_index.cpu()))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    if datasetname not in ["bbbp"]:
        node_label = list(np.asarray(torch.argmax(x, dim=1).cpu()))
    else: 
        node_label = list(np.asarray(x[:,0].cpu()))
    
    max_label = max(node_label) + 1
    nmb_nodes = len(node_label)

    pos = nx.kamada_kawai_layout(G)

    label2nodes = []
    for i in range(max_label):
        label2nodes.append([])
    for i in range(nmb_nodes):
        if i in G.nodes():
            label2nodes[node_label[i]].append(i)

    for i in range(max_label):
        node_filter = []
        for j in range(len(label2nodes[i])):
            node_filter.append(label2nodes[i][j])
        if i < len(colors): cc = colors[i]
        else: cc = colors[-1]
        if edge_color=='red':
            nx.draw_networkx_nodes(G, pos,
                                nodelist=node_filter,
                                node_color=cc,
                                node_size=100)
    
    if edge_color=='red':
        nx.draw_networkx_labels(G, pos, {o:o for o in list(pos.keys())})
        nx.draw_networkx_edges(G, pos, width=2,  edge_color='grey', arrows=False)

    if Hedges is None:
        edges = np.transpose(np.asarray(edge_index.cpu()))
        _edges = np.min(sum(np.asarray([edges == nd for nd in good_nodes])), axis=-1)
        Hedges = np.nonzero(_edges)[0]
    elif good_nodes is None: 
        good_nodes = list(set(edge_index[0,Hedges].tolist() + edge_index[1,Hedges].tolist()))

    nx.draw_networkx_edges(G, pos,
                                edges[Hedges],
                                width=5, edge_color=edge_color, arrows=True)

    plt.axis('off')
    # plt.clf()

def NC_vis_graph(edge_index, y, node_idx, datasetname=None, un_edge_index=None, nodelist=None, H_edges=None):
    y = torch.argmax(y, dim=-1)
    node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
    node_color = ['#FFA500', '#4970C6', '#FE0000', 'green','orchid','darksalmon','darkslategray','gold','bisque','tan','navy','indigo','lime',]
    colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

    node_idx = int(node_idx)
    edges = np.transpose(np.asarray(edge_index.cpu()))
    if un_edge_index is not None: un_edges = np.transpose(np.asarray(un_edge_index.cpu()))
    if nodelist is not None:
        edgelist = [(n_frm, n_to) for (n_frm, n_to) in edges if
                                n_frm in nodelist and n_to in nodelist]
        nodelist = nodelist.tolist()
    elif H_edges is not None:
        if un_edge_index is not None: edgelist = un_edges[H_edges]
        else: edgelist = edges[H_edges]
        nodelist = list(set(list(edgelist[:,0])+list(edgelist[:,1])))
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    if datasetname == "tree_grid":
        G = nx.ego_graph(G, node_idx, radius=3)
    elif datasetname == "ba_community":
        G = nx.ego_graph(G, node_idx, radius=2)
    else:
        G = nx.ego_graph(G, node_idx, radius=3)

    for n in nodelist:
        if n not in G.nodes:
            nodelist.remove(n)
    def remove_unavail(edgelist):
        for i, tup in enumerate(edgelist):
            if tup[0] not in G.nodes or tup[1] not in G.nodes:
                edgelist = np.delete(edgelist, i, axis=0)
                return edgelist, i
        return edgelist, len(edgelist)
    edgelist, i = remove_unavail(edgelist)
    while i != len(edgelist):
        edgelist, i = remove_unavail(edgelist)

    pos = nx.kamada_kawai_layout(G) # calculate according to graph.nodes()
    pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}
    colors = [colors[pp] for pp in list(G.nodes)]

    nx.draw_networkx_nodes(G, pos,
                            nodelist=list(G.nodes()),
                            node_color=colors,
                            node_size=100)
    if isinstance(colors, list):
        list_indices = int(np.where(np.array(G.nodes()) == node_idx)[0])
        node_idx_color = colors[list_indices]
    else:
        node_idx_color = colors

    nx.draw_networkx_nodes(G, pos=pos,
                            nodelist=[node_idx],
                            node_color=node_idx_color,
                            node_size=400)

    nx.draw_networkx_edges(G, pos, width=1, edge_color='grey', arrows=False)
    if nodelist is not None or H_edges is not None:
        nx.draw_networkx_edges(G, pos=pos_nodelist,
                            edgelist=edgelist, width=2,
                            edge_color='red',
                            arrows=False)

    labels = {o:o for o in list(G.nodes)}
    nx.draw_networkx_labels(G, pos,labels)
    
    # plt.axis('off')

def NLP_vis_graph(x, edge_index, nodelist, words, Hedges=None, figname=None):
    edges = np.transpose(np.asarray(edge_index.cpu()))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    node_label = list(np.asarray(torch.argmax(x, dim=1).cpu()))

    pos = nx.kamada_kawai_layout(G)
    words_dict = {i: words[i] for i in G.nodes}

    # nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_size=300)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='grey')

    if nodelist is not None:
        pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
        nx.draw_networkx_nodes(G, pos_coalition,
                                nodelist=nodelist,
                                node_color='yellow',
                                node_shape='o',
                                node_size=600)
    if Hedges is None:
        edgelist = [(n_frm, n_to) for (n_frm, n_to) in G.edges()
                    if n_frm in nodelist and n_to in nodelist]
        nx.draw_networkx_edges(G, pos=pos_coalition, edgelist=edgelist, width=5, edge_color='red')
    else: 
        nx.draw_networkx_edges(G, pos=pos, edgelist=edges[Hedges], width=5, edge_color='red')

    
    
    nx.draw_networkx_labels(G, pos, words_dict)

    plt.axis('off')
    plt.show()




def find_thres(scos):
    last_sc, last_diff = None, 0.0
    init=scos[0]
    for p, sc in enumerate(scos):
        if last_sc is None:
            last_sc = sc
        if p>2 and (2.1*sc < init or last_sc > 2*sc or sc < last_diff*1.5 or (last_sc >1.5 *sc and last_diff>0.0 and last_sc-sc > last_diff*3)):
        # if p>1 and (last_sc > 2*sc or sc < last_diff*1.5 or (last_sc >1.5 *sc and last_diff>0.0 and last_sc-sc > last_diff*3)):
            # print(f'sc: {sc}, last_sc: {last_sc}, lst diff:{last_diff}')
            return p
        last_diff = last_sc-sc
        last_sc = sc
    return p


