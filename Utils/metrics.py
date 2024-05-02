import numpy as np
import torch
import torch_geometric
import copy

def acc(dataname, node, all_nodes, Hnodes=None, Hedges=None, confid=None):
    accs = []
    if dataname in ['ba_shape', 'ba_community']:
        factor = node//5
        start, end = factor*5, (factor+1)*5
        for i in all_nodes:
            if i>=start and i<end:
                real = 1
            else: real = 0
            if i in Hnodes.tolist():
                pred = 1
            else: pred = 0
            accs.append(int(real==pred))
    elif dataname == 'tree_grid':
        factor = (node-511)//9
        start, end = 511+factor*9, 511+(factor+1)*9
        for i in all_nodes:
            if i>=start and i<end:
                real = 1
            else: real = 0
            if i in Hnodes.tolist():
                pred = 1
            else: pred = 0
            accs.append(int(real==pred))
    elif dataname == 'ba_2motifs':
        for i in all_nodes:
            if i>=20 and i<25:
                real=1
            else: real=0
            if i in Hnodes.tolist():
                pred = 1
            else: pred = 0
            accs.append(int(real==pred))
    return accs

def efidelity(Hedges, map_gnn, data, device):

    nodes = list(set(data.edge_index[0,Hedges].cpu().tolist()+data.edge_index[1,Hedges].cpu().tolist()))
    all_nodes = set(range(data.x.shape[0]))
    left_edges = list(set(range(data.edge_index.shape[1]))-set(Hedges))
    _nodes = list(set(data.edge_index[0,left_edges].cpu().tolist()+data.edge_index[1,left_edges].cpu().tolist()))
    Hnodes = list(all_nodes - set(_nodes))

    all_nodes = list(all_nodes)

    (y, orig_pred, wo_pred, pred), orig_scores, wo_scores, scores = obtain_edge_scores(Hedges, Hnodes, nodes, map_gnn, data, device)
    # (y, orig_pred, wo_pred, pred), orig_scores, wo_scores, scores = obtain_edge_scores(Hedges, None, nodes, map_gnn, data, device)

    orig_prob = torch.exp(orig_scores)
    wo_prob = torch.exp(wo_scores)
    masked_prob = torch.exp(scores)

    # orig_prob, wo_prob, masked_prob = obtain_edge_logits(Hedges, Hnodes, nodes, map_gnn, data, device)

    fid_acc_plus = abs(float(orig_pred==y)-float(wo_pred==y))
    fid_prob_plus = float(orig_prob[y])-float(wo_prob[y])

    fid_acc_minus = abs(float(orig_pred==y)-float(pred==y))
    fid_prob_minus = float(orig_prob[y])-float(masked_prob[y])

    return (fid_acc_minus, fid_prob_minus), (fid_acc_plus, fid_prob_plus)

def nc_efidelity(Hedges, map_gnn, data, device):

    nodes = list(set(data.edge_index[0,Hedges].cpu().tolist()+data.edge_index[1,Hedges].cpu().tolist()))
    all_nodes = set(range(data.x.shape[0]))
    left_edges = list(set(range(data.edge_index.shape[1]))-set(Hedges))
    _nodes = list(set(data.edge_index[0,left_edges].cpu().tolist()+data.edge_index[1,left_edges].cpu().tolist()))
    Hnodes = list(all_nodes - set(_nodes))

    all_nodes = list(all_nodes)

    (y, _,_,_), orig_scores, _, scores = nc_obtain_edge_scores(Hedges, Hnodes, nodes, map_gnn, data, device)

    orig_prob = orig_scores
    masked_prob = scores

    fid_prob_minus = orig_prob[y]-masked_prob[y]

    return (_, fid_prob_minus), (_, _)

def construct_data_egamma(x, edge_index, device, Hedges=None, Hnodes=None):

    H_edges = torch.zeros(edge_index.shape[1]).to(device)
    H_edges[Hedges]=1
    H_edges = H_edges.bool()
    d = {}
    d["x"] = x
    d["edge_index"] = edge_index[:,~H_edges]
    
    d["batch"] = torch.zeros(x.shape[0]).to(device).type(torch.int64) 
    if Hnodes is not None:
        d["batch"][Hnodes]=1

    d = torch_geometric.data.Data.from_dict(d).to(device)
    return d

def construct_data_gamma(x, edge_index, device, Hedges=None, Hnodes=None):

    H_edges = torch.zeros(edge_index.shape[1]).to(device)
    H_edges[Hedges]=1
    H_edges = H_edges.bool()
    d = {}
    d["x"] = x
    d["edge_index"] = edge_index[:,H_edges]

    # d["batch"] = torch.zeros(x.shape[0]).to(device).type(torch.int64) 
    d["batch"] = torch.ones(x.shape[0]).to(device).type(torch.int64) 
    if Hnodes is not None:
        d["batch"][Hnodes]=0
    d = torch_geometric.data.Data.from_dict(d).to(device)
    return d

def obtain_edge_logits(Hedges, Hnodes, nodes, map_gnn, data, device):

    orig_scores = map_gnn.fwd_base(data.x, data.edge_index)[0]

    d = construct_data_gamma(data.x, data.edge_index, device, Hedges, nodes)
    scores = map_gnn.fwd_base(d.x, d.edge_index)[0]

    wo_d = construct_data_egamma(data.x, data.edge_index, device, Hedges, Hnodes)
    wo_scores = map_gnn.fwd_base(wo_d.x, wo_d.edge_index)[0]

    return orig_scores, wo_scores, scores

def obtain_edge_scores(Hedges, Hnodes, nodes, map_gnn, data, device):

    orig_scores = map_gnn(data)[0]
    orig_pred = int(torch.argmax(orig_scores))
    
    d = construct_data_gamma(data.x, data.edge_index, device, Hedges, nodes)
    scores = map_gnn(d)[0]
    pred = int(torch.argmax(scores))

    wo_d = construct_data_egamma(data.x, data.edge_index, device, Hedges, Hnodes)
    wo_scores = map_gnn(wo_d)[0]
    wo_pred = int(torch.argmax(wo_scores))

    return (int(data.y), orig_pred, wo_pred, pred), orig_scores, wo_scores, scores

def nc_obtain_edge_scores(Hedges, Hnodes, nodes, map_gnn, data, device):

    orig_scores = map_gnn(data.x, data.edge_index)

    H_edges = torch.zeros(data.edge_index.shape[1]).to(device)
    H_edges[Hedges]=1
    H_edges = H_edges.bool()
    x = data.x
    edge_index = data.edge_index[:,H_edges]
    scores = map_gnn(x, edge_index)
    # print(edge_index)
    # print(scores[300:310])
    # print(orig_scores[300:310])

    return (data.y.bool(), None, None, None), orig_scores, None, scores

def nfidelity(Hnodes, map_gnn, data, device):

    edges = np.transpose(np.asarray(data.edge_index.cpu()))
    _edges = np.min(sum(np.asarray([edges == nd for nd in Hnodes])), axis=-1)
    Hedges = np.nonzero(_edges)[0]

    (y, orig_pred, wo_pred, pred), orig_scores, wo_scores, scores = obtain_node_scores(None, Hnodes, map_gnn, data, device)

    orig_prob = torch.exp(orig_scores)
    wo_prob = torch.exp(wo_scores)
    masked_prob = torch.exp(scores)

    fid_acc_plus = abs(float(orig_pred==y)-float(wo_pred==y))
    fid_prob_plus = float(orig_prob[y])-float(wo_prob[y])

    fid_acc_minus = abs(float(orig_pred==y)-float(pred==y))
    fid_prob_minus = float(orig_prob[y])-float(masked_prob[y])

    return (fid_acc_minus, fid_prob_minus), (fid_acc_plus, fid_prob_plus)

def construct_data_nbeta(x, edge_index, device, Hedges=None, Hnodes=None):

    d = {}
    d["x"] = copy.deepcopy(x.detach())
    d["x"][Hnodes,:] = 0
    d["edge_index"] = edge_index
    d["batch"] = torch.zeros(x.shape[0]).to(device).type(torch.int64) 
    d["batch"][Hnodes]=1
    d = torch_geometric.data.Data.from_dict(d).to(device)
    return d

def construct_data_beta(x, edge_index, device, Hedges=None, Hnodes=None):

    d = {}
    d["x"] = copy.deepcopy(x.detach())
    for i in range(x.shape[0]):
        if i not in Hnodes:
            d["x"][i,:] = 0
    d["edge_index"] = edge_index
    d["batch"] = torch.ones(x.shape[0]).to(device).type(torch.int64) 
    d["batch"][Hnodes]=0
    d = torch_geometric.data.Data.from_dict(d).to(device)
    return d

def obtain_node_scores(Hedges, Hnodes, map_gnn, data, device):

    orig_scores = map_gnn(data)[0]
    orig_pred = int(torch.argmax(orig_scores))
    
    d = construct_data_beta(data.x, data.edge_index, device, Hedges, Hnodes)
    scores = map_gnn(d)[0]
    pred = int(torch.argmax(scores))
    
    wo_d = construct_data_nbeta(data.x, data.edge_index, device, Hedges, Hnodes)
    wo_scores = map_gnn(wo_d)[0]
    wo_pred = int(torch.argmax(wo_scores))

    return (int(data.y), orig_pred, wo_pred, pred), orig_scores, wo_scores, scores


