import argparse
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import time

from Utils.utils import check_task, load_model, detect_exp_setting, GC_vis_graph, NC_vis_graph, show
from Utils.metrics import efidelity

from Utils.datasets import get_dataset

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataname = args.dataset
    task_type = check_task(dataname)
    dataset = get_dataset(dataname)
    n_fea, n_cls = dataset.num_features, dataset.num_classes
    explain_ids = detect_exp_setting(dataname, dataset)
    gnn_model = load_model(dataname, args.gnn, n_fea, n_cls)
    gnn_model.eval()
    print(f"GNN Model Loaded. {dataname}, {task_type}. \nnum of samples to explain: {len(explain_ids)}")
    
    Fidelities = []
    neg_fids = []
    A_sparsities, Times = [], []

    if task_type == "GC":
        loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

        for i, d in enumerate(loader): 
            if i in explain_ids: 
                # if i<500: continue
                epsilon, sparsity = args.epsilon, args.sparsity
                d = d.to(device)
                logits = gnn_model(d)[0]
                if torch.argmax(logits) != int(d.y): continue

                start = time.time()
                x, edge_index = d.x, d.edge_index
                e_mots = []
                for e in range(edge_index.shape[1]):
                    fina = gnn_model.fwd(x, edge_index, de=e, epsilon=epsilon)[0]
                    ress = (logits - fina).cpu().detach().numpy()[int(d.y)]
                    e_mots.append(ress)
                e_mots = torch.tensor(np.array(e_mots).T).to(device)

                num_edges = max(2, int(edge_index.shape[1]*(1.0-sparsity)))
                econfi, Hedges = torch.topk(e_mots, edge_index.shape[1], dim=-1)[0].cpu().detach().numpy(), torch.topk(e_mots, num_edges, dim=-1)[1].cpu().detach().numpy()

                if args.linear_search>0:
                    diffs = []
                    for l in range(1, len(Hedges), 2):
                        f_neg, f_pos = efidelity(Hedges[:l+3], gnn_model, d, device)
                        diff = f_pos[1] - f_neg[1]
                        # diff = f_pos[1]
                        diffs.append(diff)
                        # if args.do_plot:
                        #     print(d.edge_index[:,Hedges[:l+3]])
                        #     print(diff,"\n")
                    best_index = diffs.index(max(diffs))
                    Hedges = Hedges[:2*(best_index+2)]

                Times.append(time.time()-start)
                
                f_neg, f_pos = efidelity(Hedges, gnn_model, d, device)
                Fidelities.append(f_pos[1])
                neg_fids.append(f_neg[1])
                A_sparsities.append(1.0-float(len(Hedges)/d.edge_index.shape[1]))
                print(i, sum(neg_fids)/float(len(neg_fids)+1e-13), sum(Fidelities)/float(len(Fidelities)+1e-13), sum(A_sparsities)/float(len(A_sparsities)+1e-13))
                if args.do_plot: print(i, int(torch.argmax(logits)), int(d.y), f_neg[1],f_pos[1])
                
                if args.do_plot:
                    print("econfi", econfi)
                    print(edge_index[:,Hedges],"Hedges\n")
                    GC_vis_graph(x, edge_index, Hedges=Hedges, good_nodes=None, datasetname=dataname)
                    show()
    
        print(f'Avg time: {sum(Times)/float(len(Times))}')
        print(f"Fidelity-: {sum(neg_fids)/float(len(neg_fids)+1e-13)}")
        print(f"Fidelity+: {sum(Fidelities)/float(len(Fidelities)+1e-13)}")
        print(f"Actual avg sparsity: {sum(A_sparsities)/float(len(A_sparsities)+1e-13)}")
    
    elif task_type == "NC":

        epsilon, topk = args.epsilon, args.topk
        dataset.to(device)
        x, edge_index, y = dataset.x, dataset.edge_index, dataset.y
        logits = gnn_model(x, edge_index)
        gnn_preds = torch.argmax(logits, dim=-1)

        start = time.time()
        e_mots = []
        for e in range(edge_index.shape[1]):
            fina = gnn_model.fwd(x, edge_index, de=e, epsilon=epsilon)
            ress = torch.norm(logits - fina, dim=-1).cpu().detach().numpy()
            e_mots.append(ress)
        e_mots = torch.tensor(np.array(e_mots).T).to(device)
        after = time.time()
        print(f'Avg time: {(after-start)/float(x.shape[0])} s')

        num_edges = 50
        (confidence, Hedges) = torch.topk(e_mots, num_edges, dim=-1)
        confidence = confidence.cpu().detach().numpy()
        Hedges = Hedges.cpu().detach().numpy()

        for i in explain_ids: 
            if torch.argmax(y[i], dim=-1) != gnn_preds[i]: continue
            if args.do_plot>0:
                print(f'confidence: {confidence[i]}')
                print(f'Hedges: {edge_index[:,Hedges[i]]}')
                NC_vis_graph(edge_index=edge_index, y=y, datasetname=dataname, node_idx=i, H_edges=Hedges[i][:topk])
                show()

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ba_2motifs')
    parser.add_argument('--gnn', type=str, default='gin')
    parser.add_argument('--sparsity', type=float, default=0.7)
    parser.add_argument('--topk', type=int, default=14)
    parser.add_argument('--do_plot', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0) # No need to change this
    parser.add_argument('--linear_search', type=int, default=1)

    
    return parser.parse_args()

if __name__ == "__main__":

    args = build_args()
    main(args)
    print("done")