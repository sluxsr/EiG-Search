# EiG-Search: Generating Edge-Induced Subgraphs for GNN Explanation in Linear Time 
[[OpenReivew]](https://openreview.net/forum?id=HO0g6cHVZx) [[arXiv]]()

![image](docs/eig_figure1.png)

## Run Our Code:
> $ python main.py --dataset ba_2motifs --gnn gin --sparsity 0.7 --do_plot 0 --linear_search 1

* *Graph classificiation dataset options*: ba_2motifs, Mutagenicity, MUTAG, NCI1

* *Node classificiation dataset options*: ba_shape, ba_communicity, trid_grid

* *GNN varients*: gin, gcn, sage

* *Sparsity range*: 0.0~1.0 (For graph classificiation)

* *Topk values*： even integers (For node classificiation) E.g., If you want to select top 6 undirected edges, enter 12. It means equivalently top 12 directed edges. 

* *do_plot*: 1 if want to visualize, else 0

* *linear_search*: 1 if turn on the linear-complexity search module, else 0

## Environment:
>CUDA                      11.3<br>
python                    3.9.18<br>
pytorch                   1.11.0          <br>
pytorch-cluster           1.6.0           <br>
pytorch-cuda              11.8            <br>
pytorch-memlab            0.3.0           <br>
pytorch-mutex             1.0             <br>
pytorch-scatter           2.0.9           <br>
pytorch-sparse            0.6.15          <br>
pyg                       2.1.0<br>
numpy                     1.26.3<br>
ninja                     1.10.2              <br>
networkx                  3.2.1<br>

## Cite our paper: 
>@inproceedings{<br>
    lu2024eig,<br>
    title={{EiG-Search}: Generating Edge-Induced Subgraphs for {GNN} Explanation in Linear Time},<br>
    author={Shengyao Lu and Bang Liu and Keith G Mills and Jiao He and Di Niu}, <br>
    booktitle={Forty-first International Conference on Machine Learning},<br>
    year={2024},<br>
}
