import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects

# x_axis = [1, 2, 3]
# y_axis = [0, 1, 2]

# x_axis_ = [1.1, 2.2, 3.3]
# y_axis_ = [0, 1, 2.01]

# plt.plot(x_axis, y_axis)
# plt.plot(x_axis_, y_axis_)figlegend
# plt.title('title name')
# plt.xlabel('x_axis name')
# plt.ylabel('y_axis name')
# plt.show()

fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

# for ax in axs.flat:
#     x_axis = [1, 2, 3]
#     y_axis = [0, 1, 2]

#     x_axis_ = [1.1, 2.2, 3.3]
#     y_axis_ = [0, 1, 2.01]

#     ax.set_title('title name')
#     ax.plot(x_axis, y_axis)
#     ax.plot(x_axis_, y_axis_)
    
#     ax.set_xlabel('x_axis name')
#     ax.set_ylabel('y_axis name')

for ax in axs.flat:
    ax.set_xlabel('Average sparsity (%)', size=8)
    ax.set_ylabel('Overall fidelity', size=8)

    ax.set_xticks(np.arange(0.45, 0.95, step=0.1))  # Set label locations.
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)

common_x = [0.5, 0.6, 0.7, 0.8, 0.9]

ba_2motifs_gin_sa_x = [0.483, 0.604, 0.685, 0.786, 0.863]
ba_2motifs_gin_sa_y = [0.425, 0.485, 0.504, 0.011, -0.087]
ba_2motifs_gin_our_x = [0.512,0.629,0.691,0.817,0.834,0.9]

ba_2motifs_gin_gradcam_x = [0.504,0.603,0.696,0.798,0.871]
ba_2motifs_gin_gradcam_y = [0.61, 0.607,0.519,0.022,0.021]

ba_2motifs_gin_gnnexplainer_y = [0.0065, 0.0644, 0.0467, -0.0418, -0.1753]
ba_2motifs_gin_pgexplainer_y = [0.0802, 0.0905, 0.0489, -0.0835, -0.2503]
ba_2motifs_gin_subgraphx_y = [0.7231, 0.4386, 0.1567, -0.3512, -0.4798]
ba_2motifs_gin_degree_y = [0.0878, -0.0243, -0.0848, -0.2022, -0.2503]
ba_2motifs_gin_pgm_explainer_y = [0.0024, -0.0746, -0.1641, -0.2077, -0.2605]
ba_2motifs_gin_rg_explainer_y = [-0.0696, -0.1721, -0.028, -0.2219, -0.2645]
ba_2motifs_gin_rcxplainer_y = [0.4791, 0.429, 0.2979, -0.1421, -0.0105]
ba_2motifs_gin_our_y = [0.9999,0.9784,0.961,0.866,0.8183,0.457]

axs.flat[0].set_title('(a) BA-2Motifs (GIN)', size=10)
sa, = axs.flat[0].plot(ba_2motifs_gin_sa_x, ba_2motifs_gin_sa_y, color='#DAEE01', linestyle='--', marker="+", linewidth=1, markersize=3)
gradcam, = axs.flat[0].plot(ba_2motifs_gin_gradcam_x, ba_2motifs_gin_gradcam_y, color='orange', linestyle='--', marker="+", linewidth=1, markersize=3)
gnnexplainer, = axs.flat[0].plot(common_x, ba_2motifs_gin_gnnexplainer_y, color='#C9C0BB', linestyle='--', marker="+", linewidth=1, markersize=3)
pgexplainer, = axs.flat[0].plot(common_x, ba_2motifs_gin_pgexplainer_y, color='black', linestyle='--', marker="+", linewidth=1, markersize=3)
subgraphx, = axs.flat[0].plot(common_x, ba_2motifs_gin_subgraphx_y, color='red', linestyle='--', marker="+", linewidth=1, markersize=3)
degree, = axs.flat[0].plot(common_x, ba_2motifs_gin_degree_y, color='green', linestyle='--', marker="+", linewidth=1, markersize=3)
pgm_explainer, = axs.flat[0].plot(common_x, ba_2motifs_gin_pgm_explainer_y, color='pink', linestyle='--', marker="+", linewidth=1, markersize=3)
rg_explainer, = axs.flat[0].plot(common_x, ba_2motifs_gin_rg_explainer_y, color='purple', linestyle='--', marker="+", linewidth=1, markersize=3)
rcexplainer, = axs.flat[0].plot(common_x, ba_2motifs_gin_rcxplainer_y, color='Aqua', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[0].plot(ba_2motifs_gin_our_x, ba_2motifs_gin_our_y, color='blue', linestyle='-', marker='o', markerfacecolor='none', linewidth=1, markersize=3)

# ----------------------------------------------------------------------------------

ba_2motifs_gcn_sa_x = [0.491,0.608,0.698,0.804,0.874]
ba_2motifs_gcn_gradcam_x = [0.49,0.607,0.695,0.803,0.876]
ba_2motifs_gcn_our_x = [0.612,0.668,0.698,0.721,0.8,0.9]

ba_2motifs_gcn_sa_y = [0.499, 0.499, 0.498, 0.435, 0.02]
ba_2motifs_gcn_gradcam_y = [0.5, 0.5, 0.5, 0.444, 0.144]
ba_2motifs_gcn_gnnexplainer_y = [0.001, -0.0033, -0.0075, -0.0491, -0.202]
ba_2motifs_gcn_pgexplainer_y = [0.0972, -0.0152, -0.116, -0.2927, -0.4488]
ba_2motifs_gcn_subgraphx_y = [0.4805, 0.3946, 0.4135, -0.0102, -0.3148]
ba_2motifs_gcn_degree_y = [0.1017, 0.0821, -0.2222, -0.3614, -0.4257]
ba_2motifs_gcn_pgm_explainer_y = [0.0261, -0.055, -0.1149, -0.1784, -0.2736]
ba_2motifs_gcn_rg_explainer_y = [0.0893, -0.0014, -0.0856, -0.2052, -0.3214]
ba_2motifs_gcn_rcexplainer_y = [0.3304, 0.117, 0.009, -0.0019, -0.0217]
ba_2motifs_gcn_our_y = [0.607,0.609,0.615,0.58,0.498,0.077]

axs.flat[3].set_title('(d) BA-2Motifs (GCN)', size=10)
axs.flat[3].plot(ba_2motifs_gcn_sa_x, ba_2motifs_gcn_sa_y, color='#DAEE01', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[3].plot(ba_2motifs_gcn_gradcam_x, ba_2motifs_gcn_gradcam_y, color='orange', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[3].plot(common_x, ba_2motifs_gcn_gnnexplainer_y, color='#C9C0BB', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[3].plot(common_x, ba_2motifs_gcn_pgexplainer_y, color='black', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[3].plot(common_x, ba_2motifs_gcn_subgraphx_y, color='red', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[3].plot(common_x, ba_2motifs_gcn_degree_y, color='green', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[3].plot(common_x, ba_2motifs_gcn_pgm_explainer_y, color='pink', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[3].plot(common_x, ba_2motifs_gcn_rg_explainer_y, color='purple', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[3].plot(common_x, ba_2motifs_gcn_rcexplainer_y, color='Aqua', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[3].plot(ba_2motifs_gcn_our_x, ba_2motifs_gcn_our_y, color='blue', linestyle='-', marker='o', markerfacecolor='none', linewidth=1, markersize=3)

# ----------------------------------------------------------------------------------

mutagenicity_gin_sa_x = [0.497,0.599,0.685,0.793,0.896]
mutagenicity_gin_gradcam_x = [0.504,0.596,0.71,0.794,0.896]
mutagenicity_gin_our_x = [0.627,0.689,0.744,0.794,0.833,0.899]

mutagenicity_gin_sa_y = [0.36, 0.292, 0.213, 0.123, 0.027]
mutagenicity_gin_gradcam_y = [0.349, 0.355, 0.333, 0.304, 0.148]
mutagenicity_gin_gnnexplainer_y = [-0.0018, 0.1808, 0.2854, 0.2767, 0.1624]
mutagenicity_gin_pgexplainer_y = [0.1908, 0.0704, -0.0727, -0.0453, 0.3283]
mutagenicity_gin_subgraphx_y = [0.0197, 0.0417, 0.204, 0.2113, 0.1845]
mutagenicity_gin_degree_y = [-0.0934, -0.0489, 0.0238, 0.0081, -0.0061]
mutagenicity_gin_pgm_explainer_y = [0.0826, 0.0765, 0.092, 0.1215, 0.1781]
mutagenicity_gin_rg_explainer_y = [-0.0217, 0.1799, 0.3194, 0.2457, 0.0727]
mutagenicity_gin_rcexplainer_y = [0.0529, 0.0669, 0.1028, 0.126, 0.0779]
mutagenicity_gin_our_y = [0.866,0.852,0.831,0.813,0.796,0.706]

axs.flat[1].set_title('(b) Mutagenicity (GIN)', size=10)
axs.flat[1].plot(mutagenicity_gin_sa_x, mutagenicity_gin_sa_y, color='#DAEE01', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[1].plot(mutagenicity_gin_gradcam_x, mutagenicity_gin_gradcam_y, color='orange', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[1].plot(common_x, mutagenicity_gin_gnnexplainer_y, color='#C9C0BB', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[1].plot(common_x, mutagenicity_gin_pgexplainer_y, color='black', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[1].plot(common_x, mutagenicity_gin_subgraphx_y, color='red', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[1].plot(common_x, mutagenicity_gin_degree_y, color='green', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[1].plot(common_x, mutagenicity_gin_pgm_explainer_y, color='pink', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[1].plot(common_x, mutagenicity_gin_rg_explainer_y, color='purple', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[1].plot(common_x, mutagenicity_gin_rcexplainer_y, color='Aqua', linestyle='--', marker="+", linewidth=1, markersize=3)
our_line, = axs.flat[1].plot(mutagenicity_gin_our_x, mutagenicity_gin_our_y, color='blue', linestyle='-', marker='o', markerfacecolor='none', linewidth=1, markersize=3)

# ----------------------------------------------------------------------------------

mutagenicity_gcn_sa_x = [0.503,0.618,0.709,0.802,0.893]
mutagenicity_gcn_gradcam_x = [0.5, 0.6, 0.7, 0.8, 0.9]
mutagenicity_gcn_our_x = [0.54,0.575,0.666,0.734,0.789,0.8,0.858,0.9]

mutagenicity_gcn_sa_y = [0.476, 0.306, 0.147, -0.033, -0.275]
mutagenicity_gcn_gradcam_y = [0.24, 0.198, 0.103, 0.004, -0.26]
mutagenicity_gcn_gnnexplainer_y = [-0.0032, -0.1987, -0.3474, -0.4836, -0.6896]
mutagenicity_gcn_pgexplainer_y = [0.4573, 0.3842, 0.3121, 0.2309, -0.1852]
mutagenicity_gcn_subgraphx_y = [0.2753, 0.2231, 0.0745, -0.1927, -0.5739]
mutagenicity_gcn_degree_y = [0.0175, -0.0818, -0.1471, -0.2359, -0.4423]
mutagenicity_gcn_pgm_explainer_y = [0.1438, -0.0007, -0.149, -0.2817, -0.4667]
mutagenicity_gcn_rg_explainer_y = [-0.2603, -0.085, -0.0999, -0.2165, -0.1676]
mutagenicity_gcn_rcexplainer_y = [-0.153, -0.1719, -0.2152, -0.3059, -0.4252]
mutagenicity_gcn_our_y = [0.951,0.94,0.894,0.834,0.742,0.708,0.524,0.277]

axs.flat[4].set_title('(e) Mutagenicity (GCN)', size=10)
axs.flat[4].plot(mutagenicity_gcn_sa_x, mutagenicity_gcn_sa_y, color='#DAEE01', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[4].plot(mutagenicity_gcn_gradcam_x, mutagenicity_gcn_gradcam_y, color='orange', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[4].plot(common_x, mutagenicity_gcn_gnnexplainer_y, color='#C9C0BB', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[4].plot(common_x, mutagenicity_gcn_pgexplainer_y, color='black', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[4].plot(common_x, mutagenicity_gcn_subgraphx_y, color='red', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[4].plot(common_x, mutagenicity_gcn_degree_y, color='green', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[4].plot(common_x, mutagenicity_gcn_pgm_explainer_y, color='pink', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[4].plot(common_x, mutagenicity_gcn_rg_explainer_y, color='purple', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[4].plot(common_x, mutagenicity_gcn_rcexplainer_y, color='Aqua', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[4].plot(mutagenicity_gcn_our_x, mutagenicity_gcn_our_y, color='blue', linestyle='-', marker='o', markerfacecolor='none', linewidth=1, markersize=3)

# ----------------------------------------------------------------------------------

nci1_gin_sa_x = [0.499,0.603,0.698,0.797,0.9]
nci1_gin_gradcam_x = [0.5,0.57,0.698,0.798,0.898]
nci1_gin_our_x = [0.617,0.69,0.717,0.759,0.79,0.878]

nci1_gin_sa_y = [0.13, 0.046, -0.055, -0.202, -0.394]
nci1_gin_gradcam_y = [0.075, 0.017, -0.037, -0.097, -0.222]
nci1_gin_gnnexplainer_y = [0.0016, -0.0296, -0.0681, -0.1597, -0.3419]
nci1_gin_pgexplainer_y = [0.3841, 0.4078, 0.4177, 0.3207, -0.0054]
nci1_gin_subgraphx_y = [-0.089, -0.1637, -0.224, -0.3293, -0.5034]
nci1_gin_degree_y = [-0.4112, -0.3275, -0.2841, -0.3364, -0.4912]
nci1_gin_pgm_explainer_y = [0.0118, -0.0981, -0.1785, -0.2644, -0.3886]
nci1_gin_rg_explainer_y = [-0.4297, -0.3466, -0.3042, -0.3519, -0.4941]
nci1_gin_rcexplainer_y = [0, -0.0792, -0.1539, -0.2625, -0.4616]
nci1_gin_our_y = [0.704,0.619,0.584,0.529,0.49, 0.287]

axs.flat[2].set_title('(c) NCI1 (GIN)', size=10)
axs.flat[2].plot(nci1_gin_sa_x, nci1_gin_sa_y, color='#DAEE01', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[2].plot(nci1_gin_gradcam_x, nci1_gin_gradcam_y, color='orange', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[2].plot(common_x, nci1_gin_gnnexplainer_y, color='#C9C0BB', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[2].plot(common_x, nci1_gin_pgexplainer_y, color='black', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[2].plot(common_x, nci1_gin_subgraphx_y, color='red', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[2].plot(common_x, nci1_gin_degree_y, color='green', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[2].plot(common_x, nci1_gin_pgm_explainer_y, color='pink', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[2].plot(common_x, nci1_gin_rg_explainer_y, color='purple', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[2].plot(common_x, nci1_gin_rcexplainer_y, color='Aqua', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[2].plot(nci1_gin_our_x, nci1_gin_our_y, color='blue', linestyle='-', marker='o', markerfacecolor='none', linewidth=1, markersize=3)

# ----------------------------------------------------------------------------------

mutag_gcn_sa_x = [0.503,0.608,0.703,0.8,0.899]
mutag_gcn_gradcam_x = [0.498,0.606,0.69,0.799,0.905]
mutag_gcn_our_x = [0.499,0.608,0.704,0.8,0.893]

mutag_gcn_sa_y = [0.054, -0.041, -0.123, -0.218, -0.404]
mutag_gcn_gradcam_y = [0.096, -0.03, -0.105, -0.171, -0.233]
mutag_gcn_gnnexplainer_y = [0.0066, 0.0143, 0.0485, 0.0135, -0.161]
mutag_gcn_pgexplainer_y = [-0.0779, -0.1714, -0.3339, -0.5233, -0.7781]
mutag_gcn_subgraphx_y = [0.0773, -0.0276, -0.0928, -0.2248, -0.3822]
mutag_gcn_degree_y = [-0.0162, -0.0614, -0.1881, -0.3502, -0.5206]
mutag_gcn_pgm_explainer_y = [-0.0121, -0.0507, -0.0777, -0.2258, -0.4916]
mutag_gcn_rg_explainer_y = [0.0833, -0.0584, -0.1491, -0.2355, -0.5716]
mutag_gcn_rcexplainer_y = [0.0194, -0.0159, -0.0559, -0.1055, -0.2998]
mutag_gcn_our_y = [0.346,0.181,0.087,0.004,-0.116]

axs.flat[5].set_title('(f) MUTAG (GCN)', size=10)
axs.flat[5].plot(mutag_gcn_sa_x, mutag_gcn_sa_y, color='#DAEE01', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[5].plot(mutag_gcn_gradcam_x, mutag_gcn_gradcam_y, color='orange', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[5].plot(common_x, mutag_gcn_gnnexplainer_y, color='#C9C0BB', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[5].plot(common_x, mutag_gcn_pgexplainer_y, color='black', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[5].plot(common_x, mutag_gcn_subgraphx_y, color='red', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[5].plot(common_x, mutag_gcn_degree_y, color='green', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[5].plot(common_x, mutag_gcn_pgm_explainer_y, color='pink', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[5].plot(common_x, mutag_gcn_rg_explainer_y, color='purple', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[5].plot(common_x, mutag_gcn_rcexplainer_y, color='Aqua', linestyle='--', marker="+", linewidth=1, markersize=3)
axs.flat[5].plot(mutag_gcn_our_x, mutag_gcn_our_y, color='blue', linestyle='-', marker='o', markerfacecolor='none', linewidth=1, markersize=3)

# ----------------------------------------------------------------------------------

plt.figlegend(
    (sa, gradcam, gnnexplainer, pgexplainer, subgraphx, degree, pgm_explainer, rg_explainer, rcexplainer, our_line),
    ('SA (node)', 'Grad-CAM', 'GNNExplainer', 'PGExplainer', 'SubgraphX', 'DEGREE', 'PGM-Explainer', 'RG-Explainer', 'RCExplainer', 'EiG-Search (Ours)'),
    loc='lower center',
    ncol = 5,
    framealpha=0.0,
    prop={'size': 8})

plt.show()

# top=0.95,
# bottom=0.185,
# left=0.08,
# right=0.985,
# hspace=0.45,
# wspace=0.35


