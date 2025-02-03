import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects


fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

for ax in axs.flat:
    ax.set_xlabel('Average sparsity (%)', size=8)
    ax.set_ylabel('Fidelity-', size=8)

    ax.set_xticks(np.arange(0.45, 0.95, step=0.1))  # Set label locations.
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)

common_x = [0.5, 0.6, 0.7, 0.8, 0.9]

ba_2motifs_gin_sa_x = [0.483, 0.604, 0.685, 0.786, 0.863]
ba_2motifs_gin_gradcam_x = [0.504,0.603,0.696,0.798,0.871]
ba_2motifs_gin_our_x = [0.512,0.629,0.691,0.817,0.834,0.9]

ba_2motifs_gin_sa_y = [0, 0, 0, 0.489, 0.586]
ba_2motifs_gin_gradcam_y = [0.02, 0.024, 0.042, 0.477, 0.48]
ba_2motifs_gin_gnnexplainer_y = [0.5091, 0.5046, 0.5005, 0.4997, 0.4999]
ba_2motifs_gin_pgexplainer_y = [0.4928, 0.5226, 0.536, 0.537, 0.5042]
ba_2motifs_gin_subgraphx_y = [0.0485, 0.1941, 0.3061, 0.5234, 0.5433]
ba_2motifs_gin_degree_y = [0.406, 0.4656, 0.4698, 0.5192, 0.4921]
ba_2motifs_gin_pgm_explainer_y = [0.5126, 0.5357, 0.5552, 0.5428, 0.5011]
ba_2motifs_gin_rg_explainer_y = [0.5965, 0.5004, 0.495, 0.4953, 0.4925]
ba_2motifs_gin_rcxplainer_y = [0.0705, 0.135, 0.205, 0.6326, 0.5005]
ba_2motifs_gin_our_y = [-0.0004,-0.0004,0,0,-0.0003,0.047]

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

ba_2motifs_gcn_sa_y = [0, 0, 0.001, 0.064, 0.479]
ba_2motifs_gcn_gradcam_y = [-0.001, -0.001, -0.001, 0.056, 0.356]
ba_2motifs_gcn_gnnexplainer_y = [0.498, 0.499, 0.499, 0.499, 0.499]
ba_2motifs_gcn_pgexplainer_y = [0.3869, 0.4554, 0.488, 0.498, 0.4991]
ba_2motifs_gcn_subgraphx_y = [0.0191, 0.0965, 0.0742, 0.4992, 0.499]
ba_2motifs_gcn_degree_y = [0.3377, 0.3405, 0.4641, 0.499, 0.499]
ba_2motifs_gcn_pgm_explainer_y = [0.4493, 0.4771, 0.4923, 0.4985, 0.499]
ba_2motifs_gcn_rg_explainer_y = [0.4093, 0.499, 0.4951, 0.4985, 0.499]
ba_2motifs_gcn_rcexplainer_y = [0.1686, 0.3821, 0.49, 0.499, 0.499]
ba_2motifs_gcn_our_y = [-0.001,-0.001,-0.001,-0.001,0.055,0.479]

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

mutagenicity_gin_sa_y = [0.067, 0.125, 0.182, 0.22, 0.196]
mutagenicity_gin_gradcam_y = [0.077, 0.08, 0.092, 0.122, 0.235]
mutagenicity_gin_gnnexplainer_y = [0.0472, -0.029, -0.0509, -0.0497, -0.0313]
mutagenicity_gin_pgexplainer_y = [0.1435, 0.2571, 0.3856, 0.3491, -0.0567]
mutagenicity_gin_subgraphx_y = [0.2278, 0.2086, 0.0801, 0.0185, -0.0564]
mutagenicity_gin_degree_y = [0.2774, 0.2351, 0.1764, 0.1517, 0.0924]
mutagenicity_gin_pgm_explainer_y = [0.207, 0.2212, 0.1965, 0.1467, 0.047]
mutagenicity_gin_rg_explainer_y = [0.2326, 0.1251, 0.0323, 0.06, 0.1445]
mutagenicity_gin_rcexplainer_y = [0.2615, 0.2542, 0.1977, 0.0914, 0.0426]
mutagenicity_gin_our_y = [-0.114,-0.113,-0.112,-0.111,-0.108,-0.092]

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
mutagenicity_gcn_our_x = [0.54,0.575,0.666,0.734,0.789,0.858]

mutagenicity_gcn_sa_y = [0.231, 0.305, 0.383, 0.454, 0.531]
mutagenicity_gcn_gradcam_y = [0.288, 0.308, 0.366, 0.409, 0.537]
mutagenicity_gcn_gnnexplainer_y = [0.7764, 0.8674, 0.9069, 0.9265, 0.9303]
mutagenicity_gcn_pgexplainer_y = [0.3002, 0.2931, 0.2952, 0.2878, 0.5679]
mutagenicity_gcn_subgraphx_y = [0.4728, 0.4936, 0.5734, 0.6858, 0.8306]
mutagenicity_gcn_degree_y = [0.5298, 0.5745, 0.5777, 0.5732, 0.6602]
mutagenicity_gcn_pgm_explainer_y = [0.4149, 0.4862, 0.5514, 0.6134, 0.6845]
mutagenicity_gcn_rg_explainer_y = [0.7168, 0.6678, 0.6585, 0.655, 0.4235]
mutagenicity_gcn_rcexplainer_y = [0.5857, 0.6164, 0.6507, 0.6773, 0.6965]
mutagenicity_gcn_our_y = [-0.034,-0.026,-0.001,0.038,0.099,0.251]

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

nci1_gin_sa_y = [0.591, 0.639, 0.68, 0.751, 0.8]
nci1_gin_gradcam_y = [0.656, 0.698, 0.728, 0.744, 0.753]
nci1_gin_gnnexplainer_y = [0.7741, 0.7849, 0.785, 0.8008, 0.8164]
nci1_gin_pgexplainer_y = [0.4717, 0.4312, 0.3837, 0.3549, 0.4969]
nci1_gin_subgraphx_y = [0.6748, 0.7743, 0.8157, 0.8535, 0.8661]
nci1_gin_degree_y = [0.8592, 0.8798, 0.8821, 0.8841, 0.8675]
nci1_gin_pgm_explainer_y = [0.6959, 0.7548, 0.779, 0.801, 0.7779]
nci1_gin_rg_explainer_y = [0.8628, 0.8858, 0.8888, 0.8903, 0.8921]
nci1_gin_rcexplainer_y = [0.7299, 0.7753, 0.7875, 0.7972, 0.8164]
nci1_gin_our_y = [0.158,0.228,0.258,0.307,0.339,0.514]

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

mutag_gcn_sa_y = [0.705, 0.814, 0.892, 0.928, 0.891]
mutag_gcn_gradcam_y = [0.747, 0.841, 0.875, 0.898, 0.853]
mutag_gcn_gnnexplainer_y = [0.8984, 0.8883, 0.8383, 0.8017, 0.72]
mutag_gcn_pgexplainer_y = [0.8699, 0.9062, 0.9208, 0.9265, 0.9457]
mutag_gcn_subgraphx_y = [0.763, 0.857, 0.8783, 0.8287, 0.8629]
mutag_gcn_degree_y = [0.8808, 0.8571, 0.8249, 0.6975, 0.6523]
mutag_gcn_pgm_explainer_y = [0.8863, 0.8996, 0.8948, 0.8878, 0.8724]
mutag_gcn_rg_explainer_y = [0.7437, 0.8311, 0.8907, 0.8747, 0.8123]
mutag_gcn_rcexplainer_y = [0.8303, 0.8806, 0.9171, 0.9238, 0.9169]
mutag_gcn_our_y = [0.49,0.693,0.795,0.884,0.909]

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


