import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects


fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

for ax in axs.flat:
    ax.set_xlabel('Average sparsity (%)', size=8)
    ax.set_ylabel('Fidelity+', size=8)

    ax.set_xticks(np.arange(0.45, 0.95, step=0.1))  # Set label locations.
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)

common_x = [0.5, 0.6, 0.7, 0.8, 0.9]

ba_2motifs_gin_sa_x = [0.483, 0.604, 0.685, 0.786, 0.863]
ba_2motifs_gin_gradcam_x = [0.504,0.603,0.696,0.798,0.871]
ba_2motifs_gin_our_x = [0.512,0.629,0.691,0.817,0.834,0.9]

ba_2motifs_gin_sa_y = [0.425, 0.485, 0.504, 0.5, 0.499]
ba_2motifs_gin_gradcam_y = [0.63, 0.631, 0.561, 0.499, 0.501]
ba_2motifs_gin_gnnexplainer_y = [0.5156, 0.569, 0.5472, 0.4579, 0.3246]
ba_2motifs_gin_pgexplainer_y = [0.573, 0.6131, 0.5849, 0.4535, 0.2539]
ba_2motifs_gin_subgraphx_y = [0.7716, 0.6327, 0.4628, 0.1722, 0.0635]
ba_2motifs_gin_degree_y = [0.4938, 0.4413, 0.385, 0.317, 0.2418]
ba_2motifs_gin_pgm_explainer_y = [0.515, 0.4611, 0.3911, 0.3351, 0.2406]
ba_2motifs_gin_rg_explainer_y = [0.5269, 0.3283, 0.467, 0.2734, 0.228]
ba_2motifs_gin_rcxplainer_y = [0.5496, 0.564, 0.5029, 0.4905, 0.49]
ba_2motifs_gin_our_y = [0.9995,0.978,0.961,0.866,0.818,0.504]

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

ba_2motifs_gcn_sa_y = [0.499, 0.499, 0.499, 0.499, 0.499]
ba_2motifs_gcn_gradcam_y = [0.499, 0.499, 0.499, 0.5, 0.5]
ba_2motifs_gcn_gnnexplainer_y = [0.499, 0.4957, 0.4915, 0.4499, 0.297]
ba_2motifs_gcn_pgexplainer_y = [0.4841, 0.4402, 0.372, 0.2053, 0.0503]
ba_2motifs_gcn_subgraphx_y = [0.4996, 0.4911, 0.4877, 0.489, 0.1842]
ba_2motifs_gcn_degree_y = [0.4394, 0.4226, 0.2419, 0.1376, 0.0733]
ba_2motifs_gcn_pgm_explainer_y = [0.4754, 0.4221, 0.3774, 0.3201, 0.2254]
ba_2motifs_gcn_rg_explainer_y = [0.4986, 0.4976, 0.4095, 0.2933, 0.1776]
ba_2motifs_gcn_rcexplainer_y = [0.499, 0.4991, 0.499, 0.4971, 0.4773]
ba_2motifs_gcn_our_y = [0.606,0.608,0.614,0.579,0.553,0.556]

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

mutagenicity_gin_sa_y = [0.427, 0.417, 0.395, 0.343, 0.223]
mutagenicity_gin_gradcam_y = [0.426, 0.435, 0.425, 0.426, 0.383]
mutagenicity_gin_gnnexplainer_y = [0.0454, 0.1518, 0.2345, 0.227, 0.1311]
mutagenicity_gin_pgexplainer_y = [0.3343, 0.3275, 0.3129, 0.3038, 0.2716]
mutagenicity_gin_subgraphx_y = [0.2475, 0.2503, 0.2841, 0.2298, 0.1281]
mutagenicity_gin_degree_y = [0.184, 0.1862, 0.2002, 0.1598, 0.0863]
mutagenicity_gin_pgm_explainer_y = [0.2896, 0.2977, 0.2885, 0.2682, 0.2251]
mutagenicity_gin_rg_explainer_y = [0.2109, 0.305, 0.3517, 0.3057, 0.2172]
mutagenicity_gin_rcexplainer_y = [0.3144, 0.3211, 0.3005, 0.2174, 0.1205]
mutagenicity_gin_our_y = [0.752,0.739,0.719,0.702,0.688,0.614]

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

mutagenicity_gcn_sa_y = [0.707, 0.611, 0.53, 0.421, 0.256]
mutagenicity_gcn_gradcam_y = [0.528, 0.506, 0.469, 0.413, 0.277]
mutagenicity_gcn_gnnexplainer_y = [0.7732, 0.6687, 0.5595, 0.4429, 0.2407]
mutagenicity_gcn_pgexplainer_y = [0.7575, 0.6773, 0.6073, 0.5187, 0.3827]
mutagenicity_gcn_subgraphx_y = [0.7481, 0.7167, 0.6479, 0.4931, 0.2567]
mutagenicity_gcn_degree_y = [0.5473, 0.4927, 0.4306, 0.3373, 0.2179]
mutagenicity_gcn_pgm_explainer_y = [0.5587, 0.4855, 0.4024, 0.3317, 0.2178]
mutagenicity_gcn_rg_explainer_y = [0.4565, 0.5828, 0.5586, 0.4385, 0.2559]
mutagenicity_gcn_rcexplainer_y = [0.4327, 0.4445, 0.4355, 0.3714, 0.2713]
mutagenicity_gcn_our_y = [0.917,0.914,0.893,0.872,0.841,0.775]

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

nci1_gin_sa_y = [0.721, 0.685, 0.625, 0.549, 0.406]
nci1_gin_gradcam_y = [0.731, 0.715, 0.691, 0.647, 0.531]
nci1_gin_gnnexplainer_y = [0.7757, 0.7553, 0.7169, 0.6411, 0.4745]
nci1_gin_pgexplainer_y = [0.8558, 0.839, 0.8014, 0.6756, 0.4915]
nci1_gin_subgraphx_y = [0.5858, 0.6106, 0.5917, 0.5242, 0.3627]
nci1_gin_degree_y = [0.448, 0.5523, 0.598, 0.5477, 0.3763]
nci1_gin_pgm_explainer_y = [0.7077, 0.6567, 0.6005, 0.5366, 0.3893]
nci1_gin_rg_explainer_y = [0.4331, 0.5392, 0.5846, 0.5384, 0.398]
nci1_gin_rcexplainer_y = [0.7299, 0.6961, 0.6336, 0.5347, 0.3548]
nci1_gin_our_y = [0.862,0.847,0.842,0.836,0.829,0.801]

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

mutag_gcn_sa_y = [0.759, 0.773, 0.769, 0.71, 0.487]
mutag_gcn_gradcam_y = [0.843, 0.811, 0.77, 0.727, 0.62]
mutag_gcn_gnnexplainer_y = [0.905, 0.9026, 0.8868, 0.8252, 0.559]
mutag_gcn_pgexplainer_y = [0.792, 0.7348, 0.5869, 0.4032, 0.1676]
mutag_gcn_subgraphx_y = [0.8403, 0.8294, 0.7855, 0.6039, 0.4807]
mutag_gcn_degree_y = [0.8646, 0.7957, 0.6368, 0.3473, 0.1317]
mutag_gcn_pgm_explainer_y = [0.8742, 0.8489, 0.8171, 0.662, 0.3808]
mutag_gcn_rg_explainer_y = [0.827, 0.7727, 0.7416, 0.6392, 0.2407]
mutag_gcn_rcexplainer_y = [0.8497, 0.8647, 0.8612, 0.8183, 0.6171]
mutag_gcn_our_y = [0.836,0.874,0.882,0.888,0.793]

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


