import os
import os.path as osp
import sys
from collections import Counter
from itertools import permutations

import numpy as np
import scipy.stats
import statsmodels.sandbox.stats.multicomp
import networkx as nx

droot = osp.abspath("./")

kw = 0.5
fpath_gene_name = osp.join(droot, "expression_dataTuck_node_name.npy")
fpath_binary = osp.join(droot, f"TE_result_matrix.npy")
fpath_ed = osp.join(droot, f"TE_result_matrix.txt")

# fpath_binary = osp.join(droot, "TE_result_matrix_ex02.npy")
# fpath_ed = osp.join(droot, "TE_result_matrix_ex02.txt")

if os.path.isfile(fpath_binary):
    result_matrix = np.load(fpath_binary)
else:
    with open(fpath_ed, "rt") as fin:
        result_matrix = np.loadtxt(fpath_ed, delimiter='\t', dtype=np.float32)
    np.save(fpath_binary, result_matrix)

print(result_matrix.shape)

# ifile = open(fpath_ed)

gene_name = np.load(fpath_gene_name)

pairs = permutations(range(len(gene_name)), 2)
pairs = np.asarray(tuple(pairs))

# tf_name = np.loadtxt("human_tfs.txt", dtype=str)
# gene_tf, inds_gene_tf, _ = np.intersect1d(gene_name, tf_name, return_indices=True)

# Source pick start
tf2ind = {}
ind2tf = {}
with open("mouse_tfs.txt", "rt") as fin:
    for line in fin:
        tf_name = line.strip()
        ix, = np.where(tf_name == gene_name)
        if ix.size > 0:
            ix = int(ix)
            tf2ind[tf_name] = ix
            ind2tf[ix] = tf_name

# Filter the souces with TF list
n_included = 0
n_excluded = 0
pairs_filtered = []
for (i_trg, i_src) in pairs:
    if i_src in ind2tf:
        pairs_filtered.append((i_trg, i_src))
        n_included += 1
    else:
        n_excluded += 1
pairs = np.array(pairs_filtered)
print('pairs shape: ', pairs.shape)
# Source pick end

# Indexing to get the 1D arrays
source = gene_name.T[pairs[:, 1]]
target = gene_name.T[pairs[:, 0]]

te = result_matrix[pairs[:, 0], pairs[:, 1]]

te_zscore = (te - np.mean(te)) / np.std(te)
te_pval = 1 - scipy.stats.norm.cdf(te_zscore)
te_fdr = statsmodels.sandbox.stats.multicomp.multipletests(te_pval, alpha=0.05, method='fdr_bh')

# fdrCutoff=float(sys.argv[1])
fdr_cutoff = 0.01

inds_cutoff = te_fdr[1] < fdr_cutoff  # Get the indices of significant pairs

source_cutoff = source[inds_cutoff]
target_cutoff = target[inds_cutoff]
te_cutoff = te[inds_cutoff]

te_grn = np.stack((source_cutoff, te_cutoff, target_cutoff), axis=1)
# print(te_grn.shape)

# cnts_outdegree = Counter(te_grn[:, 0])
# cnts_outdegree = Counter(te_grn[:, 2])
dg = nx.from_edgelist(te_grn[:, [0, 2]], create_using=nx.DiGraph)
out_degrees = sorted(dg.out_degree, key=lambda x: x[1], reverse=True)

fpath_save = osp.join(droot, f"te_result_grn_{kw}.fdr" + str(fdr_cutoff) + ".sif")
# fpath_save = osp.join(droot, "te_result_grn_ex02.fdr"+str(fdr_cutoff)+".sif")

np.savetxt(fpath_save, te_grn, delimiter='\t', fmt="%s")
print('save te result grn in ', fpath_save)

np.savetxt(fpath_save + ".outdegrees.txt", out_degrees, fmt="%s")
print('save te result outdegrees in ', fpath_save)