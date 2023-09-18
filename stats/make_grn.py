import os
import os.path as osp
import sys
from collections import Counter
from itertools import permutations
import argparse

import numpy as np
import scipy.stats
import statsmodels.sandbox.stats.multicomp
import networkx as nx


parser = argparse.ArgumentParser(description='dpath parser')
parser.add_argument('--fp_rm', type=str, dest='fp_rm', required=True)
parser.add_argument('--fp_nn', type=str, dest='fp_nn', required=True)
parser.add_argument('--fp_tf', type=str, dest='fp_tf', required=False, default='None')

args = parser.parse_args()

droot = osp.dirname(args.fp_rm)

kw = 0.5
fpath_rm = osp.abspath(args.fp_rm)
fpath_nn = osp.abspath(args.fp_nn)

# fpath_binary = osp.join(droot, "TE_result_matrix_ex02.npy")
# fpath_ed = osp.join(droot, "TE_result_matrix_ex02.txt")


result_matrix = np.loadtxt(fpath_rm, delimiter='\t', dtype=np.float32)

print(result_matrix.shape)

# ifile = open(fpath_ed)

gene_name = np.load(fpath_nn)

pairs = permutations(range(len(gene_name)), 2)
pairs = np.asarray(tuple(pairs))

if args.fp_tf!='None':
    # Source pick start
    tf2ind = {}
    ind2tf = {}
    with open(osp.abspath(args.fp_tf), "rt") as fin:
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

fpath_save = osp.join(droot, f"te_result_grn.fdr" + str(fdr_cutoff) + ".sif")
# fpath_save = osp.join(droot, "te_result_grn_ex02.fdr"+str(fdr_cutoff)+".sif")

np.savetxt(fpath_save, te_grn, delimiter='\t', fmt="%s")
print('save te result grn in ', fpath_save)

np.savetxt(fpath_save + ".outdegrees.txt", out_degrees, fmt="%s")
print('save te result outdegrees in ', fpath_save)