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
# parser.add_argument('--fp_exp', type=str, dest='fp_exp', required=True)
parser.add_argument('--fp_tf', type=str, dest='fp_tf', required=False, default='None')
parser.add_argument('--fdr', type=float, dest='fdr', required=False, default=0.01)
parser.add_argument('--t_degrees', type=int, dest='dgs', required=False, default=0)
parser.add_argument('--trim_threshold', type=float, dest='trim_threshold', required=False, default=0)

args = parser.parse_args()

droot = osp.dirname(args.fp_rm)

fpath_rm = osp.abspath(args.fp_rm)
# fpath_exp = osp.abspath(args.fp_exp)
fdr = args.fdr
dgs = args.dgs

trim_threshold = args.trim_threshold

result_matrix = np.loadtxt(fpath_rm, delimiter='\t', dtype=str)
gene_name = result_matrix[0][1:]
result_matrix = result_matrix[1:, 1:].astype(np.float32)

print("Number of genes: ", len(result_matrix))

# gene_name = np.load(fpath_exp[:-4] + '_node_name.npy')

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


print('Number of pairs: ', len(pairs))
# Source pick end

# Indexing to get the 1D arrays
source = gene_name.T[pairs[:, 1]]
target = gene_name.T[pairs[:, 0]]

te = result_matrix[pairs[:, 0], pairs[:, 1]]

te_zscore = (te - np.mean(te)) / np.std(te)
te_pval = 1 - scipy.stats.norm.cdf(te_zscore)
te_fdr = statsmodels.sandbox.stats.multicomp.multipletests(te_pval, alpha=0.05, method='fdr_bh')

# fdrCutoff=float(sys.argv[1])

out_cnt = dgs
if out_cnt != 0:
    fdr = 0.0001

while (True):
    inds_cutoff = te_fdr[1] < fdr  # Get the indices of significant pairs

    source_cutoff = source[inds_cutoff]
    target_cutoff = target[inds_cutoff]
    te_cutoff = te[inds_cutoff]

    te_grn = np.stack((source_cutoff, te_cutoff, target_cutoff), axis=1)

    dg = nx.from_edgelist(te_grn[:, [0, 2]], create_using=nx.DiGraph)
    out_degrees = sorted(dg.out_degree, key=lambda x: x[1], reverse=True)
    
    degree_cnt = 0
    
    for degree in out_degrees:
        degree_cnt += degree[1]
    
    if out_cnt == 0:  # Specifying a specific fdr
        print(f"fdr: {fdr}")
        print(f"Degrees: {degree_cnt}")
        break

    if degree_cnt >= out_cnt:
        print(f"fdr: {fdr}")
        print(f"Degrees: {degree_cnt}")
        break

    if fdr <= 0.01:
        fdr += 0.00001
    else:
        fdr += 0.01

# trimming
TF, TFtarget, TFtargetTE, TFtargetIndirect = [], [], [], []
for row in te_grn:
    if row[0] not in TF:
        TF.append(row[0])
        TFtarget.append([row[2]])
        TFtargetTE.append([float(row[1])])
        TFtargetIndirect.append([0])
    else:
        tmp_ind = TF.index(row[0])
        TFtarget[tmp_ind].append(row[2])
        TFtargetTE[tmp_ind].append(float(row[1]))
        TFtargetIndirect[tmp_ind].append(0)

for i in range(len(TF)):
    for j in range(len(TFtarget[i])):
        for k in range(len(TFtarget[i])):
            if j != k and TFtarget[i][j] in TF:
                tmp_inds = TF.index(TFtarget[i][j])
                if TFtarget[i][k] in TFtarget[tmp_inds]:
                    if TFtargetTE[i][k] < min(TFtargetTE[i][j], TFtargetTE[tmp_inds][TFtarget[tmp_inds].index(TFtarget[i][k])]) + trim_threshold:
                        TFtargetIndirect[i][k] = 1

trimmed_te_grn = []
for i in range(len(TF)):
    for j in range(len(TFtarget[i])):
        if TFtargetIndirect[i][j] == 0:
            trimmed_te_grn.append([TF[i], str(TFtargetTE[i][j]), TFtarget[i][j]])

trimmed_te_grn = np.array(trimmed_te_grn, dtype=str)

# outdegree
dg = nx.from_edgelist(te_grn[:, [0, 2]], create_using=nx.DiGraph)
out_degrees = sorted(dg.out_degree, key=lambda x: x[1], reverse=True)

# trimmed_outdegree
dg = nx.from_edgelist(trimmed_te_grn[:, [0, 2]], create_using=nx.DiGraph)
trim_out_degrees = sorted(dg.out_degree, key=lambda x: x[1], reverse=True)

rm_base = osp.basename(fpath_rm)

fpath_save = osp.join(droot, f"{rm_base[:-4]}.byGRN.fdr" + str(fdr) + ".sif")

np.savetxt(fpath_save, te_grn, delimiter='\t', fmt="%s")
print('save grn in ', fpath_save)

fpath_timmed = fpath_save[:-4] + '.trimIndirect' + str(trim_threshold) + '.sif'
np.savetxt(fpath_timmed, trimmed_te_grn, delimiter='\t', fmt="%s")
print('save trimmed grn in ', fpath_timmed)

np.savetxt(fpath_save + ".outdegrees.txt", out_degrees, fmt="%s")
print('save grn outdegrees in ', fpath_save + ".outdegrees.txt")

np.savetxt(fpath_timmed + ".outdegrees.txt", trim_out_degrees, fmt="%s")
print('save trimmed grn outdegrees in ', fpath_timmed + ".outdegrees.txt")