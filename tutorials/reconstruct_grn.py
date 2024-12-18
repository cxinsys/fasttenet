import os
import os.path as osp
import sys
import argparse

import numpy as np

import fasttenet as fte

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dpath parser')
    parser.add_argument('--fp_rm', type=str, dest='fp_rm', required=True)
    parser.add_argument('--fp_tf', type=str, dest='fp_tf', required=False, default='None')
    parser.add_argument('--fdr', type=float, dest='fdr', required=False, default=0.01)
    parser.add_argument('--links', type=int, dest='links', required=False, default=0)
    parser.add_argument('--trim_threshold', type=float, dest='trim_threshold', required=False, default=0)
    parser.add_argument('--backend', type=str, dest='backend', required=False, default='gpu')
    parser.add_argument('--device_ids', type=int, dest='device_ids', required=False, default=1)
    parser.add_argument('--batch_size', type=int, dest='batch_size', required=False, default=0)

    args = parser.parse_args()

    droot = osp.dirname(args.fp_rm)
    fpath_rm = osp.abspath(args.fp_rm)
    fdr = args.fdr
    links = args.links
    trim_threshold = args.trim_threshold
    backend = args.backend
    device_ids = args.device_ids
    batch_size = args.batch_size

    result_matrix = np.loadtxt(fpath_rm, delimiter='\t', dtype=str)
    gene_name = result_matrix[0][1:]
    result_matrix = result_matrix[1:, 1:].astype(np.float32)

    print("Number of genes: ", len(result_matrix))

    if args.fp_tf != 'None':
        fpath_tf = osp.abspath(args.fp_tf)
        tf = np.loadtxt(fpath_tf, dtype=str)
    else:
        tf = None

    weaver = fte.NetWeaver(result_matrix=result_matrix,
                           gene_names=gene_name,
                           tfs=tf,
                           fdr=fdr,
                           links=links,
                           is_trimming=True,
                           trim_threshold=trim_threshold,
                           dtype=np.float32
                           )

    grn, trimmed_grn = weaver.run(backend=backend,
                                  device_ids=device_ids,
                                  batch_size=batch_size)

    outdegrees = weaver.count_outdegree(grn)
    trimmed_ods = weaver.count_outdegree(trimmed_grn)

    rm_base = osp.basename(fpath_rm)

    if links != 0:
        fpath_save = osp.join(droot, f"{rm_base[:-4]}.links" + str(links) + ".sif")
    else:
        fpath_save = osp.join(droot, f"{rm_base[:-4]}.fdr" + str(fdr) + ".sif")

    np.savetxt(fpath_save, grn, delimiter='\t', fmt="%s")
    print('save grn in ', fpath_save)

    fpath_timmed = fpath_save[:-4] + '.trimIndirect' + str(trim_threshold) + '.sif'
    np.savetxt(fpath_timmed, trimmed_grn, delimiter='\t', fmt="%s")
    print('save trimmed grn in ', fpath_timmed)

    np.savetxt(fpath_save + ".outdegrees.txt", outdegrees, fmt="%s")
    print('save grn outdegrees in ', fpath_save + ".outdegrees.txt")

    np.savetxt(fpath_timmed + ".outdegrees.txt", trimmed_ods, fmt="%s")
    print('save trimmed grn outdegrees in ', fpath_timmed + ".outdegrees.txt")