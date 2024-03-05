import os
import os.path as osp
import argparse

import numpy as np

import fasttenet as fte

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dpath parser')
    parser.add_argument('--droot', type=str, dest='droot', required=False, default=osp.abspath('./'))
    parser.add_argument('--fp_exp', type=str, dest='fp_exp', required=True)
    parser.add_argument('--fp_trj', type=str, dest='fp_trj', required=True)
    parser.add_argument('--fp_br', type=str, dest='fp_br', required=True)
    parser.add_argument('--fp_tf', type=str, dest='fp_tf', required=False, default=None)
    parser.add_argument('--sp_rm', type=str, dest='sp_rm', required=False)

    args = parser.parse_args()


    droot = args.droot
    # dpath_exp_data = osp.abspath(args.fp_exp)
    dpath_exp_data = osp.join(droot, args.fp_exp)
    dpath_trj_data = osp.join(droot,args.fp_trj)
    dpath_branch_data = osp.join(droot,args.fp_br)
    dpath_tf_data = osp.join(droot,args.fp_tf)

    spath_result_matrix = osp.join(droot, args.sp_rm)

    # Create worker
    # expression data, trajectory data, branch data path is required
    # tf data path is optional
    # save path is optional
    worker = fte.FastTENET(dpath_exp_data=dpath_exp_data, # Required
                           dpath_trj_data=dpath_trj_data, # Required
                           dpath_branch_data=dpath_branch_data, # Required
                           dpath_tf_data=dpath_tf_data, # Optional
                           spath_result_matrix=spath_result_matrix, # Optional
                           make_binary=True) # Optional, default: False

    result_matrix = worker.run(device='gpu',
                               device_ids=7,
                               batch_size=2 ** 16, # k1 - 2080ti: 2**15, 3090: 2**16 / k3 - 2**14, 2**15
                               num_kernels=1,
                               method='pushing',
                               kp=0.5,
                               percentile=0,
                               win_length=10,
                               polyorder=3,
                               # kw_smooth=True,
                               # data_smooth=True
                               )

    # print(result_matrix)
