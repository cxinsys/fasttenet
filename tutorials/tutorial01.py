import os
import os.path as osp
import argparse

import numpy as np

import fasttenet as fte

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dpath parser')
    parser.add_argument('--fp_exp', type=str, dest='fp_exp', required=True)
    parser.add_argument('--fp_trj', type=str, dest='fp_trj', required=True)
    parser.add_argument('--fp_br', type=str, dest='fp_br', required=True)
    parser.add_argument('--fp_tf', type=str, dest='fp_tf', required=False, default=None)
    parser.add_argument('--sp_rm', type=str, dest='sp_rm', required=False)

    args = parser.parse_args()


    droot = osp.abspath('./')
    # dpath_exp_data = osp.abspath(args.fp_exp)
    dpath_exp_data = osp.abspath(args.fp_exp)
    dpath_trj_data = osp.abspath(args.fp_trj)
    dpath_branch_data = osp.abspath(args.fp_br)
    dpath_tf_data = osp.abspath(args.fp_tf)

    spath_result_matrix = osp.abspath(args.sp_rm)

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

    result_matrix = worker.run(device='gpu', device_ids=[0, 1, 2, 3, 4, 5, 6, 7], batch_size=2 ** 16, kp=0.5,
                               percentile=0, win_length=10, polyorder=3)

    print(result_matrix)