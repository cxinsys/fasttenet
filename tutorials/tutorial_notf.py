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
    parser.add_argument('--backend', type=str, dest='backend', required=False, default='gpu')
    parser.add_argument('--num_devices', type=int, dest='num_devices', required=False, default=8)
    parser.add_argument('--batch_size', type=int, dest='batch_size', required=False, default=2**15)
    parser.add_argument('--sp_rm', type=str, dest='sp_rm', required=False)

    args = parser.parse_args()

    droot = args.droot
    dpath_exp_data = osp.join(droot, args.fp_exp)
    dpath_trj_data = osp.join(droot,args.fp_trj)
    dpath_branch_data = osp.join(droot,args.fp_br)

    spath_result_matrix = osp.join(droot, args.sp_rm)

    backend = args.backend
    num_devices = args.num_devices
    batch_size = args.batch_size

    # Create worker
    # expression data, trajectory data, branch data path is required
    # tf data path is optional
    # save path is optional
    worker = fte.FastTENET(dpath_exp_data=dpath_exp_data, # Required
                           dpath_trj_data=dpath_trj_data, # Required
                           dpath_branch_data=dpath_branch_data, # Required
                           spath_result_matrix=spath_result_matrix, # Optional
                           make_binary=True) # Optional, default: False

    result_matrix = worker.run(backend=backend, device_ids=num_devices, procs_per_device=1, batch_size=batch_size,
                               num_kernels=1, binning_method='FSBW-L', kp=0.5)

    # print(result_matrix)
