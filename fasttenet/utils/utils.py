import os.path as osp

import numpy as np
import cupy as cp

def get_gpu_list():
    n_gpus = cp.cuda.runtime.getDeviceCount()

    return [i for i in range(n_gpus)]

def make_binary(droot, fpath_exp_data):
    exp_data = np.loadtxt(fpath_exp_data, delimiter=',', dtype=str)

    node_name = exp_data[0, 1:]
    exp_data = exp_data[1:, 1:].T.astype(np.float32)

    np.save(osp.join(droot, 'node_name.npy'), node_name)
    np.save(osp.join(droot, 'exp_data.npy'), exp_data)
    print('Binary file save at {}'.format(droot))