import os.path as osp

import numpy as np

try:
    import cupy
except (ModuleNotFoundError, ImportError) as err:
    pass

try:
    import torch
except (ModuleNotFoundError, ImportError) as err:
    pass

def get_device_list():
    if 'torch' in sys.modules:
        n_gpus = torch.cuda.device_count()
    elif 'cupy' in sys.modules:
        n_gpus = cupy.cuda.runtime.getDeviceCount()
    else:
        raise ImportError("GPU module (CuPy or PyTorch) not found. Please install it before proceeding.")

    return [i for i in range(n_gpus)]


def align_data(data, trj, branch):
    selected_trj = trj[branch == 1]
    inds_sorted_trj = np.argsort(selected_trj)

    selected_exp_data = data[:, branch == 1]
    refined_exp_data = selected_exp_data[:, inds_sorted_trj]

    return refined_exp_data