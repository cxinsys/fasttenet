import os.path as osp

import numpy as np
import cupy as cp

def get_device_list():
    n_gpus = cp.cuda.runtime.getDeviceCount()

    return [i for i in range(n_gpus)]