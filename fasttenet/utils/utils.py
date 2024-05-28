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