import cupy as cp

def get_gpu_list():
    n_gpus = cp.cuda.runtime.getDeviceCount()

    return [i for i in range(n_gpus)]