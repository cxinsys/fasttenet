import numpy as np
import torch

def calculate_memory_usage(shape, dtype):
    """
    Calculate memory usage for the solve function.

    Parameters:
        shape (tuple): Shape of the main array.
        dtype (numpy.dtype): Data type of the arrays.
        indices (tuple): Batch indices (start, end).

    Returns:
        float: Total memory usage in MB.
    """
    dtype_size = np.dtype(dtype).itemsize

    # Calculate memory for each array
    arr_memory = np.prod(shape) * dtype_size
    batch_arr_memory = np.prod(shape) * dtype_size
    min_arr_memory = np.prod((shape[0], shape[1], shape[1])) * dtype_size
    trimmed_arr_memory = batch_arr_memory
    inds_zeros_memory = np.prod(shape) * dtype_size
    adjusted_inds_memory = np.prod(shape) * dtype_size
    trimmed_pairs_memory = np.prod(shape) * dtype_size
    flattend_index_memory = np.prod(shape) * dtype_size
    tes_memory = np.prod(shape) * dtype_size

    # Total memory usage
    fix = (
            arr_memory +
            inds_zeros_memory +
            adjusted_inds_memory +
            trimmed_pairs_memory +
            flattend_index_memory +
            tes_memory
    )

    other = (batch_arr_memory + min_arr_memory + trimmed_arr_memory)

    # Convert to MB
    fix = fix / (1024 ** 2)
    other = other / (1024 ** 2)

    return fix, other

def get_gpu_memory(gpu_index=0):
    torch.cuda.set_device(gpu_index)  # GPU 선택
    total_memory = torch.cuda.get_device_properties(gpu_index).total_memory

    return 0.9 * (total_memory / (1024 ** 2))

def calculate_batchsize(shape, dtype=np.float32, num_gpus=1, num_ppd=1):
    fix, other = calculate_memory_usage(shape=shape, dtype=dtype)
    gpu_mem = get_gpu_memory(gpu_index=0)

    free_mem = gpu_mem - (fix * num_ppd)

    if free_mem < 0:
        raise ValueError("The number of processors you want to use is too many for the batch size. ")

    mem_per_gpu = np.ceil(other / num_gpus).astype(np.int64)

    num_batch_per_gpu = np.ceil(mem_per_gpu / free_mem).astype(np.int32)

    num_batch = num_batch_per_gpu * num_gpus * num_ppd

    num_genes = shape[0]

    batch_size = np.floor(num_genes / num_batch).astype(np.int64)

    return batch_size