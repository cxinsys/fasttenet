import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError) as err:
    print("[WARNING] Cannot use GPU computing based on CuPy")

def parse_device(device):
    if device is None:
        return "cpu", 0

    device = device.lower()
    _device = device
    _device_id = 0

    if ":" in device:
        _device, _device_id = device.split(":")
        _device_id = int(_device_id)

    if _device not in ["cpu", "gpu", "cuda"]:
        raise ValueError("device should be one of 'cpu', "\
                         "'gpu', or 'cuda', not %s" %(device))

    return _device, _device_id

def get_array_module(device):
    _device, _device_id = parse_device(device)

    if "gpu" in _device or "cuda" in _device:
        return CuPyModule(_device, _device_id)
    else:
        return NumpyModule(_device, _device_id)

class ArrayModule:
    def __init__(self, device, device_id):
        self._device = device
        self._device_id = device_id

    def __enter__(self):
        return

    def __exit__(self, *args, **kwargs):
        return

    @property
    def device(self):
        return self._device

    @property
    def device_id(self):
        return self._device_id
    
class NumpyModule(ArrayModule):
    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)

    def array(self, *args, **kwargs):
        return np.array(*args, **kwargs)

    def take(self, *args, **kwargs):
        return np.take(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        return np.repeat(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        return np.concatenate(*args, **kwargs)

    def stack(self, *args, **kwargs):
        return np.stack(*args, **kwargs)

    def unique(self, *args, **kwargs):
        return np.unique(*args, **kwargs)

    def zeros(self, *args, **kwargs):
        return np.zeros(*args, **kwargs)

    def lexsort(self, *args, **kwargs):
        return np.lexsort(*args, **kwargs)

    def arange(self, *args, **kwargs):
        return np.arange(*args, **kwargs)

    def subtract(self, *args, **kwargs):
        return np.subtract(*args, **kwargs)

    def multiply(self, *args, **kwargs):
        return np.multiply(*args, **kwargs)

    def divide(self, *args, **kwargs):
        return np.divide(*args, **kwargs)

    def log2(self, *args, **kwargs):
        return np.log2(*args, **kwargs)

    def bincount(self, *args, **kwargs):
        return np.bincount(*args, **kwargs)

    def asnumpy(self, *args, **kwargs):
        return np.asarray(*args, **kwargs)

    def argsort(self, *args, **kwargs):
        return np.argsort(*args, **kwargs)


class CuPyModule(NumpyModule):
    def __init__(self, device=None, device_id=None):
        super().__init__(device, device_id)

        self._device = cp.cuda.Device()
        self._device.id = self._device_id
        self._device.use()

    def __enter__(self):
        return self._device.__enter__()

    def __exit__(self, *args, **kwargs):
        return self._device.__exit__(*args, **kwargs)

    def array(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.array(*args, **kwargs)

    def take(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.take(*args, **kwargs)

    def repeat(self, array, repeats):
        with cp.cuda.Device(self.device_id):
            repeats = cp.asnumpy(repeats).tolist()
            return cp.repeat(array, repeats)

    def concatenate(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.concatenate(*args, **kwargs)

    def stack(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.stack(*args, **kwargs)

    def unique(self, array, return_counts=False, axis=None):
        with cp.cuda.Device(self.device_id):
            if axis is None:
                return cp.unique(array, return_counts=return_counts)
            else:
                if len(array.shape) != 2:
                    raise ValueError("Input array must be 2D")
                sortarr = array[cp.lexsort(array.T[::-1])]
                mask = cp.empty(array.shape[0], dtype=cp.bool_)
                mask[0] = True
                mask[1:] = cp.any(sortarr[1:]!=sortarr[:-1], axis=1)

                ret = sortarr[mask]

                if not return_counts:
                    return ret

                ret = ret,
                if return_counts:
                    nonzero = cp.nonzero(mask)[0]
                    idx = cp.empty((nonzero.size + 1,), nonzero.dtype)
                    idx[:-1] = nonzero
                    idx[-1] = mask.size
                    ret += idx[1:] - idx[:-1],

                return ret

    def zeros(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.zeros(*args, **kwargs)

    def lexsort(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.lexsort(*args, **kwargs)

    def arange(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.arange(*args, **kwargs)

    def multiply(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.multiply(*args, **kwargs)

    def subtract(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.subtract(*args, **kwargs)

    def divide(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.divide(*args, **kwargs)

    def log2(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.log2(*args, **kwargs)

    def bincount(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.bincount(*args, **kwargs)

    def asnumpy(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.asnumpy(*args, **kwargs)

    def argsort(self, *args, **kwargs):
        with cp.cuda.Device(self.device_id):
            return cp.argsort(*args, **kwargs)