import time
import math
from itertools import permutations
import multiprocessing
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np
from scipy.signal import savgol_filter

from fasttenet.data import load_exp_data, load_time_data
from fasttenet.tenet import TE
from fasttenet.utils import get_gpu_list

class FastTENET(object):
    def __init__(self,
                 device=None,
                 device_ids=None,
                 dpath_exp_data=None,
                 dpath_trj_data=None,
                 dpath_branch_data=None,
                 dpath_tf_data=None,
                 spath_result_matrix=None,
                 batch_size=None,
                 kp=0.5,
                 percentile=0,
                 win_length=10,
                 polyorder=3,
                 make_binary=False
                 ):

        self._kp = kp
        self._percentile = percentile
        self._win_length = win_length
        self._polyorder = polyorder
        self._batch_size = batch_size

        self._device = device
        self._device_ids = device_ids

        if not dpath_exp_data:
            raise ValueError("Expression data should be refined")
        if not dpath_trj_data:
            raise ValueError("Trajectory should be refined")
        if not dpath_branch_data:
            raise ValueError("Branch data should be refined")

        self._node_name, self._exp_data = load_exp_data(dpath_exp_data, make_binary)
        self._trajectory = load_time_data(dpath_trj_data, dtype=np.float32)
        self._branch = load_time_data(dpath_branch_data, dtype=np.int32)

        self._tf = None

        if dpath_tf_data is not None:
            self._tf = np.loadtxt(dpath_tf_data, dtype=str)

        self._spath_result_matrix = spath_result_matrix

        self._refined_exp_data = None
        self._bin_arr = None
        self._result_matrix = None

    def save_result_matrix(self, spath_result_matrix=None):
        if spath_result_matrix is None:
            if self._spath_result_matrix is None:
                raise ValueError("Save path should be refined")
            spath_result_matrix = self._spath_result_matrix

        if self._result_matrix is None:
            raise ValueError("Result matrix should be refined")

        np.savetxt(spath_result_matrix, self._result_matrix, delimiter='\t', fmt='%8f')
        print("Save result matrix: {}".format(spath_result_matrix))

    # data refining

    def refine_data(self):
        if self._refined_exp_data is None:
            selected_trj = self._trajectory[self._branch == 1]
            inds_sorted_trj = np.argsort(selected_trj)

            selected_exp_data = self._exp_data[:, self._branch == 1]
            self._refined_exp_data = selected_exp_data[:, inds_sorted_trj]

        return self._refined_exp_data

    # calculate kernel width

    def kernel_width(self, kp=None, percentile=None):
        arr = self.refine_data()

        if percentile > 0:
            arr2 = arr.copy()
            arr2.sort(axis=1)

            i_beg = int(arr2.shape[1] / 100 * percentile)

            std = np.std(arr2[:, i_beg:-i_beg], axis=1, ddof=1)
        else:
            std = np.std(arr, axis=1, ddof=1)

        kw = kp * std
        kw[kw == 0] = 1
        return kw

    # binning

    def create_binned_array(self,
                            kp=None,
                            percentile=None,
                            win_length=None,
                            polyorder=None,
                            dtype=np.int32):

        if not kp:
            kp = self._kp

        if not percentile:
            percentile = self._percentile

        if not win_length:
            win_length = self._win_length

        if not polyorder:
            polyorder = self._polyorder

        if self._bin_arr is None:
            arr = self.refine_data()

            kw = self.kernel_width(kp, percentile)

            arr = savgol_filter(arr, win_length, polyorder)

            mins = np.min(arr, axis=1)
            # maxs = np.max(arr, axis=1)

            self._bin_arr = arr.copy()
            self._bin_arr = (arr.T - mins) // kw
            self._bin_arr = self._bin_arr.T.astype(dtype)

        return self._bin_arr

    # multiprocessing worker(calculate tenet)

    def work(self,
             device=None,
             device_ids=None,
             batch_size=None,
             kp=None,
             percentile=None,
             win_length=None,
             polyorder=None,
             ):

        if not device:
            if not self._device:
                self._device = device = "cpu"
            device = self._device

        if not device_ids:
            if not self._device_ids:
                if 'gpu' or 'cuda' in device:
                    self._device_ids = get_gpu_list()
                else:
                    self._device_ids = device_ids = [0]
            device_ids = self._device_ids

        if not batch_size:
            if not self._batch_size:
                raise ValueError("batch size should be refined")
            batch_size = self._batch_size

        if not kp:
            kp = self._kp

        if not percentile:
            percentile = self._percentile

        if not win_length:
            win_length = self._win_length

        if not polyorder:
            polyorder = self._polyorder


        if self._bin_arr is None:
            arr = self.create_binned_array(kp=kp, percentile=percentile, win_length=win_length, polyorder=polyorder)

        pairs = []
        if self._tf is not None:
            _, inds_source, _ = np.intersect1d(self._node_name, self._tf, return_indices=True)

            for ix_t in range(len(self._node_name)):
                for ix_s in inds_source:
                    if ix_t==ix_s:
                        continue
                    pairs.append((ix_t, ix_s))
        else:
            pairs = permutations(range(len(arr)), 2)

        pairs = np.asarray(tuple(pairs), dtype=np.int32)
        n_pairs = len(pairs)

        tmp_rm = np.zeros((len(arr), len(arr)), dtype=np.float32)

        n_gpus = len(device_ids)
        n_subpairs = math.ceil(n_pairs / n_gpus)

        multiprocessing.set_start_method('spawn', force=True)
        shm = shared_memory.SharedMemory(create=True, size=tmp_rm.nbytes)
        np_shm = np.ndarray(tmp_rm.shape, dtype=tmp_rm.dtype, buffer=shm.buf)
        np_shm[:] = tmp_rm[:]

        sem = Semaphore()

        processes = []
        t_beg_batch = time.time()
        if "cpu" in device:
            print("[CPU device selected]")
            print("[Num. Pairs: {}, Batch Size: {}]".format(n_pairs, batch_size))
        elif "gpu" in device:
            print("[GPU device selected]")
            print("[Num. {}S: {}, Num. Pairs: {}, Num. GPU_Pairs: {}, Batch Size: {}]".format(device.upper(), n_gpus, n_pairs,
                                                                                              n_subpairs, batch_size))

        for i, i_beg in enumerate(range(0, n_pairs, n_subpairs)):
            i_end = i_beg + n_subpairs

            device_name = device + ":" + str(device_ids[i])
            # print("tenet device: {}".format(device_name))
            te = TE(device=device_name)

            _process = Process(target=te.solve, args=(batch_size,
                                                              pairs[i_beg:i_end],
                                                              arr,
                                                              shm.name,
                                                              np_shm,
                                                              sem))
            processes.append(_process)
            _process.start()

        for _process in processes:
            _process.join()

        print("Total processing elapsed time {}sec.".format(time.time() - t_beg_batch))

        self._result_matrix = np_shm.copy()

        shm.close()
        shm.unlink()

        if self._result_matrix is not None:
            self.save_result_matrix(self._spath_result_matrix)

        return self._result_matrix



