import time
import math
from itertools import permutations
import multiprocessing
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np
from scipy.signal import savgol_filter
import mate

from fasttenet.data import load_exp_data, load_time_data
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
        self._result_matrix = None

        self._mate = mate.MATE(device=device,
                               device_ids=device_ids,
                               batch_size=batch_size,
                               kp=kp,
                               percentile=percentile,
                               win_length=win_length,
                               polyorder=polyorder)

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

    # multiprocessing worker(calculate tenet)

    def run(self,
            device=None,
            device_ids=None,
            batch_size=None,
            kp=0.5,
            percentile=0,
            win_length=10,
            polyorder=3,
            ):

        if not device:
            device = "cpu"

        if not device_ids:
            if 'cpu' in device:
                device_ids = [0]
            else:
                device_ids = get_gpu_list()

        if not batch_size:
            raise ValueError("batch size should be refined")

        arr = self.refine_data()

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

        self._result_matrix = self._mate.run(arr=arr,
                                             pairs=pairs,
                                             device=device,
                                             device_ids=device_ids,
                                             batch_size=batch_size,
                                             kp=kp,
                                             percentile=percentile,
                                             win_length=win_length,
                                             polyorder=polyorder)

        if self._result_matrix is not None:
            self.save_result_matrix(self._spath_result_matrix)

        return self._result_matrix



