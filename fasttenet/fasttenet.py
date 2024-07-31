import os
import os.path as osp
import time
import math
from itertools import permutations
import multiprocessing
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np
from scipy.signal import savgol_filter
import mate

from fasttenet.data import load_exp_data, load_time_data
from fasttenet.utils import get_device_list

class FastTENET(object):
    def __init__(self,
                 dpath_exp_data=None,
                 dpath_trj_data=None,
                 dpath_branch_data=None,
                 dpath_tf_data=None,
                 spath_result_matrix=None,
                 make_binary=False,
                 config=None,
                 ):

        self._tf = None
        self._refined_exp_data = None
        self._result_matrix = None

        if config:
            inits = config['INIT']
            droot = inits['DROOT']
            dpath_exp_data = osp.join(droot, inits['FPATH_EXP'])
            dpath_trj_data = osp.join(droot, inits['FPATH_TRJ'])
            dpath_branch_data = osp.join(droot, inits['FPATH_BRANCH'])
            dpath_tf_data = osp.join(droot, inits['FPATH_TF'])
            make_binary = bool(inits['MAKE_BINARY'])

            self._node_name, self._exp_data = load_exp_data(dpath_exp_data, make_binary)
            self._trajectory = load_time_data(dpath_trj_data, dtype=np.float32)
            self._branch = load_time_data(dpath_branch_data, dtype=np.int32)

            if dpath_tf_data is not None:
                self._tf = np.loadtxt(dpath_tf_data, dtype=str)

            self._spath_result_matrix = osp.join(droot, inits['SPATH_RESULT'])

        else:
            if not dpath_exp_data:
                raise ValueError("Expression data should be refined")
            if not dpath_trj_data:
                raise ValueError("Trajectory should be refined")
            if not dpath_branch_data:
                raise ValueError("Branch data should be refined")

            self._node_name, self._exp_data = load_exp_data(dpath_exp_data, make_binary)
            self._trajectory = load_time_data(dpath_trj_data, dtype=np.float32)
            self._branch = load_time_data(dpath_branch_data, dtype=np.int32)

            if dpath_tf_data is not None:
                self._tf = np.loadtxt(dpath_tf_data, dtype=str)

            self._spath_result_matrix = spath_result_matrix

        self._mate = None

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
            procs_per_device=None,
            batch_size=None,
            num_kernels=1,
            method='shift_left',
            kp=0.5,
            percentile=0,
            win_length=10,
            polyorder=3,
            kw_smooth=True,
            data_smooth=True,
            dt=1,
            config=None
            ):

        if not device:
            if config:
                device = config['DEVICE']
            else:
                device = "cpu"

        if not device_ids:
            if config:
                device_ids = int(config['DEVICE_IDS'])
            else:
                if 'cpu' in device:
                    device_ids = [0]
                else:
                    device_ids = get_device_list()

        if not procs_per_device:
            if config:
                procs_per_device = int(config['PROCS_PER_DEVICE'])
            else:
                procs_per_device = 1

        if not batch_size:
            if config:
                batch_string = config['BATCH_SIZE'].split('**')
                if len(batch_string) > 1:
                    batch_size = int(batch_string[0]) ** int(batch_string[1])
                else:
                    batch_size = int(batch_string[0])
            else:
                raise ValueError("batch size should be refined")

        if config:
            method = config['METHOD']
            num_kernels = int(config['NUM_KERNELS'])
            kp = float(config['KP'])
            dt = int(config['DT'])

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

        if device == 'lightning' or device == 'gpu' or device == 'cuda':
            self._mate = mate.MATELightning(arr=arr,
                                            pairs=pairs,
                                            kp=kp,
                                            num_kernels=num_kernels,
                                            method=method,
                                            dt=dt
                                            )
            self._result_matrix = self._mate.run(device='gpu',
                                                 devices=device_ids,
                                                 batch_size=batch_size,
                                                 num_workers=procs_per_device
                                                 )
        else:
            self._mate = mate.MATE(kp=kp,
                                   num_kernels=num_kernels,
                                   method=method,
                                   )
            self._result_matrix = self._mate.run(arr=arr,
                                                 pairs=pairs,
                                                 device=device,
                                                 device_ids=device_ids,
                                                 procs_per_device=procs_per_device,
                                                 batch_size=batch_size,
                                                 dt=dt
                                                 )

        if self._result_matrix is not None:
            self.save_result_matrix(self._spath_result_matrix)

        return self._result_matrix



