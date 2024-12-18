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

from fasttenet.utils import load_exp_data, load_time_data
from fasttenet.utils import get_device_list, align_data

class FastTENET(object):
    def __init__(self,
                 dpath_exp_data=None,
                 dpath_trj_data=None,
                 dpath_branch_data=None,
                 dpath_tf_data=None,
                 spath_result_matrix=None,
                 make_binary=False,
                 config=None,
                 aligned_data=None,
                 node_name=None,
                 tfs=None
                 ):

        self._tf = None
        self._result_matrix = None
        self._refined_data = None

        if config:
            inits = config['INIT']
            droot = inits['DROOT']
            dpath_exp_data = osp.join(droot, inits['FPATH_EXP'])
            dpath_trj_data = osp.join(droot, inits['FPATH_TRJ'])
            dpath_branch_data = osp.join(droot, inits['FPATH_BRANCH'])

            if 'FPATH_TF' in inits:
                dpath_tf_data = osp.join(droot, inits['FPATH_TF'])

            make_binary = bool(inits['MAKE_BINARY'])

            self._node_name, self._exp_data = load_exp_data(dpath_exp_data, make_binary)
            self._trajectory = load_time_data(dpath_trj_data, dtype=np.float32)
            self._branch = load_time_data(dpath_branch_data, dtype=np.int32)

            if dpath_tf_data is not None:
                self._tf = np.loadtxt(dpath_tf_data, dtype=str)

            self._spath_result_matrix = osp.join(droot, inits['SPATH_RESULT'])

        else:
            if aligned_data is not None:
                self._refined_data = aligned_data

                if node_name is None:
                    raise ValueError("node name data should be defined if using refined data directly")

                self._node_name = node_name
                self._tf = tfs
            else:
                if not dpath_exp_data or not dpath_trj_data or not dpath_branch_data:
                    raise ValueError("One of the following variable is not defined correctly: dpath_exp_data, dpath_trj_data, dpath_branch_data")

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
                raise ValueError("Unexpected error on 'spath_result_matrix' has been occurred")
            spath_result_matrix = self._spath_result_matrix

        if self._result_matrix is None:
            raise ValueError("Unexpected error on 'result_matrix' has been occurred")

        tmp_rm = np.concatenate([self._node_name[:, None], self._result_matrix.astype(str)], axis=1)
        extended_nn = np.concatenate((['TE'], self._node_name))
        tmp_rm = np.concatenate([extended_nn[None, :], tmp_rm])

        np.savetxt(spath_result_matrix, tmp_rm.T, delimiter='\t', fmt='%s')
        print("Save result matrix: {}".format(spath_result_matrix))

    # data refining

    def run(self,
            backend=None,
            device_ids=None,
            procs_per_device=None,
            batch_size=0,
            num_kernels=1,
            binning_method='FSBW-L',
            kp=0.5,
            binning_opt: dict = None,
            smoothing_opt: dict = None,
            dt=1,
            config=None
            ):

        if not backend:
            if config:
                backend = config['BACKEND']
            else:
                backend = "cpu"

        if not device_ids:
            if config:
                if type(config['DEVICE_IDS']) == int:
                    device_ids = int(config['DEVICE_IDS'])
                else:
                    device_ids = list(config['DEVICE_IDS'])
            else:
                if 'cpu' in backend:
                    device_ids = [0]
                else:
                    device_ids = get_device_list()

        if not procs_per_device:
            if config:
                procs_per_device = int(config['PROCS_PER_DEVICE'])
            else:
                procs_per_device = 1

        if not batch_size and backend.lower() != "tenet":
            if config:
                batch_string = config['BATCH_SIZE'].split('**')
                if len(batch_string) > 1:
                    batch_size = int(batch_string[0]) ** int(batch_string[1])
                else:
                    batch_size = int(batch_string[0])
            else:
                raise ValueError("batch size should be refined")

        if config:
            binning_method = config['BINNING_METHOD']
            kp = float(config['KP'])

            if 'NUM_KERNELS' in config:
                num_kernels = int(config['NUM_KERNELS'])
            if 'DT' in config:
                dt = int(config['DT'])
            if 'BINNINGOPT' in config:
                binning_opt = config['BINNING_OPT']
            if 'SMOOTHOPT' in config:
                smoothing_opt = config['SMOOTHING_OPT']

        if self._refined_data is not None:
            arr = self._refined_data
        else:
            arr = align_data(data=self._exp_data, trj=self._trajectory, branch=self._branch)

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

        if backend == 'lightning' or backend == 'gpu' or backend == 'cuda':
            self._mate = mate.MATELightning(arr=arr,
                                            pairs=pairs,
                                            kp=kp,
                                            num_kernels=num_kernels,
                                            binning_method=binning_method,
                                            binning_opt=binning_opt,
                                            smoothing_opt=smoothing_opt,
                                            dt=dt
                                            )
            self._result_matrix = self._mate.run(backend='gpu',
                                                 devices=device_ids,
                                                 batch_size=batch_size,
                                                 num_workers=procs_per_device
                                                 )
        else:
            self._mate = mate.MATE(kp=kp,
                                   num_kernels=num_kernels,
                                   binning_method=binning_method,
                                   binning_opt=binning_opt,
                                   smoothing_opt=smoothing_opt,
                                   )
            self._result_matrix = self._mate.run(arr=arr,
                                                 pairs=pairs,
                                                 backend=backend,
                                                 device_ids=device_ids,
                                                 procs_per_device=procs_per_device,
                                                 batch_size=batch_size,
                                                 dt=dt
                                                 )

        if self._result_matrix is not None:
            self.save_result_matrix(self._spath_result_matrix)

        return self._result_matrix



