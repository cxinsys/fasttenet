import time
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np

from fasttenet.array import get_array_module

class TE(object):
    def __init__(self,
                 device=None,
                 batch_size=None,
                 pairs=None,
                 n_pairs=None,
                 inds_pair=None,
                 bin_arr=None,
                 len_time=None,
                 shm_name=None,
                 result_matrix=None,
                 sem=None,
                 dt=1
                 ):

        self._am = get_array_module(device)

        self._batch_size = batch_size
        self._pairs = pairs
        self._n_pairs = n_pairs

        self._inds_pair = inds_pair
        self._bin_arr = bin_arr

        if inds_pair is not None:
            with self.am:
                self._inds_pair = self.am.array(inds_pair, dtype=inds_pair.dtype)

        if bin_arr is not None:
            with self.am:
                self._bin_arr = self.am.array(bin_arr, dtype=bin_arr.dtype)

        self._len_time = len_time

        self._shm_name = shm_name
        self._result_matrix = result_matrix
        self._sem = sem

        self._dt = dt

    @property
    def am(self):
        return self._am

    def solve(self,
              batch_size=None,
              pairs=None,
              bin_arr=None,
              shm_name=None,
              result_matrix=None,
              sem=None,
              n_pairs=None,
              inds_pair=None,
              len_time=None,
              dt=1
              ):

        if not batch_size:
            if not self._batch_size:
                raise ValueError("batch size should be defined")
            batch_size = self._batch_size

        if pairs is None:
            if self._pairs is None:
                raise ValueError("pairs should be defined")
            pairs = self._pairs

        if not n_pairs:
            if not self._n_pairs:
                self._n_pairs = n_pairs = len(pairs)
            n_pairs = self._n_pairs

        if inds_pair is None:
            if self._inds_pair is None:
                self._inds_pair = inds_pair = np.arange(batch_size)
            inds_pair = self._inds_pair

        if bin_arr is None:
            if self._bin_arr is None:
                raise ValueError("binned array should be defined")
            bin_arr = self._bin_arr

        if not len_time:
            if not self._len_time:
                self._len_time = len_time = len(bin_arr[1])
            len_time = self._len_time

        if not shm_name:
            if not self._shm_name:
                raise ValueError("shared memory name should be defined")
            shm_name = self._shm_name

        if result_matrix is None:
            if self._result_matrix is None:
                raise ValueError("result matrix should be defined")
            result_matrix = self._result_matrix

        if not sem:
            if not self._sem:
                raise ValueError("semaphore should be defined")
            sem = self._sem

        if not dt:
            if not self._dt:
                self._dt = dt = 1
            dt = self._dt

        for i_iter, i_beg in enumerate(range(0, n_pairs, batch_size)):
            t_beg_batch = time.time()

            print("[%s, Batch #%d]" % (str(self.am.device).upper(), i_iter + 1))

            i_end = i_beg + batch_size
            inds_pair = self.am.arange(len(pairs[i_beg:i_end]))

            tile_inds_pair = self.am.repeat(inds_pair, len_time-1)


            bin_arr = self.am.array(bin_arr, dtype=bin_arr.dtype)
            # pairs = self.am.array(pairs, dtype=pairs.dtype)
            target_arr = self.am.take(bin_arr, pairs[i_beg:i_end, 0], axis=0)
            source_arr = self.am.take(bin_arr, pairs[i_beg:i_end, 1], axis=0)

            vals = self.am.stack((target_arr[:, dt:],
                                  target_arr[:, :-dt],
                                  source_arr[:, :-dt]),
                                 axis=2)



            pair_vals = self.am.concatenate((tile_inds_pair[:, None], vals.reshape(-1, 3)), axis=1)

            uvals_xt1_xt_yt, cnts_xt1_xt_yt = self.am.unique(pair_vals, return_counts=True, axis=0)

            subuvals_xt1_xt, n_subuvals_xt1_xt = self.am.unique(uvals_xt1_xt_yt[:, :-1], return_counts=True, axis=0)
            subuvals_xt_yt, n_subuvals_xt_yt = self.am.unique(self.am.take(uvals_xt1_xt_yt, [0, 2, 3], axis=1), return_counts=True, axis=0)
            subuvals_xt, n_subuvals_xt = self.am.unique(self.am.take(uvals_xt1_xt_yt, [0, 2], axis=1), return_counts=True, axis=0)

            uvals_xt1_xt, cnts_xt1_xt = self.am.unique(pair_vals[:, :-1], return_counts=True, axis=0)
            uvals_xt_yt, cnts_xt_yt = self.am.unique(self.am.take(pair_vals, [0, 2, 3], axis=1), return_counts=True, axis=0)
            uvals_xt, cnts_xt = self.am.unique(self.am.take(pair_vals, [0, 2], axis=1), return_counts=True, axis=0)

            cnts_xt1_xt = self.am.repeat(cnts_xt1_xt, n_subuvals_xt1_xt)

            cnts_xt_yt = self.am.repeat(cnts_xt_yt, n_subuvals_xt_yt)
            ind_xt_yt = self.am.lexsort(self.am.take(uvals_xt1_xt_yt, [3, 2, 0], axis=1).T)
            ind2ori_xt_yt = self.am.argsort(ind_xt_yt)
            cnts_xt_yt = self.am.take(cnts_xt_yt, ind2ori_xt_yt)

            cnts_xt = self.am.repeat(cnts_xt, n_subuvals_xt)
            ind_xt = self.am.lexsort(self.am.take(uvals_xt1_xt_yt, [2, 0], axis=1).T)
            ind2ori_xt = self.am.argsort(ind_xt)
            cnts_xt = self.am.take(cnts_xt, ind2ori_xt)

            p_xt1_xt_yt = self.am.divide(cnts_xt1_xt_yt, (len_time - 1))
            p_xt1_xt = self.am.divide(cnts_xt1_xt, (len_time - 1))
            p_xt_yt = self.am.divide(cnts_xt_yt, (len_time - 1))
            p_xt = self.am.divide(cnts_xt, (len_time - 1))


            numer = self.am.multiply(p_xt1_xt_yt, p_xt)
            denom = self.am.multiply(p_xt1_xt, p_xt_yt)
            fraction = self.am.divide(numer, denom)
            log_val = self.am.log2(fraction)
            entropies = self.am.multiply(p_xt1_xt_yt, log_val)

            uvals_tot, n_subuvals_tot = self.am.unique(uvals_xt1_xt_yt[:, 0], return_counts=True)
            final_bins = self.am.repeat(uvals_tot, n_subuvals_tot)
            entropy_final = self.am.bincount(final_bins, weights=entropies)

            entropy_final = self.am.asnumpy(entropy_final)

            sem.acquire()

            new_shm = shared_memory.SharedMemory(name=shm_name)
            tmp_arr = np.ndarray(result_matrix.shape, dtype=result_matrix.dtype, buffer=new_shm.buf)
            tmp_arr[pairs[i_beg:i_end, 0], pairs[i_beg:i_end, 1]] = entropy_final

            new_shm.close()

            sem.release()

            print("[%s, Batch #%d] Batch processing elapsed time: %f" % (str(self.am.device).upper(), i_iter + 1, time.time() - t_beg_batch))