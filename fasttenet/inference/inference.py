import os
import os.path as osp
import sys
from collections import Counter
from itertools import permutations
import argparse
import multiprocessing

import numpy as np
import scipy.stats
import statsmodels.sandbox.stats.multicomp
import networkx as nx
from tqdm import tqdm

from mate.array import get_array_module
from mate.utils import get_device_list
from fasttenet.inference.utils import calculate_batchsize

class NetWeaver(object):
    def __init__(self,
                 result_matrix=None,
                 gene_names=None,
                 tfs=None,
                 fdr=0.01,
                 links=0,
                 is_trimming=True,
                 trim_threshold=0.0,
                 dtype=np.float32):

        self.result_matrix = result_matrix
        self.gene_names = gene_names
        self.tfs = tfs
        self.fdr = fdr
        self.links = links
        self.is_trimming = is_trimming
        self.trim_threshold = trim_threshold
        self.dtype = dtype

    def run(self,
            backend=None,
            device_ids=None,
            procs_per_device=None,
            batch_size=0,
            ):
        if not backend:
            backend = 'cpu'

        if not device_ids:
            if backend == 'cpu':
                device_ids = [0]
            else:
                device_ids = get_device_list()

        if not procs_per_device:
            procs_per_device = 1

        if type(device_ids) is int:
            list_device_ids = [x for x in range(device_ids)]
            device_ids = list_device_ids

        if self.tfs is not None:
            _, inds_source, _ = np.intersect1d(self.gene_names, self.tfs, return_indices=True)
            inds_all = np.arange(len(self.gene_names))

            repeated_sinds = np.repeat(inds_source, len(inds_all))
            repeated_ainds = np.tile(inds_all, len(inds_source))
            _, inds_inter, _ = np.intersect1d(repeated_sinds, repeated_ainds, return_indices=True)
            inds_diff = np.setdiff1d(np.arange(len(repeated_ainds)), inds_inter)

            pairs = np.array([repeated_sinds[inds_diff], repeated_ainds[inds_diff]]).T
        else:
            pairs = permutations(range(len(self.result_matrix)), 2)
            pairs = np.asarray(tuple(pairs), dtype=np.int32)

        print('Number of pairs: ', len(pairs))

        # Indexing to get the 1D arrays
        source = self.gene_names.T[pairs[:, 0]]
        target = self.gene_names.T[pairs[:, 1]]

        te = self.result_matrix[pairs[:, 0], pairs[:, 1]]

        if self.links != 0:
            sorted_inds = np.argsort(te)
            te_cutoff = te[sorted_inds][:self.links]
            source_cutoff = source[sorted_inds][:self.links]
            target_cutoff = target[sorted_inds][:self.links]
            pairs = pairs[sorted_inds][:self.links]
        else:
            te_zscore = (te - np.mean(te)) / np.std(te)
            te_pval = 1 - scipy.stats.norm.cdf(te_zscore)
            te_fdr = statsmodels.sandbox.stats.multicomp.multipletests(te_pval, alpha=0.05, method='fdr_bh')

            inds_cutoff = te_fdr[1] < self.fdr

            source_cutoff = source[inds_cutoff]
            target_cutoff = target[inds_cutoff]
            te_cutoff = te[inds_cutoff]
            pairs = pairs[inds_cutoff]

        te_grn = np.stack((source_cutoff, te_cutoff, target_cutoff), axis=1)

        print("[Statistical analysis done]")

        if self.is_trimming:
            if not batch_size:
                print("[Batch Size auto calculated]")
                batch_size = calculate_batchsize(shape=self.result_matrix.shape,
                                                 dtype=self.dtype,
                                                 num_gpus=len(device_ids),
                                                 num_ppd=procs_per_device)

            print("[TRIMMING APPLIED]")
            print("[DEVICE: {}, Num. Processor: {}, Process per device: {}, Batch Size: {}"
                  .format(backend, len(device_ids), procs_per_device, batch_size))

            # T = Trimmer()
            multiprocessing.set_start_method('spawn', force=True)

            trimmed_pair_list, tes_list = [], []
            arr = np.zeros(self.result_matrix.shape, dtype=self.dtype)
            arr[pairs[:, 0], pairs[:, 1]] = te_grn[:, 1].astype(self.dtype)

            with multiprocessing.Pool(processes=len(device_ids) * procs_per_device) as pool:
                list_backend = []
                list_dtype = []
                list_indices = []
                list_batches = []
                list_arrs = []
                list_threshold = []
                list_ids = []

                outer_batch = np.ceil(len(self.result_matrix) / (len(device_ids) * procs_per_device)).astype(np.int32)
                for i, start in enumerate(range(0, len(self.result_matrix), outer_batch)):
                    end = start + outer_batch

                    list_backend.append(backend + ":" + str(device_ids[i % len(device_ids)]))
                    list_dtype.append(self.dtype)
                    list_indices.append((start, end))
                    list_batches.append(batch_size)
                    list_arrs.append(arr)
                    list_threshold.append(self.trim_threshold)
                    list_ids.append(i)

                inputs = zip(list_backend, list_dtype, list_indices, list_batches, list_arrs, list_threshold, list_ids)

                for batch_result in pool.istarmap(self.trim, inputs):
                    if batch_result is None:
                        continue

                    tmp_tes, tmp_pairs = batch_result

                    trimmed_pair_list.extend(tmp_pairs)
                    tes_list.extend(tmp_tes)

            trimmed_pairs = np.concatenate(trimmed_pair_list, axis=0)
            tes = np.concatenate(tes_list, axis=0)

            trimmed_source = self.gene_names.T[trimmed_pairs[:, 0]]
            trimmed_target = self.gene_names.T[trimmed_pairs[:, 1]]

            trimmed_te_grn = np.stack((trimmed_source, tes, trimmed_target), axis=1)

            return te_grn, trimmed_te_grn

        return te_grn

    def trim(self,
             backend='cpu',
             dtype=np.float32,
             indices=None,
             batch_size=None,
             arr=None,
             threshold=0.0,
             id=0):

        am = get_array_module(backend)

        arr = am.array(arr, dtype=dtype)
        procs_arr = arr[indices[0]:indices[1]]

        trimmed_pair_list, tes_list = [], []
        for i, start in enumerate(
                tqdm(range(0, len(procs_arr), batch_size), desc=f"Process {id}", position=id, leave=True)):
            end = start + batch_size

            batch_arr = procs_arr[start:end]
            min_arr = am.max(am.minimum(batch_arr[:, :, None], arr[None, :, :]), axis=1) + threshold
            trimmed_arr = am.where(batch_arr < min_arr, 0, batch_arr)

            inds_nonzeros = am.nonzero(trimmed_arr)

            if len(inds_nonzeros[0]) == 0:
                continue

            adjusted_inds = (inds_nonzeros[0] + indices[0] + start, inds_nonzeros[1])
            trimmed_pairs = am.transpose(am.stack(adjusted_inds, axis=0), axes=(1, 0))

            flattend_index = inds_nonzeros[0] * trimmed_arr.shape[1] + inds_nonzeros[1]
            tes = am.take(am.reshape(trimmed_arr, [-1]), flattend_index)

            trimmed_pair_list.append(am.asnumpy(trimmed_pairs))
            tes_list.append(am.asnumpy(tes))

        return tes_list, trimmed_pair_list

    def count_outdegree(self, grn):
        dg = nx.from_edgelist(grn[:, [0, 2]], create_using=nx.DiGraph)
        out_degrees = sorted(dg.out_degree, key=lambda x: x[1], reverse=True)

        return out_degrees