import os
import os.path as osp

import numpy as np

import fasttenet as fte

if __name__ == "__main__":
    droot = osp.abspath('./')
    # dpath_exp_data = osp.join(droot, 'expression_data_ex.csv')
    dpath_exp_data = osp.join(droot, 'expression_dataTuck.csv')
    dpath_trj_data = osp.join(droot, 'pseudotimeTuck.txt')
    dpath_branch_data = osp.join(droot, 'cell_selectTuck.txt')
    dpath_tf_data = osp.join(droot, 'mouse_tfs.txt')

    spath_result_matrix = osp.join(droot, "TE_result_matrix.txt")

    # Create worker
    # expression data, trajectory data, branch data path is required
    # tf data path is optional
    # save path is optional
    worker = fte.FASTTENET(dpath_exp_data=dpath_exp_data,
                           dpath_trj_data=dpath_trj_data,
                           dpath_branch_data=dpath_branch_data,
                           dpath_tf_data=dpath_tf_data,
                           spath_result_matrix=spath_result_matrix)

    # if device_ids is not refined, use whole gpus
    result_matrix = worker.work(device='gpu', device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
                                batch_size=2 ** 15)

    print(result_matrix)