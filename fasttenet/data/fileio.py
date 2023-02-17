import os
import os.path as osp

import numpy as np

def load_exp_data(dpath, make_binary=False):
    fpath_binary = dpath[:-4]+'.npy'
    if osp.isfile(fpath_binary):
        print('Binary file already exist, load data from binary file')
        exp_data = np.load(fpath_binary)
        node_name = np.load(fpath_binary[:-4]+'_node_name.npy')
    else:
        exp_data = np.loadtxt(dpath, delimiter=',', dtype=str)
        node_name = exp_data[0, 1:]
        exp_data = exp_data[1:, 1:].T.astype(np.float32)

        if make_binary==True:
            np.save(dpath[:-4] + '_node_name.npy', node_name)
            np.save(dpath[:-4] + '.npy', exp_data)

    return node_name, exp_data

def load_time_data(dpath, dtype=np.float32):
    return np.loadtxt(dpath, dtype=dtype)

def make_binary(fpath_exp_data):
    droot = osp.dirname(fpath_exp_data)
    exp_data = np.loadtxt(fpath_exp_data, delimiter=',', dtype=str)

    node_name = exp_data[0, 1:]
    exp_data = exp_data[1:, 1:].T.astype(np.float32)

    np.save(osp.join(droot, osp.basename(fpath_exp_data)[:-4]+'_node_name.npy'), node_name)
    np.save(osp.join(droot, osp.basename(fpath_exp_data)[:-4]+'.npy'), exp_data)
    print('Binary file save at {}'.format(droot))

# def load_exp_data(droot, fname_trj, fname_ts, fname_exp, binarization=True):
#     droot = osp.abspath(droot)
#     fpath_trj = osp.join(droot, fname_trj)
#     fpath_ts = osp.join(droot, fname_ts)
#     fpath_exp = osp.join(droot, fname_exp)
#
#     trajectory = np.loadtxt(fpath_trj, dtype=np.float32)
#     branch = np.loadtxt(fpath_ts, dtype=np.int32) # variable name need to be modified
#
#     fname_binary = fname_exp[:-4]+'.npy'
#     fname_node_name = 'node_name.npy'
#
#     if binarization==True and osp.isfile(osp.join(droot, fname_binary)) and osp.isfile(osp.join(droot, fname_node_name)):
#         node_name = np.load(osp.join(droot, fname_node_name))
#         exp_data = np.load(osp.join(droot, fname_binary))
#     else:
#         exp_data = np.loadtxt(fpath_exp, delimiter=',', dtype=str)
#
#         node_name = exp_data[0, 1:]
#         exp_data = exp_data[1:, 1:].T.astype(np.float32)
#
#         np.save(osp.join(droot, fname_binary), node_name)
#         np.save(osp.join(droot, fname_node_name), exp_data)
#
#     trajectory = trajectory[branch==1]
#     sorted_trajectory = np.argsort(trajectory)
#
#     exp_data = exp_data[:, branch==1]
#     exp_data = exp_data[:, sorted_trajectory]
#
#     return node_name,
