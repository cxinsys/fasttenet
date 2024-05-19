import os
import os.path as osp
import argparse
from omegaconf import OmegaConf

import numpy as np

import fasttenet as fte

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dpath parser')
    parser.add_argument('--config', type=str, dest='config', required=True, help='config file path')

    args = parser.parse_args()

    fpath_config = args.config
    conf = OmegaConf.load(fpath_config)


    # Create worker
    # expression data, trajectory data, branch data path is required
    # tf data path is optional
    # save path is optional
    worker = fte.FastTENET(config=conf) # Optional, default: False

    result_matrix = worker.run(config=conf)