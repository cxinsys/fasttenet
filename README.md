<p align="center"><img src="assets/logo.png" alt="Drawing" width="395px"/></p>


## Indroduction
- FastTENET is a library that supports multi-gpu acceleration of the [TENET](https://github.com/neocaleb/TENET) algorithm.

## Installation
- :snake: [Anaconda](https://www.anaconda.com) is recommended to use and develop FastTENET.
- :penguin: Linux distros are tested and recommended to use and develop FastTENET.

### Anaconda virtual environment

After installing anaconda, create a conda virtual environment for FastTENET.
In the following command, you can change the Python version
(e.g.,`python=3.7` or `python=3.9`).

```
conda create -n fasttenet python=3.9
```

Now, we can activate our virtual environment for FastTENET as follows.

```
conda activate fasttenet
```
<br>

FastTENET requires following backend-specific dependencies to be installed:


- CuPy: [Installing CuPy from Conda-Forge with cudatoolkit](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge)

Install Cupy from Conda-Forge with cudatoolkit supported by your driver
```angular2html
conda install -c conda-forge cupy cuda-version=xx.x (check your CUDA version)
```
<br>

- JAX: [Installing JAX refer to the installation guide in the project README](https://github.com/google/jax#installation)

**You must first install [CUDA](https://developer.nvidia.com/cuda-downloads) and [CuDNN](https://developer.nvidia.com/cudnn) before installing JAX**

After install CUDA and CuDNN you can specify a particular CUDA and CuDNN version for jax explicitly
```angular2html
pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.x and cudnn >= 8.6
pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

JAX preallocate 90% of the totla GPU memory when the first JAX operation is run \
Use 'XLA_PYTHON_CLIENT_PREALLOCATE=false' to disables the preallocation behavior\
(https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)

### Install from GitHub repository


**You must install [MATE](https://github.com/cxinsys/mate) before installing FastTENET**

First, clone the recent version of this repository.

```
git clone https://github.com/cxinsys/fasttenet.git
```


Now, we need to install FastTENET as a module.

```
cd fasttenet
pip install -e .
```

## Tutorial

### FastTENET class
#### Create FastTENET instance

FastTENET class requires data path as parameter

#### parameters
- **dpath_exp_data**: expression data path, required
- **dpath_trj_data**: trajectory data path, required
- **dpath_branch_data**: branch(cell select) data path, required
- **dpath_tf_data**: tf data path, required
- **spath_result_matrix**: result matrix data path, optional, default: None
- **make_binary**: if True, make binary expression and node name file, optional, default: False

```angular2html
import fasttenet as fte

worker = fte.FastTENET(dpath_exp_data=dpath_exp_data,
                           dpath_trj_data=dpath_trj_data,
                           dpath_branch_data=dpath_branch_data,
                           dpath_tf_data=dpath_tf_data,
                           spath_result_matrix=spath_result_matrix,
                           make_binary=True)
```


#### Run FastTENET

#### parameters
- **device**: optional, default: 'cpu'
- **device_ids**: list or number of devcies to use, optional, default: [0] (cpu), [list of whole gpu devices] (gpu) 
- **procs_per_device**: The number of processes to create per device when using non 'cpu' devices, optional, default: 1
- **batch_size**: Required
- **kp**: kernel percentile, optional, default: 0.5
- **method**: approximations for calculating TE, optional, 'pushing' method is recommended to achieve results similar to TENET. 


```angular2html
result_matrix = worker.run(device='gpu',
                                device_ids=8,
                                procs_per_device=4,
                                batch_size=2 ** 16,
                                kp=0.5,
                                method='pushing',
                                )
```

### Run FastTENET with tutorial_notf.py

- **Before Run tutorial_notf.py batch_size parameter must be modified to fit your gpu memory size**

#### Usage
```angular2html
python tutorials/tutorial_notf.py --fp_exp [expression file path] --fp_trj [trajectory file path] --fp_br [cell select file path] --sp_rm [save file path]
```

#### Example
```angular2html
python tutorials/tutorial_notf.py --fp_exp expression_dataTuck.csv --fp_trj pseudotimeTuck.txt --fp_br cell_selectTuck.txt --sp_rm TE_result_matrix.txt
```

#### Output
```angular2html
TE_result_matrix.txt
```

### Run FastTENET with tutorial_tf.py
- **Before Run tutorial_tf.py batch_size parameter must be modified to fit your gpu memory size**

#### Usage
```angular2html
python tutorials/tutorial_tf.py --fp_exp [expression file path] --fp_trj [trajectory file path] --fp_br [cell select file path] --fp_tf [tf file path] --sp_rm [save file path]
```

#### Example
```angular2html
python tutorials/tutorial_tf.py --fp_exp expression_dataTuck.csv --fp_trj pseudotimeTuck.txt --fp_br cell_selectTuck.txt --fp_tf mouse_tfs.txt --sp_rm TE_result_matrix.txt
```

#### Output
```angular2html
TE_result_matrix.txt
```

### Run make_grn.py
#### parameters
- **fp_rm**: result matrix, required
- **fp_nn**: node name file, required
- **fp_tf**: tf list file, optional
- **fdr**: specifying fdr, optional, default: 0.01
- **t_degrees**: specifying number of outdegrees, optional, generate final GRNs by incrementally increasing the fdr \
value until the total number of outdegrees is greater than the parameter value.

#### Usage
When specifying an fdr
```angular2html
python stats/make_grn.py --fp_rm [result matrix path] --fp_nn [node name file path] --fp_tf [tf file path] --fdr [fdr]
```

#### Example
```angular2html
python stats/make_grn.py --fp_rm TE_result_matrix.txt --fp_nn expression_dataTuck_node_name.npy --fp_tf mouse_tf.txt --fdr 0.01
```

#### Output
```angular2html
te_result_grn.fdr0.01.sif, te_result_grn.fdr0.01.sif.outdegrees.txt
```

#### Usage
When specifying an t_degrees
```angular2html
python stats/make_grn.py --fp_rm [result matrix path] --fp_nn [node name file path] --fp_tf [tf file path] --t_degrees [number of outdegrees]
```

#### Example
```angular2html
python stats/make_grn.py --fp_rm TE_result_matrix.txt --fp_nn expression_dataTuck_node_name.npy --fp_tf mouse_tf.txt --t_degrees 1000
```

#### Output
If the t_degrees parameter was used, the filename will include the fdr value when the outdegrees condition is met. 
```angular2html
te_result_grn.fdr0.06.sif, te_result_grn.fdr0.06.sif.outdegrees.txt
```

## TODO

- [x] add 'jax' backend module
- [x] add 'lightning' backend module