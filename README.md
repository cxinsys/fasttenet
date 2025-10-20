<p align="center"><img src="assets/logo.png" alt="Drawing" width="395px"/></p>


## Indroduction
- FastTENET is an accelerated [TENET](https://github.com/neocaleb/TENET) algorithm based on manycore computing.

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

### Install from PyPi

```
pip install fasttenet
```
- **Default backend framework of the FastTENET is PyTorch.**
- **You need to install other backend frameworks such as CuPy, Jax, and TensorFlow**

<br>

### Install from GitHub repository


[//]: # (**You must install [MATE]&#40;https://github.com/cxinsys/mate&#41; before installing FastTENET**)

First, clone the recent version of this repository.

```
git clone https://github.com/cxinsys/fasttenet.git
```


Now, we need to install FastTENET as a module.

```
cd fasttenet
pip install -e .
```


- **Default backend framework of the FastTENET is PyTorch Lightning.**

<br>

### Install backend frameworks

FastTENET supports several backend frameworks including CuPy, JAX, TensorFlow, PyTorch and PyTorch-Lightning. \
To use frameworks, you need to install the framework manually

<br>

- **PyTorch Lightning**

PyTorch Lightning is a required dependency library for FastTENET and is installed automatically when you install FastTENET.\
If the library is not installed, you can install it manually via pip.

```angular2html
python -m pip install lightning
```
<br>

- **PyTorch**: [Installing custom PyTorch version](https://pytorch.org/get-started/locally/#start-locally)

PyTorch is a required dependency library for FastTENET and is installed automatically when you install FastTENET.\
If the library is not installed, you can install it manually.

```angular2html
conda install pytorch torchvision torchaudio pytorch-cuda=xx.x -c pytorch -c nvidia (check your CUDA version)
```
<br>

- **CuPy**: [Installing CuPy from Conda-Forge with cudatoolkit](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge)

Install Cupy from Conda-Forge with cudatoolkit supported by your driver
```angular2html
conda install -c conda-forge cupy cuda-version=xx.x (check your CUDA version)
```
<br>

- **JAX**: [Installing JAX refer to the installation guide in the project README](https://github.com/google/jax#installation)

[//]: # (**You must first install [CUDA]&#40;https://developer.nvidia.com/cuda-downloads&#41; and [CuDNN]&#40;https://developer.nvidia.com/cudnn&#41; before installing JAX**)

[//]: # ()
[//]: # (After install CUDA and CuDNN you can specify a particular CUDA and CuDNN version for jax explicitly)
Install JAX with CUDA > 12.x
```angular2html
pip install -U "jax[cuda12]"
```

[//]: # (JAX preallocate 90% of the totla GPU memory when the first JAX operation is run \)
Use 'XLA_PYTHON_CLIENT_PREALLOCATE=false' to disables the preallocation behavior\
(https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)

<br>

- **TensorFlow**: [Installing TensorFlow refer to the installation guide](https://www.tensorflow.org/install/pip?hl=en#linux)

[//]: # (**You must first install [CUDA]&#40;https://developer.nvidia.com/cuda-downloads&#41; and [CuDNN]&#40;https://developer.nvidia.com/cudnn&#41; before installing JAX**)

[//]: # ()
[//]: # (After install CUDA and CuDNN you can specify a particular CUDA and CuDNN version for jax explicitly)
Install TensorFlow-GPU with CUDA
```angular2html
python3 -m pip install tensorflow[and-cuda]
```

<br>


## FastTENET tutorial

### Create FastTENET instance

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

<br>

- **aligned_data**: when directly using rearranged data with expression data, trajectory data and branch data, optional
- **node_name**: 1d array of node names, required when using data directly
- **tf**: 1d array of tf names, optional when using data directly

```angular2html
import fasttenet as fte

node_name, exp_data = fte.load_exp_data(dpath_exp_data, make_binary=True)
trajectory = fte.load_time_data(dpath_trj_data, dtype=np.float32)
branch = fte.load_time_data(dpath_branch_data, dtype=np.int32)
tf = np.loadtxt(dpath_tf_data, dtype=str)

aligned_data = fte.align_data(data=exp_data, trj=trajectory, branch=branch)
        
worker = fte.FastTENET(aligned_data=aligned_data,
                       node_name=node_name,
                       tfs=tf,
                       spath_result_matrix=spath_result_matrix) # Optional
```

<br>
<br>

### Run FastTENET

#### parameters
- **backend**: optional, default: 'cpu'
- **device_ids**: list or number of devcies to use, optional, default: [0] (cpu), [list of whole gpu devices] (gpu) 
- **procs_per_device**: The number of processes to create per device when using non 'cpu' devices, optional, default: 1
- **batch_size**: Required
- **kp**: kernel percentile, optional, default: 0.5
- **binning_method**: discretization method for expression values, optional, 'FSBW-L' is recommended to achieve results similar to TENET. 


```angular2html
result_matrix = worker.run(backend='gpu',
                           device_ids=8,
                           procs_per_device=4,
                           batch_size=2 ** 16,
                           kp=0.5,
                           binning_method='FSBW-L')
```

<br>
<br>

### Run FastTENET with config file

- **Before run tutorial_config.py, batch_size parameter must be modified to fit your gpu memory size**
- **You can set parameters and run FastTENET via a YAML file**
- **The config file must have values set for all required parameters**

#### Usage
```angular2html
python tutorial_config.py --config [config file path]
```

#### Example
```angular2html
python tutorial_config.py --config ../configs/config_tuck_sub.yml
```

#### Output
```angular2html
TE_result_matrix.txt

ex)
TE	GENE_1	GENE_2	GENE_3	...	GENE_M
GENE_1	0	0.05	0.02	...	0.004
GENE_2	0.01	0	0.04	...	0.12
GENE_3	0.003	0.003	0	...	0.001
.
.
.
GENE_M	0.34	0.012	0.032	...	0
```

### Run FastTENET with tutorial_notf.py

- **Before run tutorial_notf.py, batch_size parameter must be modified to fit your gpu memory size**

#### Usage
```angular2html
python tutorial_notf.py --fp_exp [expression file path] 
                        --fp_trj [trajectory file path] 
                        --fp_br [cell select file path] 
                        --backend [name of backend framework]
                        --num_devices [number of devices]
                        --batch_size [batch size]
                        --sp_rm [save file path]
```

#### Example
```angular2html
python tutorial_notf.py --fp_exp expression_dataTuck.csv 
                        --fp_trj pseudotimeTuck.txt 
                        --fp_br cell_selectTuck.txt 
                        --backend lightning
                        --num_devices 8
                        --batch_size 32768
                        --sp_rm TE_result_matrix.txt
```

#### Output
```angular2html
TE_result_matrix.txt
```

### Run FastTENET with tutorial_tf.py
- **Before run tutorial_tf.py, batch_size parameter must be modified to fit your gpu memory size**

#### Usage
```angular2html
python tutorial_tf.py --fp_exp [expression file path] 
                      --fp_trj [trajectory file path] 
                      --fp_br [cell select file path] 
                      --fp_tf [tf file path] 
                      --backend [name of backend framework]
                      --num_devices [number of devices]
                      --batch_size [batch size]
                      --sp_rm [save file path]
```

#### Example
```angular2html
python tutorial_tf.py --fp_exp expression_dataTuck.csv 
                      --fp_trj pseudotimeTuck.txt 
                      --fp_br cell_selectTuck.txt 
                      --fp_tf mouse_tfs.txt 
                      --backend lightning
                      --num_devices 8
                      --batch_size 32768
                      --sp_rm TE_result_matrix.txt
```

#### Output
```angular2html
TE_result_matrix.txt

ex)
TE	GENE_1	GENE_2	GENE_3	...	GENE_M
GENE_1	0	0.05	0.02	...	0.004
GENE_2	0.01	0	0.04	...	0.12
GENE_3	0.003	0.003	0	...	0.001
.
.
.
GENE_M	0.34	0.012	0.032	...	0
```

<br>
<br>

## Downstream analysis tutorial

### Create NetWeaver instance



#### parameters

- **result_matrix**: result TE matrix of FastTENET, required
- **gene_names**: gene names from result matrix, required
- **tfs**: tf list, optional
- **fdr**: specifying fdr, optional, default: 0.01
- **links**: specifying number of outdegrees, optional, default: 0
- **is_trimming**: if set True, trimming operation is applied on grn, optional, default: True
- **trim_threshold**: trimming threshold, optional, default: 0

```angular2html
result_matrix = np.loadtxt(fpath_result_matrix, delimiter='\t', dtype=str)
gene_name = result_matrix[0][1:]
result_matrix = result_matrix[1:, 1:].astype(np.float32)

tf = np.loadtxt(fpath_tf, dtype=str)

weaver = fte.NetWeaver(result_matrix=result_matrix,
                       gene_names=gene_name,
                       tfs=tf,
                       fdr=fdr,
                       links=links,
                       is_trimming=True,
                       trim_threshold=trim_threshold,
                       dtype=np.float32
                       )
```

### Run weaver
- **backend**: optional, default: 'cpu'
- **device_ids**: list or number of devices to use, optional, default: [0] (cpu), [list of whole gpu devices] (gpu) 
- **batch_size**: if set to 0, batch size will automatically calculated, optional, default: 0

```angular2html
grn, trimmed_grn = weaver.run(backend=backend,
                              device_ids=device_ids,
                              batch_size=batch_size)
```

### Count outdegree
- **grn**: required

```angular2html
outdegrees = weaver.count_outdegree(grn)
trimmed_ods = weaver.count_outdegree(trimmed_grn)
```

<br>
<br>

### Downstream analysis with reconstruct_grn.py

reconstruct_grn.py is a tutorial script for the output of grn and outdegree files.

#### Usage
When specifying an fdr
```angular2html
python reconstruct_grn.py --fp_rm [result matrix path] --fp_tf [tf file path] --fdr [fdr] --backend [backend] --device_ids [number of device]
```

#### Example
```angular2html
python reconstruct_grn.py --fp_rm TE_result_matrix.txt --fp_tf mouse_tf.txt --fdr 0.01 --backend gpu --device_ids 1
```

#### Output
```angular2html
TE_result_matrix.fdr0.01.sif, TE_result_matrix.fdr0.01.sif.outdegrees.txt
TE_result_matrix.fdr0.01.trimIndirect0.sif, TE_result_matrix.fdr0.01.trimIndirect0.sif.outdegrees.txt
```

<br>

#### Usage
When specifying the links
```angular2html
python reconstruct_grn.py --fp_rm [result matrix path] --fp_tf [tf file path] --links [links] --backend [backend] --device_ids [number of device]
```

#### Example
```angular2html
python reconstruct_grn.py --fp_rm TE_result_matrix.txt--fp_tf mouse_tf.txt --links 1000 --backend gpu --device_ids 1
```

#### Output
```angular2html
TE_result_matrix.links1000.sif, TE_result_matrix.links1000.sif.outdegrees.txt
TE_result_matrix.links1000.trimIndirect0.sif, TE_result_matrix.links1000.trimIndirect0.sif.outdegrees.txt
```

## TODO

- [x] add 'JAX' backend module
- [x] add 'PyTorch Lightning' backend module
- [x] add 'TensorFlow' backend module
