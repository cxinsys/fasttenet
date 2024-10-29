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
- **Default backend framework of the FastTENET is PyTorch Lightning.**
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
                                binning_method='FSBW-L',
                                )
```

### Run FastTENET with YAML config file

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
```

### Run make_grn.py
#### parameters
- **fp_rm**: result matrix, required
- **fp_exp**: expression file path for extracting node name file, required
- **fp_tf**: tf list file, optional
- **fdr**: specifying fdr, optional, default: 0.01
- **t_degrees**: specifying number of outdegrees, optional, generate final GRNs by incrementally increasing the fdr \
value until the total number of outdegrees is greater than the parameter value.
- **trim_threshold**: trimming threshold, optional, default: 0

#### Usage
When specifying an fdr
```angular2html
python make_grn.py --fp_rm [result matrix path] --fp_exp [expression file path] --fp_tf [tf file path] --fdr [fdr]
```

#### Example
```angular2html
python make_grn.py --fp_rm TE_result_matrix.txt --fp_exp expression_dataTuck.csv --fp_tf mouse_tf.txt --fdr 0.01
```

#### Output
```angular2html
TE_result_matrix.byGRN.fdr0.01.sif, TE_result_matrix.byGRN.fdr0.01.sif.outdegrees.txt
TE_result_matrix.byGRN.fdr0.01.trimIndirect0.sif, TE_result_matrix.byGRN.fdr0.01.trimIndirect0.sif.outdegrees.txt
```

#### Usage
When specifying the t_degrees
```angular2html
python make_grn.py --fp_rm [result matrix path] --fp_exp [expression file path] --fp_tf [tf file path] --t_degrees [number of outdegrees]
```

#### Example
```angular2html
python make_grn.py --fp_rm TE_result_matrix.txt --fp_exp expression_dataTuck.csv --fp_tf mouse_tf.txt --t_degrees 1000
```

#### Output
```angular2html
TE_result_matrix.byGRN.fdr0.06.sif, TE_result_matrix.byGRN.fdr0.06.sif.outdegrees.txt
TE_result_matrix.byGRN.fdr0.06.trimIndirect0.sif, TE_result_matrix.byGRN.fdr0.06.trimIndirect0.sif.outdegrees.txt
```

## TODO

- [x] add 'JAX' backend module
- [x] add 'PyTorch Lightning' backend module
- [x] add 'TensorFlow' backend module
