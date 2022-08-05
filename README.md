## Demo-Code for the Prediction of 2D Flow Fields

This repo contains the code used to produce the results of [1]. The dataset was based on pointclouds similar to the one provided as  `./example.vtu`. Inspect it with e.g. Paraview (https://www.paraview.org/).

This repository contains training and evaluation script for learning and predicting steady mesh-based simulations. Here, computational fluid dynamics (CFD) was used to produce the dataset.

[1] https://doi.org/10.1145/3539781.3539789 

**If you use the code here, please consider citing this paper:**

```latex
@inproceedings{10.1145/3539781.3539789,
author = {Str\"{o}nisch, Sebastian and Meyer, Marcus and Lehmann, Christoph},
title = {Flow Field Prediction on Large Variable Sized 2D Point Clouds with Graph Convolution},
year = {2022},
isbn = {9781450394109},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539781.3539789},
doi = {10.1145/3539781.3539789},
booktitle = {Proceedings of the Platform for Advanced Scientific Computing Conference},
articleno = {6},
numpages = {10},
keywords = {surrogate model, graph convolution, machine learning, CFD},
location = {Basel, Switzerland},
series = {PASC '22}
}
```



#### Requirements

The experiments described in this paper [1] used a single NVidia V100 GPU and the following software stack (based on `fosscuda-2019b` software bundle):

* GCC 8.3.0
* Python 3.7.4
* CUDA 10.1
* cuDNN 7.6.4

Python Packages used (with all dependencies):

```cmd
Package                            Version
---------------------------------- -----------
Cython                             0.29.6
meshio                             4.3.10
numpy                              1.17.3
PyYAML                             5.1.2
scipy                              1.3.1
torch                              1.6.0
torch-cluster                      1.5.7
torch-geometric                    1.6.1
torch-scatter                      2.0.5
torch-sparse                       0.6.7
```

Torch-geometric and associated packages are installed from here: https://github.com/pyg-team/pytorch_geometric
Easy installation is described here: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

#### Training

```sh
python -u ./train.py -t GCN_2LSTM --batch 1 -e 1400 --optimizer 'adam' -n -1 -s <path-to-sample-files>
```

Or use precompiled dataset with: `--dataset <precompiled-dataset>`

#### Evaluation
```sh
python -u ./eval.py --modeldir <directory-of-trained-model> -s <point-cloud-files> --idents '["<sample-identifier>"]'
```

 
