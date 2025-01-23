# GNN Powergrid
This repository implements Graph Neural Networks for Power Flow (PF) simulation. This implementation may inspire future works whcih are based mainly on control (RL) problems.

<div align="center">
  <img src="./imgs/gnn_eq_powergrid.png">
</div>

<div align="center">
  <img src="./imgs/gnn_scheme_powergrid.png">
</div>

<!-- ![eq](./imgs/gnn_eq_powergrid.png)
![scheme](./imgs/gnn_scheme_powergrid.png) -->
## Getting started

See the getting started notebooks at the root of this repository for the examples on how to use this package

## Installation
To be able to run the experiments in this repository, the following steps show how to install this package and its dependencies from source.

### Requirements
- Python >= 3.6

### Setup a Virtualenv (optional)
#### Create a Conda env (recommended)
```commandline
conda create -n venv_gnn python=3.10
conda activate venv_gnn
```
#### Create a virtual environment

```commandline
cd my-project-folder
pip3 install -U virtualenv
python3 -m virtualenv venv_gnn
```
#### Enter virtual environment
```commandline
source venv_gnn/bin/activate
```

### Install from source
```commandline
git clone https://gitlab.inesctec.pt/cpes/european-projects/ai4realnet/irt-systemx/task-1.2/graph-neural-solver.git
cd graph-neural-solver
pip3 install -U .[recommended]
cd ..
```

### To contribute
```commandline
pip3 install -e .[recommended]
```