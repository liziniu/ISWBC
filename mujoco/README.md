
# MuJoCo Locomotion Control

This repository has the code for reproducing experiments in NeurIPS 2023 spotlight paper: **Imitation Learning from Imperfection: Theoretical Justifications and Algorithms**.

[[Paper Link]](https://openreview.net/forum?id=vO04AzsB49)

This folder contains code for the MuJoCo locomotion control tasks.

##  Install

- Python 3.6

```
conda create -n ilwsd python=3.6
conda activate ilwsd
```

- Cudatoolkit == 10.1

```
conda install cudatoolkit=10.1
```

- Cudnn = 7.6.5


```
conda install cudnn
```


- Python Packages


```
pip install tensorflow-gpu==2.1
pip install tf-agents-nightly==0.2.0.dev20191125
pip install gym==0.15.4
pip install pyyaml==3.13
pip install tensorflow-probability==0.8.0
pip install tqdm
pip install mujoco-py==1.50.1.68
pip install tensorflow-gan==2.1.0
```

## Dataset

We only provide the expert dataset in the Github repository. The full replay dataset can be downloaded from [here](https://drive.google.com/drive/folders/1r0KM08z-f7qtVFJl9Z3pssfFw440TZ0K?usp=sharing).


## Run

- full replay

```
bash scripts/run_iswbc_full_replay.sh
```

- noisy expert

```
bash experiments/run_iswbc_noisy_expert.sh
```


## Acknowledgements

Our codebase is based on the implementation of [DemoDICE](https://github.com/KAIST-AILab/imitation-dice).