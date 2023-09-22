
# Atari Video Games

This repository has the code for reproducing experiments in NeurIPS 2023 spotlight paper: **Imitation Learning from Imperfection: Theoretical Justifications and Algorithms**.

[[Paper Link]](https://openreview.net/forum?id=vO04AzsB49)

This folder contains code for the Atari video games tasks.

##  Install


```
conda env create -f environment.yml
conda activate batch_rl
```


## Dataset

We use the offline datasets from Rishabh Agarwal's work on *An optimistic perspective on offline reinforcement learning*.

[[Download Link]](https://console.cloud.google.com/storage/browser/atari-replay-datasets?pli=1)


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

Our codebase is based on the implementation of [batch_rl](https://github.com/google-research/batch_rl).