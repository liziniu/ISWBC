
# Image Classification

This repository has the code for reproducing experiments in NeurIPS 2023 spotlight paper: **Imitation Learning from Imperfection: Theoretical Justifications and Algorithms**.

[[Paper Link]](https://openreview.net/forum?id=vO04AzsB49)

This folder contains code for the image classification task.

##  Install


```
conda env create -f environment.yml
conda activate image
```


## Dataset

We use the DOMAINNET datasets from Xingchao Peng's work on *Moment matching for multi-source domain adaptation*.

[[Download Link]](http://csr.bu.edu/ftp/visda/2019/multi-source)


## Run


```
bash scripts/run_bc.sh
bash scripts/run_nbcu.sh
bash scripts/run_iswbc.sh
```


## Acknowledgements

We thank the authors from *Outsourcing training without uploading data via efficient collaborative open-source sampling* for providing the source code to make this experiment possible.