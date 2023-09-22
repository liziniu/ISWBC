#!/bin/bash
set -e
set -x

algo='iswbc'
seed=2023
id_domains='1'
client_ratio=1.0
num_layers_disc=2
num_layers_clas=2
clip_min=0.1
clip_max=10.0
epochs_disc=100

export CUDA_VISIBLE_DEVICES='0'

# id domain is the expert data source to evaluate performance


if [ "$(uname)" == "Darwin" ]; then
  python -m main \
    --seed $seed \
    --algo $algo \
    --id_domains $id_domains
elif [ "$(uname)" == "Linux" ]; then
  for id_domains in '0' '1' '2' '3' '4' '5'
  do
  for seed in 2021 2022 2023 2024 2025
  do
    python -m main \
      --seed $seed \
      --algo $algo \
      --id_domains $id_domains \
      --client_ratio $client_ratio \
      --num_layers_disc $num_layers_disc \
      --num_layers_clas $num_layers_clas \
      --epochs_disc $epochs_disc \
      --clip_min $clip_min \
      --clip_max $clip_max
  done
  wait
  done
fi