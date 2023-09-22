#!/bin/bash
set -e
set -x

script="main"
algo='bc'
seed=2023
id_domains='1'
client_ratio=0.5
num_layers_clas=2

export CUDA_VISIBLE_DEVICES='1'

# id domain is the expert data source to evaluate performance

if [ "$(uname)" == "Darwin" ]; then
  python -m "$script" \
    --seed $seed \
    --algo $algo \
    --id_domains $id_domains
elif [ "$(uname)" == "Linux" ]; then
  for id_domains in '0' '1' '2' '3' '4' '5'
  do
    for seed in 2021 2022 2023 2024 2025
    do
      python -m "$script" \
        --seed $seed \
        --algo $algo \
        --id_domains $id_domains \
        --client_ratio $client_ratio \
        --num_layers_clas $num_layers_clas
    done
    wait
  done
fi