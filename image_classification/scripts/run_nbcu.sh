#!/bin/bash
set -e
set -x

algo='nbcu'
seed=2023
id_domains='0'
client_ratio=0.5
num_layers_clas=2

export CUDA_VISIBLE_DEVICES='2'

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
      --num_layers_clas $num_layers_clas &
  done
  wait
  done
fi