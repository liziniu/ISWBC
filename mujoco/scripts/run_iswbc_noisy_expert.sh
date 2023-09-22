#!/bin/bash

set -e
set -x

seed=2022
stochastic_policy=1
dataset_dir="dataset/rlkit"
env_id="HalfCheetah-v2"
grad_reg_coef=1.0
num_expert=1
tau=0.0

export CUDA_VISIBLE_DEVICES="1"


for env_id in "Walker2d-v2" "Ant-v2" "HalfCheetah-v2" "Hopper-v2"
do
   for seed in 2021 2022 2023 2024 2025
   do
    python -m run_iswbc_noisy_expert \
      --seed=$seed \
      --env-id=$env_id \
      --dataset-dir=$dataset_dir \
      --stochastic-policy=$stochastic_policy \
      --num-expert-trajectory=$num_expert \
      --grad-reg-coef=$grad_reg_coef \
      --tau=$tau &
   done
   wait
done