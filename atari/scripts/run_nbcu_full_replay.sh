#!/bin/bash

set -e
set -x

base_dir="log"
env_name="Phoenix"
agent_name="nbcu"
replay_dir="dataset/$env_name/1"


export CUDA_VISIBLE_DEVICES="0"


for seed in 2021 2022 2023 2024 2025
do
  replay_dir="dataset/$env_name/1"
  python -um experiments.train \
    --base_dir=$base_dir \
    --env_name=$env_name \
    --seed=$seed \
    --agent_name=$agent_name \
    --gin_files='experiments/configs/nbcu.gin' \
    --replay_dir=$replay_dir \
    --gin_bindings="NaiveBehaviourCloningAgent.replay_suffix=[49]" \
    --gin_bindings="NaiveBehaviourCloningAgent.supplementary_replay_suffix=[45,46,47,48]" \
    --gin_bindings="FixedReplayRunner.num_iterations=100" \
    --gin_bindings="FixedReplayRunner.training_steps=2000" \
    --gin_bindings="WrappedFixedReplayBuffer.replay_capacity=100000" \
    --gin_bindings="atari_lib.create_atari_environment.game_name='$env_name'" \
    --alsologtostderr &
done
wait




