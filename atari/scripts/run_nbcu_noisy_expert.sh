#!/bin/bash

set -e
set -x

base_dir="log"
env_name="SpaceInvaders"
agent_name="nbcu"
seed=2021
replay_dir="dataset/$env_name/1"


export CUDA_VISIBLE_DEVICES="3"

# If you change replay_suffix, please make sure the buffer index is correct.
# Refer to Line 115 and 211 in replay_memory.py

for seed in 2021 2022 2023 2024 2025
do
  python -um experiments.train \
    --base_dir=$base_dir \
    --env_name=$env_name \
    --seed=$seed \
    --agent_name=$agent_name \
    --gin_files='experiments/configs/nbcu.gin' \
    --replay_dir=$replay_dir \
    --gin_bindings="NaiveBehaviourCloningAgent.replay_suffix=[49]" \
    --gin_bindings="NaiveBehaviourCloningAgent.supplementary_replay_suffix=[45,46,47,48]" \
    --gin_bindings="NaiveBehaviourCloningAgent.noisy_action=1" \
    --gin_bindings="WrappedFixedReplayBuffer.replay_capacity=100000" \
    --gin_bindings="FixedReplayRunner.num_iterations=100" \
    --gin_bindings="FixedReplayRunner.training_steps=2000" \
    --gin_bindings="atari_lib.create_atari_environment.game_name='$env_name'" \
    --alsologtostderr &
done
wait



