#!/bin/bash

set -e
set -x

base_dir="log"
env_name="Qbert"
agent_name="iswbc"
replay_dir="dataset/$env_name/1"
clip_min=0.0
clip_max=10
lip_coef=1.0
l2_coef_disc=0.0
l2_coef_clas=0.0001
num_layers_disc=1

export CUDA_VISIBLE_DEVICES="0"

for seed in 2021 2022 2023 2024 2025
do
  python -um experiments.train \
    --base_dir=$base_dir \
    --env_name=$env_name \
    --seed=$seed \
    --agent_name=$agent_name \
    --gin_files='experiments/configs/iswbc.gin' \
    --replay_dir=$replay_dir \
    --gin_bindings="ISWeightedBehaviourCloningAgent.replay_suffix=[49]" \
    --gin_bindings="ISWeightedBehaviourCloningAgent.supplementary_replay_suffix=[45,46,47,48]" \
    --gin_bindings="ISWeightedBehaviourCloningAgent.mode='joint'" \
    --gin_bindings="ISWeightedBehaviourCloningAgent.feature_extractor='shared'" \
    --gin_bindings="ISWeightedBehaviourCloningAgent.discriminator_steps=200000" \
    --gin_bindings="ISWeightedBehaviourCloningAgent.clip_min=$clip_min" \
    --gin_bindings="ISWeightedBehaviourCloningAgent.clip_max=$clip_max" \
    --gin_bindings="ISWeightedBehaviourCloningAgent.l2_coef_disc=$l2_coef_disc" \
    --gin_bindings="ISWeightedBehaviourCloningAgent.l2_coef_clas=$l2_coef_clas" \
    --gin_bindings="ISWeightedBehaviourCloningAgent.num_layers_disc=$num_layers_disc" \
    --gin_bindings="WrappedUnionReplayBuffer.replay_capacity=100000" \
    --gin_bindings="FixedReplayRunner.num_iterations=100" \
    --gin_bindings="FixedReplayRunner.training_steps=2000" \
    --gin_bindings="atari_lib.create_atari_environment.game_name='$env_name'" \
    --alsologtostderr &
done
wait



