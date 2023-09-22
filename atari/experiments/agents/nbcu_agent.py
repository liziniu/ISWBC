# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN agent with fixed replay buffer(s)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import gin
import tensorflow.compat.v1 as tf

from experiments.agents.bc_agent import BehaviourCloningAgent
from experiments.replay_memory import WrappedFixedReplayBuffer


@gin.configurable
class NaiveBehaviourCloningAgent(BehaviourCloningAgent):
    """An implementation of the DQN agent with fixed replay buffer(s)."""

    def __init__(self, sess, num_actions, replay_suffix, supplementary_replay_suffix, noisy_action=False, **kwargs):
        """Initializes the agent and constructs the components of its graph.

        Args:
          sess: tf.Session, for executing ops.
          num_actions: int, number of actions the agent can take at any state.
          replay_data_dir: str, log Directory from which to load the replay buffer.
          replay_suffix: int, If not None, then only load the replay buffer
            corresponding to the specific suffix in data directory.
          init_checkpoint_dir: str, directory from which initial checkpoint before
            training is loaded if there doesn't exist any checkpoint in the current
            agent directory. If None, no initial checkpoint is loaded.
          **kwargs: Arbitrary keyword arguments.
        """
        tf.logging.info('\t NBCU will mix the expert and supplementary dataset')
        self.supplementary_replay_suffix = supplementary_replay_suffix
        self.noisy_action = noisy_action
        self.num_actions = num_actions

        # use union dataset as the replay buffer
        replay_suffix = replay_suffix + supplementary_replay_suffix
        super(NaiveBehaviourCloningAgent, self).__init__(sess, num_actions,
                                                         replay_suffix=replay_suffix,
                                                         supplementary_replay_suffix=None,
                                                         **kwargs)

    def _build_replay_buffer(self, use_staging, replay_suffix):
        """Creates the replay buffer used by the agent."""

        return WrappedFixedReplayBuffer(
            data_dir=self.replay_data_dir,
            replay_suffix=replay_suffix,
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            noisy_action=self.noisy_action,
            num_actions=self.num_actions,
            use_staging=use_staging,
            update_horizon=1,
            gamma=1,
            observation_dtype=self.observation_dtype.as_numpy_dtype)