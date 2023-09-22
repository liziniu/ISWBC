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

import logging
import os

import gin
import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm

from absl import logging

from experiments.agents.bc_agent import BehaviourCloningAgent
from experiments.replay_memory import WrappedUnionReplayBuffer
from experiments.discrminator_network import DiscriminatorNetwork
from experiments.network import NatureDQNNetwork

EPS = 1e-6
EPS2 = 1e-6


def clamp(x, clip_min=0.05, clip_max=None):
    if clip_max:
        y = tf.minimum(x, clip_max)
    else:
        y = x
    y = (1. - tf.cast(x <= clip_min, tf.float32)) * y
    return y
    # min_value = tf.constant(0, dtype=x.dtype)
    # max_value = tf.constant(clip_max, dtype=x.dtype)
    # return tf.cond(x < clip_min, lambda: min_value,
    #                lambda: tf.cond(x > clip_max, lambda: max_value, lambda: x))


@gin.configurable
class ISWeightedBehaviourCloningAgent(BehaviourCloningAgent):
    """An implementation of the DQN agent with fixed replay buffer(s)."""

    def __init__(self, sess, num_actions, replay_suffix, supplementary_replay_suffix,
                 noisy_action=False,
                 discriminator_steps=200000,
                 grad_penalty_coeff=0.0,
                 lip_coef=1.0,
                 clip_min=0.0,
                 clip_max=None,
                 l2_coef_disc=0.001,
                 l2_coef_clas=0.0001,
                 normalize_weights=False,
                 mode='joint',
                 feature_extractor='shared',
                 num_layers_disc=1,
                 tf_device='/cpu:*', **kwargs):
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
        tf.logging.info('\t ISWBC will mix the expert and supplementary dataset as the union dataset')
        self.tf_device = tf_device
        self.num_actions = num_actions
        self.noisy_action = noisy_action
        self.discriminator_steps = discriminator_steps
        self.mode = mode
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.l2_coef_disc = l2_coef_disc
        self.l2_coef_clas = l2_coef_clas
        self.normalize_weights = normalize_weights
        self.lip_coef = lip_coef
        self.feature_extractor = feature_extractor
        self.num_layers_disc = num_layers_disc

        tf.logging.info('Clip: ({}, {})'.format(self.clip_min, self.clip_max))

        # use union dataset as the replay buffer
        replay_suffix = replay_suffix + supplementary_replay_suffix

        self.num_actions = num_actions
        self.grad_penalty_coeff = grad_penalty_coeff
        super(ISWeightedBehaviourCloningAgent, self).__init__(sess, num_actions,
                                                              replay_suffix=replay_suffix,
                                                              supplementary_replay_suffix=None,
                                                              tf_device=tf_device,
                                                              network=NatureDQNNetwork,
                                                              **kwargs)
        tf.logging.info(tf.compat.v1.trainable_variables())

    def _build_replay_buffer(self, use_staging, replay_suffix):
        """Creates the replay buffer used by the agent."""

        return WrappedUnionReplayBuffer(
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

    def _build_discriminator_networks(self):
        """Builds the Classification network computations needed for acting and training.
        """

        # _network_template instantiates the model and returns the network object.
        # The network object can be used to generate different outputs in the graph.
        # At each call to the network, the parameters will be reused.
        if self.feature_extractor == 'shared':
            self.discriminator_convnet = DiscriminatorNetwork(
                num_actions=self.num_actions, name='Discriminator',
                feature_extractor=lambda x: tf.stop_gradient(self.main_convnet.compute_features(x)),
                num_layers=self.num_layers_disc
            )
        elif self.feature_extractor == 'separate':
            self.discriminator_convnet = DiscriminatorNetwork(num_actions=self.num_actions, name='Discriminator')
        else:
            raise ValueError(self.feature_extractor)

    def _build_train_op(self):
        """Builds a training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """

        train_ops = []

        self._pi_train_op, self._pi_summary_op = self._build_policy_train_op()
        self._disc_train_op, self._disc_summary_op = self._build_discriminator_train_op()

        train_ops.append(self._pi_train_op)
        train_ops.append(self._disc_train_op)

        return train_ops

    def _build_discriminator_train_op(self):
        # train discriminator
        alpha = tf.random.uniform(shape=(self._replay.states.get_shape()[0], 1))
        expert_action_one_hot = tf.one_hot(
            self._replay.expert_actions, self.num_actions, 1., 0., name='expert_action_one_hot')
        replay_action_one_hot = tf.one_hot(
            self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')

        alpha_tensor = tf.reshape(alpha, [alpha.get_shape()[0], 1, 1, 1])
        inter_states = (alpha_tensor * tf.cast(self._replay.expert_states, tf.float32)
                        + (1 - alpha_tensor) * tf.cast(self._replay.states, tf.float32))
        inter_actions = (alpha * tf.cast(expert_action_one_hot, tf.float32)
                         + (1 - alpha) * tf.cast(replay_action_one_hot, tf.float32))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.discriminator_convnet.variables)
            expert_output = self.discriminator_convnet(
                self._replay.expert_states, expert_action_one_hot)
            union_output = self.discriminator_convnet(
                self._replay.states, replay_action_one_hot)

            expert_loss = tf.losses.sigmoid_cross_entropy(
                tf.ones_like(expert_output),
                expert_output
            )
            union_loss = tf.losses.sigmoid_cross_entropy(
                tf.zeros_like(union_output),
                union_output
            )
            classification_loss = tf.reduce_mean(expert_loss + union_loss)

            if self.feature_extractor == 'shared':
                grad_penalty = 0
            else:
                with tf.GradientTape(watch_accessed_variables=False) as tape2:
                    tape2.watch([inter_states, inter_actions])
                    output = self.discriminator_convnet(inter_states, inter_actions)
                    output = tf.math.log(1 / (tf.nn.sigmoid(output) + EPS2) - 1 + EPS2)

                grads = tape2.gradient(output, [inter_states, inter_actions])
                grad = tf.concat([tf.keras.layers.Flatten()(grads[0]),
                                  tf.keras.layers.Flatten()(grads[1])], axis=-1)
                grad_penalty = tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1) - self.lip_coef, 2))

            weight_decay_loss = tf.add_n([tf.nn.l2_loss(var) for var in self.discriminator_convnet.variables])

            discriminator_loss = (
                    classification_loss
                    + self.grad_penalty_coeff * grad_penalty
                    + self.l2_coef_disc * weight_decay_loss
            )

        grads_and_vars = self.discriminator_optimizer.compute_gradients(
            discriminator_loss, var_list=self.discriminator_convnet.trainable_variables)
        train_op = self.discriminator_optimizer.apply_gradients(grads_and_vars)

        summary_op = []
        if self.summary_writer is not None:
            with tf.compat.v1.variable_scope('Losses'):
                summary_op.append(
                    tf.compat.v1.summary.scalar('DiscriminatorLoss', discriminator_loss)
                )
                summary_op.append(
                    tf.compat.v1.summary.scalar('ClassificationLoss', classification_loss)
                )
                summary_op.append(
                    tf.compat.v1.summary.scalar('GradientPenalty', grad_penalty)
                )
                summary_op.append(
                    tf.compat.v1.summary.scalar('DiscriminatorWeightDecay', weight_decay_loss)
                )
        summary_op = tf.compat.v1.summary.merge(summary_op)

        return train_op, summary_op

    def _build_policy_train_op(self):
        # train policy
        replay_action_one_hot = tf.one_hot(
            self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')

        weight_logits = self.discriminator_convnet(self._replay.states, replay_action_one_hot)
        weights = tf.math.log(1 / (tf.nn.sigmoid(weight_logits) + EPS2) - 1 + EPS2)
        weights = tf.stop_gradient(tf.math.exp(-weights))
        weights = weights[:, 0]
        if self.normalize_weights:
            weights = weights / tf.reduce_mean(weights)
        else:
            # weights = tf.map_fn(lambda x: clamp(x, self.clip_min, self.clip_max), weights)
            # weights = tf.clip_by_value(weights, self.clip_min, self.clip_max)
            weights = clamp(weights, self.clip_min, self.clip_max)
        replay_logits = self.main_convnet(self._replay.states).q_values

        pi = stable_softmax(replay_logits, tau=1, axis=1)
        log_pi = stable_scaled_log_softmax(replay_logits, tau=1, axis=1)
        entropy = -tf.reduce_sum(pi * log_pi, axis=1)

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=replay_action_one_hot,
            logits=replay_logits
        )
        loss = tf.reduce_mean(loss * weights)

        weight_decay_loss = tf.add_n([tf.nn.l2_loss(var) for var in self.main_convnet.variables])
        classifier_loss = (loss
                           + self.l2_coef_clas * weight_decay_loss
                           )

        summary_op = []
        if self.summary_writer is not None:
            with tf.compat.v1.variable_scope('Losses'):
                summary_op.append(
                    tf.compat.v1.summary.scalar('Weights', tf.reduce_mean(weights))
                )
                summary_op.append(
                    tf.compat.v1.summary.scalar('WeightsMax', tf.reduce_max(weights))
                )
                summary_op.append(
                    tf.compat.v1.summary.scalar('BCLoss', loss)
                )
                summary_op.append(
                    tf.compat.v1.summary.scalar('BCWeightDecay', weight_decay_loss)
                )
                summary_op.append(
                    tf.compat.v1.summary.scalar('Entropy', tf.reduce_mean(entropy))
                )
        summary_op = tf.compat.v1.summary.merge(summary_op)

        train_op = self.optimizer.minimize(classifier_loss)

        return train_op, summary_op

    def _train_step(self):
        if self.mode == 'joint':
            if self.training_steps > 0 and self.training_steps % self.summary_writing_frequency == 0:
                self._debug()
            return super()._train_step()
        elif self.mode == 'separate':
            # discriminator training
            if self.training_steps < self.discriminator_steps:
                print('Training Discriminator....')
                for _ in tqdm(range(self.discriminator_steps)):
                    self._sess.run(self._disc_train_op)
                    self.training_steps += 1
                    if (self.summary_writer is not None and
                            self.training_steps > 0 and
                            self.training_steps % self.summary_writing_frequency == 0):
                        summary = self._sess.run(self._disc_summary_op)
                        self.summary_writer.add_summary(summary, self.training_steps)
                        self._debug()

            # policy training
            self._sess.run(self._pi_train_op)
            self.sgd_steps += 1
            self.training_steps += 1

            if (self.summary_writer is not None and
                    self.sgd_steps > 0 and
                    self.sgd_steps % self.summary_writing_frequency == 0):
                summary = self._sess.run(self._pi_summary_op)
                self.summary_writer.add_summary(summary, self.sgd_steps)
        elif self.mode == 'debug':
            self._debug()
            # import pdb
            # pdb.set_trace()
        else:
            raise ValueError(self.mode)
        # super()._train_step()

    # def _create_network(self, name):
    #     """Builds the convolutional network used to compute the agent's Q-values.
    #
    #     Args:
    #       name: str, this name is passed to the tf.keras.Model and used to create
    #         variable scope under the hood by the tf.keras.Model.
    #     Returns:
    #       network: tf.keras.Model, the network instantiated by the Keras model.
    #     """
    #     if self.feature_extractor == 'shared':
    #         assert self.mode == 'joint', "Must be in joint training mode"
    #         network = self.network(self.num_actions, name=name, feature_extractor=
    #             lambda x: self.discriminator_convnet.compute_features(x))
    #     elif self.feature_extractor == 'separate':
    #         network = self.network(self.num_actions, name=name)
    #     else:
    #         raise NotImplementedError
    #     return network

    def _build_networks(self):
        self.main_convnet = self._create_network(name='Main')
        self._net_outputs = self.main_convnet(self.state_ph)

        self._argmax_action = tf.argmax(self._net_outputs.q_values, axis=1)[0]

        self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=0.00025)
        self._build_discriminator_networks()


def stable_scaled_log_softmax(x, tau, axis=-1):
    """Scaled log_softmax operation.
      Args:
        x: tensor of floats, inputs of the softmax (logits).
        tau: float, softmax temperature.
        axis: int, axis to perform the softmax operation.
      Returns:
        tau * tf.log_softmax(x/tau, axis=axis)
      """
    max_x = tf.reduce_max(x, axis=axis, keepdims=True)
    y = x - max_x
    tau_lse = max_x + tau * tf.math.log(
        tf.reduce_sum(tf.math.exp(y / tau), axis=axis, keepdims=True))
    return x - tau_lse


def stable_softmax(x, tau, axis=-1):
    """Stable softmax operation.
      Args:
        x: tensor of floats, inputs of the softmax (logits).
        tau: float, softmax temperature.
        axis: int, axis to perform the softmax operation.
      Returns:
        softmax(x/tau, axis=axis)
      """
    max_x = tf.reduce_max(x, axis=axis, keepdims=True)
    y = x - max_x
    return tf.nn.softmax(y / tau, axis=axis)
