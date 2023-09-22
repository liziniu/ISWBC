# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compact implementation of a DQN agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random

import gin.tf
import numpy as np
import tensorflow as tf
from absl import logging
from dopamine.discrete_domains import atari_lib
# from dopamine.replay_memory import circular_replay_buffer

from experiments.replay_memory import WrappedFixedReplayBuffer

# These are aliases which are used by other classes.
NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
NATURE_DQN_DTYPE = atari_lib.NATURE_DQN_DTYPE
NATURE_DQN_STACK_SIZE = atari_lib.NATURE_DQN_STACK_SIZE
nature_dqn_network = atari_lib.NatureDQNNetwork


@gin.configurable
class BehaviourCloningAgent(object):
    """An implementation of the base classification agent."""

    def __init__(self,
                 sess,
                 num_actions,
                 observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=atari_lib.NATURE_DQN_DTYPE,
                 stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
                 network=atari_lib.NatureDQNNetwork,
                 loss_name="bc",
                 replay_data_dir=None,
                 replay_suffix=None,
                 supplementary_replay_suffix=None,
                 tf_device='/cpu:*',
                 eval_mode=False,
                 use_staging=False,
                 max_tf_checkpoints_to_keep=1,
                 optimizer=tf.compat.v1.train.AdamOptimizer(
                     learning_rate=0.00025),
                 summary_writer=None,
                 summary_writing_frequency=500,
                 allow_partial_reload=False,
                 init_checkpoint_dir=None,):
        """Initializes the agent and constructs the components of its graph.

        Args:
          sess: `tf.compat.v1.Session`, for executing ops.
          num_actions: int, number of actions the agent can take at any state.
          observation_shape: tuple of ints describing the observation shape.
          observation_dtype: tf.DType, specifies the type of the observations. Note
            that if your inputs are continuous, you should set this to tf.float32.
          stack_size: int, number of frames to use in state stack.
          network: tf.Keras.Model, expecting 2 parameters: num_actions,
            network_type. A call to this object will return an instantiation of the
            network provided. The network returned can be run with different inputs
            to create different outputs. See
            dopamine.discrete_domains.atari_lib.NatureDQNNetwork as an example.
          tf_device: str, Tensorflow device on which the agent's graph is executed.
          eval_mode: bool, True for evaluation and False for training.
          use_staging: bool, when True use a staging area to prefetch the next
            training batch, speeding training up by about 30%.
          max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
            keep.
          optimizer: `tf.compat.v1.train.Optimizer`, for training the value
            function.
          summary_writer: SummaryWriter object for outputting training statistics.
            Summary writing disabled if set to None.
          summary_writing_frequency: int, frequency with which summaries will be
            written. Lower values will result in slower training.
          allow_partial_reload: bool, whether we allow reloading a partial agent
            (for instance, only the network parameters).
        """
        assert isinstance(observation_shape, tuple)
        logging.info('Creating %s agent with the following parameters:',
                     self.__class__.__name__)
        logging.info('\t tf_device: %s', tf_device)
        logging.info('\t use_staging: %s', use_staging)
        logging.info('\t optimizer: %s', optimizer)
        logging.info('\t max_tf_checkpoints_to_keep: %d', max_tf_checkpoints_to_keep)
        logging.info('\t expert replay_suffix %s', replay_suffix)
        logging.info('\t supplementary replay_suffix %s', supplementary_replay_suffix)
        logging.info('\t init_checkpoint_dir %s', init_checkpoint_dir)

        if init_checkpoint_dir is not None:
            self._init_checkpoint_dir = os.path.join(
                init_checkpoint_dir, 'checkpoints')
        else:
            self._init_checkpoint_dir = None
        self.tf_device = tf_device
        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.stack_size = stack_size
        self.use_staging = use_staging
        self.network = network
        self.loss_name = loss_name
        self.replay_data_dir = replay_data_dir
        self.replay_suffix = replay_suffix
        self.supplementary_replay_suffix = supplementary_replay_suffix
        self.eval_mode = eval_mode
        self.training_steps = 0
        self.sgd_steps = 0
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.summary_writing_frequency = summary_writing_frequency
        self.allow_partial_reload = allow_partial_reload

        with tf.device(tf_device):
            # Create a placeholder for the state input to the DQN network.
            # The last axis indicates the number of consecutive frames stacked.
            state_shape = (1,) + self.observation_shape + (stack_size,)
            self.state = np.zeros(state_shape)
            self.state_ph = tf.compat.v1.placeholder(
                self.observation_dtype, state_shape, name='state_ph')
            self._replay = self._build_replay_buffer(use_staging, self.replay_suffix)

            self._build_networks()

            self._train_op = self._build_train_op()

        if self.summary_writer is not None:
            # All tf.summaries should have been defined prior to running this.
            self._merged_summaries = tf.compat.v1.summary.merge_all()
        self._sess = sess

        var_map = atari_lib.maybe_transform_variable_names(
            tf.compat.v1.global_variables())
        self._saver = tf.compat.v1.train.Saver(
            var_list=var_map, max_to_keep=max_tf_checkpoints_to_keep)

        # Variables to be initialized by the agent once it interacts with the
        # environment.
        self._observation = None
        self._last_observation = None

    def _create_network(self, name):
        """Builds the convolutional network used to compute the agent's Q-values.

        Args:
          name: str, this name is passed to the tf.keras.Model and used to create
            variable scope under the hood by the tf.keras.Model.
        Returns:
          network: tf.keras.Model, the network instantiated by the Keras model.
        """
        network = self.network(self.num_actions, name=name)
        return network

    def _build_networks(self):
        """Builds the Classification network computations needed for acting and training.
        """

        # _network_template instantiates the model and returns the network object.
        # The network object can be used to generate different outputs in the graph.
        # At each call to the network, the parameters will be reused.
        self.main_convnet = self._create_network(name='Main')
        self._net_outputs = self.main_convnet(self.state_ph)

        self._argmax_action = tf.argmax(self._net_outputs.q_values, axis=1)[0]

    def _build_replay_buffer(self, use_staging, replay_suffix):
        """Creates the replay buffer used by the agent."""

        return WrappedFixedReplayBuffer(
            data_dir=self.replay_data_dir,
            replay_suffix=replay_suffix,
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=1,
            gamma=1,
            observation_dtype=self.observation_dtype.as_numpy_dtype)

    def _build_train_op(self):
        """Builds a training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """

        replay_logits = self.main_convnet(self._replay.states).q_values
        labels = tf.one_hot(
            self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')

        pi = stable_softmax(replay_logits, tau=1, axis=1)
        log_pi = stable_scaled_log_softmax(replay_logits, tau=1, axis=1)
        entropy = -tf.reduce_sum(pi * log_pi, axis=1)

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=replay_logits
        )

        # scope = tf.compat.v1.get_default_graph().get_name_scope()
        # trainables_online = tf.compat.v1.get_collection(
        #     tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
        #     scope=os.path.join(scope, 'Main'))
        # gradient = tf.gradients(loss, trainables_online)
        # gradient_norm = tf.compat.v1.global_norm(gradient)

        if self.summary_writer is not None:
            with tf.compat.v1.variable_scope('Losses'):
                tf.compat.v1.summary.scalar('BCLoss', tf.reduce_mean(loss))
                tf.compat.v1.summary.scalar('Entropy', tf.reduce_mean(entropy))
                # tf.compat.v1.summary.scalar('GradientNorm', gradient_norm)
        return self.optimizer.minimize(tf.reduce_mean(loss))

    def begin_episode(self, observation):
        """Returns the agent's first action for this episode.

        Args:
          observation: numpy array, the environment's initial observation.

        Returns:
          int, the selected action.
        """
        self._reset_state()
        self._record_observation(observation)

        if not self.eval_mode:
            self._train_step()

        self.action = self._select_action()
        return self.action

    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Args:
          reward: float, the reward received from the agent's most recent action.
          observation: numpy array, the most recent observation.

        Returns:
          int, the selected action.
        """
        self._record_observation(observation)
        self.action = self._select_action()
        return self.action

    def end_episode(self, reward):
        """Signals the end of the episode to the agent.

        We store the observation of the current time step, which is the last
        observation of the episode.

        Args:
          reward: float, the last reward from the environment.
        """
        assert self.eval_mode, 'Eval mode is not set to be True.'

    def _select_action(self):
        """Select an action from the set of available actions.

        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates.

        Returns:
           int, the selected action.
        """
        # Choose the action with highest Q-value at the current state.
        return self._sess.run(self._argmax_action, {self.state_ph: self.state})

    def _train_step(self):
        """Runs a single training step.

        Runs a training op if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        self._sess.run(self._train_op)
        self.sgd_steps += 1
        if (self.summary_writer is not None and
                self.training_steps > 0 and
                self.training_steps % self.summary_writing_frequency == 0):
            summary = self._sess.run(self._merged_summaries)
            self.summary_writer.add_summary(summary, self.sgd_steps)

        self.training_steps += 1

    def _record_observation(self, observation):
        """Records an observation and update state.

        Extracts a frame from the observation vector and overwrites the oldest
        frame in the state buffer.

        Args:
          observation: numpy array, an observation from the environment.
        """
        # Set current observation. We do the reshaping to handle environments
        # without frame stacking.
        self._observation = np.reshape(observation, self.observation_shape)
        # Swap out the oldest frame with the current frame.
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[0, ..., -1] = self._observation

    def _store_transition(self, last_observation, action, reward, is_terminal):
        """Stores an experienced transition.

        Executes a tf session and executes replay buffer ops in order to store the
        following tuple in the replay buffer:
          (last_observation, action, reward, is_terminal).

        Pedantically speaking, this does not actually store an entire transition
        since the next state is recorded on the following time step.

        Args:
          last_observation: numpy array, last observation.
          action: int, the action taken.
          reward: float, the reward.
          is_terminal: bool, indicating if the current state is a terminal state.
        """
        self._replay.add(last_observation, action, reward, is_terminal)

    def _reset_state(self):
        """Resets the agent state by filling it with zeros."""
        self.state.fill(0)

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Returns a self-contained bundle of the agent's state.

        This is used for checkpointing. It will return a dictionary containing all
        non-TensorFlow objects (to be saved into a file by the caller), and it saves
        all TensorFlow objects into a checkpoint file.

        Args:
          checkpoint_dir: str, directory where TensorFlow objects will be saved.
          iteration_number: int, iteration number to use for naming the checkpoint
            file.

        Returns:
          A dict containing additional Python objects to be checkpointed by the
            experiment. If the checkpoint directory does not exist, returns None.
        """
        if not tf.io.gfile.exists(checkpoint_dir):
            return None
        # Call the Tensorflow saver to checkpoint the graph.
        self._saver.save(
            self._sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration_number)
        # Checkpoint the out-of-graph replay buffer.
        self._replay.save(checkpoint_dir, iteration_number)
        bundle_dictionary = {}
        bundle_dictionary['state'] = self.state
        bundle_dictionary['training_steps'] = self.training_steps
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        """Restores the agent from a checkpoint.

        Restores the agent's Python objects to those specified in bundle_dictionary,
        and restores the TensorFlow objects to those specified in the
        checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
          agent's state.

        Args:
          checkpoint_dir: str, path to the checkpoint saved by tf.Save.
          iteration_number: int, checkpoint version, used when restoring the replay
            buffer.
          bundle_dictionary: dict, containing additional Python objects owned by
            the agent.

        Returns:
          bool, True if unbundling was successful.
        """
        try:
            # self._replay.load() will throw a NotFoundError if it does not find all
            # the necessary files.
            self._replay.load(checkpoint_dir, iteration_number)
        except tf.errors.NotFoundError:
            if not self.allow_partial_reload:
                # If we don't allow partial reloads, we will return False.
                return False
            logging.info('Unable to reload replay buffer!')
        if bundle_dictionary is not None:
            for key in self.__dict__:
                if key in bundle_dictionary:
                    self.__dict__[key] = bundle_dictionary[key]
        elif not self.allow_partial_reload:
            return False
        else:
            logging.info("Unable to reload the agent's parameters!")
        # Restore the agent's TensorFlow graph.
        self._saver.restore(self._sess,
                            os.path.join(checkpoint_dir,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True


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


