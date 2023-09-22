# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Implementation of twin_sac, a mix of TD3 (https://arxiv.org/abs/1802.09477) and SAC (https://arxiv.org/abs/1801.01290, https://arxiv.org/abs/1812.05905).

Overall structure and hyperparameters are taken from TD3. However, the algorithm
itself represents a version of SAC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import tensorflow_probability as tfp
from tf_agents.networks import network, utils
from tf_agents.specs.tensor_spec import TensorSpec

ds = tfp.distributions


def my_reset_states(metric):
    """Resets metric states.

  Args:
    metric: A keras metric to reset states for.
  """
    for var in metric.variables:
        var.assign(0)


class DeterministicActor(tf.keras.Model):
    """Gaussian policy with TanH squashing."""

    def __init__(self, state_dim, action_dim):
        """Creates an actor.

    Args:
      state_dim: State size.
      action_dim: Action size.
    """
        super(DeterministicActor, self).__init__()
        self.trunk = tf.keras.Sequential([
            tf.keras.layers.Dense(
                256,
                input_shape=(state_dim,),
                activation=tf.nn.relu,
                kernel_initializer='orthogonal'
                ),
            tf.keras.layers.Dense(
                256, activation=tf.nn.relu,
                kernel_initializer='orthogonal'
            ),
            tf.keras.layers.Dense(
                action_dim,
                kernel_initializer='orthogonal',
                activation=tf.nn.tanh,
            )
        ])

    @tf.function
    def call(self, states):
        """Computes actions for given inputs.

    Args:
      states: A batch of states.

    Returns:
      A mode action, a sampled action and log probability of the sampled action.
    """
        actions = self.trunk(states)
        return (actions, actions), None


class DeterministicBCPolicy(object):
    def __init__(self, state_dim, action_dim, lr=1e-3, l2_coef=0.0):
        self.actor = DeterministicActor(state_dim, action_dim)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.l2_coef = l2_coef

    @tf.function
    def update(self, states, actions):
        """Performs a single training step of behavior clonning.

        The method optimizes MLE on the expert dataset.

        # Args:
        #   expert_dataset_iter: An tensorflow graph iteratable object.
        """

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.actor.variables)
            # states, actions, _ = next(expert_dataset_iter)
            actor_actions = self.actor(states)[0]
            actor_loss = tf.reduce_mean((actor_actions - actions) ** 2)

            l2_loss = 0.0
            for variable in self.actor.variables:
                if 'bias' in variable.name:
                    continue
                l2_loss += tf.nn.l2_loss(variable)
            l2_loss = l2_loss * self.l2_coef
            total_loss = actor_loss + l2_loss

        actor_grads = tape.gradient(total_loss, self.actor.variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.variables))

        info_dict = {
            'bc_loss': actor_loss,
            'l2_loss': l2_loss,
        }
        return info_dict

    @tf.function
    def step(self, observation):
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        action, *_ = self.actor(observation)
        return action

    def save(self, filepath, *args):
        self.actor.save_weights(os.path.join(filepath, "actor"))

    def load(self, filepath):
        self.actor.load_weights(os.path.join(filepath, "actor"))


LOG_STD_MIN = -5
LOG_STD_MAX = 2
SCALE_DIAG_MIN_MAX = (LOG_STD_MIN, LOG_STD_MAX)
MEAN_MIN_MAX = (-7, 7)
EPS = np.finfo(np.float32).eps


class StochasticActor(network.Network):
    def __init__(self, state_dim, action_dim, hidden_size=256, name='TanhNormalPolicy',
                 mean_range=(-7., 7.), logstd_range=(-5., 2.), eps=EPS, initial_std_scaler=1,
                 kernel_initializer='glorot_uniform', activation_fn=tf.nn.relu):
        self._input_specs = TensorSpec(state_dim)
        self._action_dim = action_dim
        self._initial_std_scaler = initial_std_scaler

        super(StochasticActor, self).__init__(self._input_specs, state_spec=(), name=name)

        hidden_sizes = (hidden_size, hidden_size)

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=activation_fn,
                                           kernel_initializer=kernel_initializer, name='mlp')
        self._fc_mean = tf.keras.layers.Dense(action_dim, name='policy_mean/dense',
                                              kernel_initializer=kernel_initializer)
        self._fc_logstd = tf.keras.layers.Dense(action_dim, name='policy_logstd/dense',
                                                kernel_initializer=kernel_initializer)

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def call(self, inputs, step_type=(), network_state=(), training=True):
        del step_type  # unused
        h = inputs
        for layer in self._fc_layers:
            h = layer(h, training=training)

        mean = self._fc_mean(h)
        mean = tf.clip_by_value(mean, self.mean_min, self.mean_max)
        logstd = self._fc_logstd(h)
        logstd = tf.clip_by_value(logstd, self.logstd_min, self.logstd_max)
        std = tf.exp(logstd) * self._initial_std_scaler
        pretanh_action_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        pretanh_action = pretanh_action_dist.sample()
        action = tf.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.log_prob(pretanh_action_dist, pretanh_action, is_pretanh_action=True)

        return (tf.tanh(mean), action, log_prob), network_state

    def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = tf.tanh(pretanh_action)
        else:
            pretanh_action = tf.atanh(tf.clip_by_value(action, -1 + self.eps, 1 - self.eps))

        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
        log_prob = pretanh_log_prob - tf.reduce_sum(tf.math.log(1 - action ** 2 + self.eps), axis=-1)

        return log_prob, pretanh_log_prob

    def get_log_prob(self, states, actions):
        """Evaluate log probs for actions conditined on states.
        Args:
            states: A batch of states.
            actions: A batch of actions to evaluate log probs on.
        Returns:
            Log probabilities of actions.
        """
        h = states
        for layer in self._fc_layers:
            h = layer(h, training=True)

        mean = self._fc_mean(h)
        mean = tf.clip_by_value(mean, self.mean_min, self.mean_max)
        logstd = self._fc_logstd(h)
        logstd = tf.clip_by_value(logstd, self.logstd_min, self.logstd_max)
        std = tf.exp(logstd) * self._initial_std_scaler

        pretanh_action_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        pretanh_actions = tf.atanh(tf.clip_by_value(actions, -1 + self.eps, 1 - self.eps))
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_actions)

        log_probs = pretanh_log_prob - tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.eps), axis=-1)
        log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
        return log_probs
