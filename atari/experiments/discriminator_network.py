

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin

import tensorflow as tf


DiscriminatorNetworkType = collections.namedtuple('dqn_network', ['logits'])


@gin.configurable
class DiscriminatorNetwork(tf.keras.Model):
    """The convolutional network used for discriminator"""

    def __init__(self, num_actions, feature_extractor=None, num_layers=1, name='Discriminator'):
        """Creates the layers used for calculating logits.

        Args:
          num_actions: int, number of actions.
          name: str, used to create scope for network parameters.
        """
        super(DiscriminatorNetwork, self).__init__(name=name)

        self.num_actions = num_actions
        self.feature_extractor = feature_extractor
        self.num_layers = num_layers
        # Defining layers.
        activation_fn = tf.keras.activations.relu
        # Setting names of the layers manually to make variable names more similar
        # with tf.slim variable names/checkpoints.
        if self.feature_extractor is None:
            self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                                activation=activation_fn, name='Conv')
            self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                                activation=activation_fn, name='Conv')
            self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                                activation=activation_fn, name='Conv')
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                                name='fully_connected')
        else:
            pass
        self.dense2 = tf.keras.layers.Dense(512, name='fully_connected',
                                            trainable=False)
        self.dense3 = tf.keras.layers.Dense(1024, activation=activation_fn,
                                            name='fully_connected')
        self.dense5 = tf.keras.layers.Dense(1024, activation=activation_fn,
                                            name='fully_connected')
        self.dense4 = tf.keras.layers.Dense(1, name='fully_connected')

    def call(self, state, action):
        """Creates the output tensor/op given the state tensor as input.

        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        information on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.

        Parameters created here will have scope according to the `name` argument
        given at `.__init__()` call.
        Args:
          state: Tensor, input tensor.
          action: Tensor, input tensor.
        Returns:
          collections.namedtuple, output ops (graph mode) or output tensors (eager).
        """
        if self.feature_extractor is None:
            x = tf.cast(state, tf.float32)
            x = x / 255
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            x = self.dense1(x)
        else:
            x = self.feature_extractor(state)

        act = tf.cast(action, tf.float32)
        act = self.dense2(act)

        h = tf.keras.layers.concatenate([x, act], axis=-1)
        if self.num_layers == 1:
            logits = self.dense4(h)
        elif self.num_layers == 2:
            h = self.dense3(h)
            logits = self.dense4(h)
        elif self.num_layers == 3:
            h = self.dense3(h)
            h = self.dense5(h)
            logits = self.dense4(h)
        else:
            raise ValueError(self.num_layers)
        return logits

    def compute_features(self, state):
        x = tf.cast(state, tf.float32)
        x = x / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x


@gin.configurable
class CostNetwork(tf.keras.Model):
    """The convolutional network used for discriminator"""

    def __init__(self, num_actions, feature_extractor=None, name='Discriminator'):
        """Creates the layers used for calculating logits.

        Args:
          num_actions: int, number of actions.
          name: str, used to create scope for network parameters.
        """
        super(CostNetwork, self).__init__(name=name)

        self.num_actions = num_actions
        self.feature_extractor = feature_extractor
        # Defining layers.
        activation_fn = tf.keras.activations.relu
        # Setting names of the layers manually to make variable names more similar
        # with tf.slim variable names/checkpoints.
        if self.feature_extractor is None:
            self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                                activation=activation_fn, name='Conv')
            self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                                activation=activation_fn, name='Conv')
            self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                                activation=activation_fn, name='Conv')
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                                name='fully_connected')
        else:
            pass
        self.dense2 = tf.keras.layers.Dense(512, name='fully_connected',
                                            trainable=False)
        # self.dense3 = tf.keras.layers.Dense(1024, activation=activation_fn,
        #                                     name='fully_connected')
        # self.dense5 = tf.keras.layers.Dense(1024, activation=activation_fn,
        #                                     name='fully_connected')
        self.dense4 = tf.keras.layers.Dense(1, name='fully_connected')

    def call(self, state, action):
        """Creates the output tensor/op given the state tensor as input.

        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        information on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.

        Parameters created here will have scope according to the `name` argument
        given at `.__init__()` call.
        Args:
          state: Tensor, input tensor.
          action: Tensor, input tensor.
        Returns:
          collections.namedtuple, output ops (graph mode) or output tensors (eager).
        """
        if self.feature_extractor is None:
            x = tf.cast(state, tf.float32)
            x = x / 255
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            x = self.dense1(x)
        else:
            x = self.feature_extractor(state)

        act = tf.cast(action, tf.float32)
        act = self.dense2(act)

        h = tf.keras.layers.concatenate([x, act], axis=-1)
        # h = self.dense3(h)
        # h = self.dense5(h)
        return self.dense4(h)


@gin.configurable
class CriticNetwork(tf.keras.Model):
    """The convolutional network used for critic"""

    def __init__(self, num_actions, name='discriminator'):
        """Creates the layers used for calculating logits.

        Args:
          num_actions: int, number of actions.
          name: str, used to create scope for network parameters.
        """
        super(CriticNetwork, self).__init__(name=name)

        self.num_actions = num_actions
        # Defining layers.
        activation_fn = tf.keras.activations.relu
        # Setting names of the layers manually to make variable names more similar
        # with tf.slim variable names/checkpoints.
        self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                            activation=activation_fn, name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                            activation=activation_fn, name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                            activation=activation_fn, name='Conv')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                            name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(1, name='fully_connected')

    def call(self, state):
        """Creates the output tensor/op given the state tensor as input.

        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        information on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.

        Parameters created here will have scope according to the `name` argument
        given at `.__init__()` call.
        Args:
          state: Tensor, input tensor.
          action: Tensor, input tensor.
        Returns:
          collections.namedtuple, output ops (graph mode) or output tensors (eager).
        """
        x = tf.cast(state, tf.float32)
        x = x / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)

        return self.dense2(x)


@gin.configurable
class DWBCDiscriminatorNetwork(tf.keras.Model):
    """The convolutional network used for discriminator"""

    def __init__(self, num_actions, feature_extractor=None, name='Discriminator'):
        """Creates the layers used for calculating logits.

        Args:
          num_actions: int, number of actions.
          name: str, used to create scope for network parameters.
        """
        super(DWBCDiscriminatorNetwork, self).__init__(name=name)

        self.num_actions = num_actions
        self.feature_extractor = feature_extractor
        # Defining layers.
        activation_fn = tf.keras.activations.relu
        # Setting names of the layers manually to make variable names more similar
        # with tf.slim variable names/checkpoints.
        if self.feature_extractor is None:
            self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                                activation=activation_fn, name='Conv')
            self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                                activation=activation_fn, name='Conv')
            self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                                activation=activation_fn, name='Conv')
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                                name='fully_connected')
        else:
            pass
        self.dense2 = tf.keras.layers.Dense(512, name='fully_connected',
                                            trainable=False)
        # self.dense3 = tf.keras.layers.Dense(1024, activation=activation_fn,
        #                                     name='fully_connected')
        # self.dense5 = tf.keras.layers.Dense(1024, activation=activation_fn,
        #                                     name='fully_connected')
        self.dense4 = tf.keras.layers.Dense(1, name='fully_connected')

    def call(self, state, action, log_pi):
        """Creates the output tensor/op given the state tensor as input.

        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        information on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.

        Parameters created here will have scope according to the `name` argument
        given at `.__init__()` call.
        Args:
          state: Tensor, input tensor.
          action: Tensor, input tensor.
          log_pi: Tensor, input tensor.
        Returns:
          collections.namedtuple, output ops (graph mode) or output tensors (eager).
        """
        if self.feature_extractor is None:
            x = tf.cast(state, tf.float32)
            x = x / 255
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            x = self.dense1(x)
        else:
            x = self.feature_extractor(state)

        act = tf.cast(action, tf.float32)
        act = self.dense2(act)

        h = tf.keras.layers.concatenate([x, act, log_pi], axis=-1)
        # h = self.dense3(h)
        # h = self.dense5(h)
        return self.dense4(h)

    def compute_features(self, state):
        x = tf.cast(state, tf.float32)
        x = x / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x

