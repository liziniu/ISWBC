import tensorflow as tf
import collections

DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])


class NatureDQNNetwork(tf.keras.Model):
    """The convolutional network used to compute the agent's Q-values."""

    def __init__(self, num_actions, feature_extractor=None, name=None):
        """Creates the layers used for calculating Q-values.

        Args:
          num_actions: int, number of actions.
          name: str, used to create scope for network parameters.
        """
        super(NatureDQNNetwork, self).__init__(name=name)

        self.feature_extractor = feature_extractor
        self.num_actions = num_actions
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
        self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')

    def call(self, state):
        """Creates the output tensor/op given the state tensor as input.

        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        information on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.

        Parameters created here will have scope according to the `name` argument
        given at `.__init__()` call.
        Args:
          state: Tensor, input tensor.
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

            q = self.dense2(x)
        else:
            x = self.feature_extractor(state)
            q = self.dense2(x)
        return DQNNetworkType(q)

    def compute_features(self, state):
        assert self.feature_extractor is None
        x = tf.cast(state, tf.float32)
        x = x / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)

        return x
