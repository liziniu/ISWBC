import numpy as np
import tensorflow as tf
from tensorflow_gan.python.losses import losses_impl as tfgan_losses
from tf_agents.networks import network, utils
from tf_agents.specs.tensor_spec import TensorSpec

from bc import StochasticActor
import os

EPS = np.finfo(np.float32).eps
EPS2 = 1e-3


class Critic(network.Network):
    def __init__(self, state_dim, action_dim, hidden_size=256, kernel_initializer='he_normal', name='ValueNetwork'):
        self._input_specs = TensorSpec(state_dim + action_dim)

        super(Critic, self).__init__(self._input_specs, state_spec=(), name=name)

        hidden_sizes = (hidden_size, hidden_size)

        self._fc_layers = utils.mlp_layers(fc_layer_params=hidden_sizes, activation_fn=tf.nn.relu,
                                           kernel_initializer=kernel_initializer, name='mlp')
        self._last_layer = tf.keras.layers.Dense(1, activation=None, name='value')
        # self._last_layer = tf.keras.layers.Dense(1, activation=None, use_bias=False,
        #                                          kernel_initializer=kernel_initializer, name='value')

    def call(self, inputs, step_type=(), network_state=(), training=False):
        del step_type  # unused
        h = inputs
        for layer in self._fc_layers:
            h = layer(h, training=training)
        h = self._last_layer(h)
        h = tf.reshape(h, [-1])

        return h, network_state


class ISWBC(tf.keras.layers.Layer):
    """
    Importance sampling weighted behavior cloning.
    """
    def __init__(self, state_dim, action_dim, hidden_size=256, actor_lr=1e-4, critic_lr=1e-4,
                 grad_reg_coef=1.0, stochastic_policy=True, tau=0.0, version="v1"):
        super(ISWBC, self).__init__()
        self.grad_reg_coef = grad_reg_coef
        self.stochastic_policy = stochastic_policy
        self.tau = tau
        self.version = version

        self.cost = Critic(state_dim, action_dim, hidden_size=hidden_size)
        if stochastic_policy:
            self.actor = StochasticActor(state_dim, action_dim, hidden_size=hidden_size)
            self.actor.create_variables()
        else:
            raise NotImplementedError

        self.cost.create_variables()

        self.cost_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    @tf.function
    def update(self, expert_states, expert_actions, union_states, union_actions):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(self.cost.variables)
            tape.watch(self.actor.variables)

            # define inputs
            expert_inputs = tf.concat([expert_states, expert_actions], -1)
            union_inputs = tf.concat([union_states, union_actions], -1)

            # call cost functions
            expert_cost_val, _ = self.cost(expert_inputs)
            union_cost_val, _ = self.cost(union_inputs)
            unif_rand = tf.random.uniform(shape=(expert_states.shape[0], 1))
            mixed_inputs1 = unif_rand * expert_inputs + (1 - unif_rand) * union_inputs
            mixed_inputs2 = unif_rand * tf.random.shuffle(union_inputs) + (1 - unif_rand) * union_inputs
            mixed_inputs = tf.concat([mixed_inputs1, mixed_inputs2], 0)

            # gradient penalty for cost
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(mixed_inputs)
                cost_output, _ = self.cost(mixed_inputs)
                cost_output = tf.math.log(1 / (tf.nn.sigmoid(cost_output) + EPS2) - 1 + EPS2)
            cost_mixed_grad = tape2.gradient(cost_output, [mixed_inputs])[0] + EPS
            cost_grad_penalty = tf.reduce_mean(
                tf.square(tf.norm(cost_mixed_grad, axis=-1, keepdims=True) - 1))
            cost_loss = tfgan_losses.minimax_discriminator_loss(expert_cost_val, union_cost_val, label_smoothing=0.) \
                        + self.grad_reg_coef * cost_grad_penalty

            # weighted BC
            # v1 follows DemoDICE and v2 follows our own paper
            # this implementation difference does not matter
            if self.version == "v1":
                union_cost = tf.math.log(1 / (tf.nn.sigmoid(union_cost_val) + EPS2) - 1 + EPS2)
                weight = tf.expand_dims(tf.math.exp(-union_cost), 1)
            else:
                cost_prob = tf.nn.sigmoid(union_cost_val)
                weight = tf.expand_dims(cost_prob / (1 - cost_prob), 1)

            indices = tf.cast(weight >= self.tau, tf.float32)
            if self.stochastic_policy:
                pi_loss = - tf.reduce_mean(
                    indices * tf.stop_gradient(weight) * self.actor.get_log_prob(union_states, union_actions))
            else:
                raise NotImplementedError

        cost_grads = tape.gradient(cost_loss, self.cost.variables)
        pi_grads = tape.gradient(pi_loss, self.actor.variables)
        self.cost_optimizer.apply_gradients(zip(cost_grads, self.cost.variables))
        self.actor_optimizer.apply_gradients(zip(pi_grads, self.actor.variables))
        info_dict = {
            'cost_loss': cost_loss,
            'actor_loss': pi_loss,
        }
        del tape
        return info_dict

    @tf.function
    def step(self, observation, deterministic: bool = True):
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        all_actions, _ = self.actor(observation)
        if deterministic:
            actions = all_actions[0]
        else:
            actions = all_actions[1]
        return actions

    def save(self, filepath, *args):
        self.actor.save_weights(os.path.join(filepath, "actor"))
        self.cost.save_weights(os.path.join(filepath, "cost"))

    def load(self, filepath):
        self.actor.load_weights(os.path.join(filepath, "actor"))
        self.cost.save_weights(os.path.join(filepath, "cost"))
