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

r"""The entry point for running experiments with fixed replay datasets.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import tensorflow.compat.v1 as tf
from absl import app
from absl import flags
from dopamine.discrete_domains import run_experiment as base_run_experiment
from dopamine.discrete_domains import atari_lib

from experiments import run_experiment
from experiments.agents.bc_agent import BehaviourCloningAgent
from experiments.agents.nbcu_agent import NaiveBehaviourCloningAgent
from experiments.agents.iswcu_agent import ISWeightedBehaviourCloningAgent

from utils.io_utils import set_global_seed, configure_logger, save_code, save_config


import gin.tf

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
                     '"third_party/py/dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')
flags.DEFINE_string('agent_name', 'bc', 'Name of the agent.')
flags.DEFINE_integer('seed', '2022', 'Random seed.')
flags.DEFINE_string('env_name', None, 'environment name')
flags.DEFINE_string('replay_dir', None, 'Directory from which to load the expert replay data')
# flags.DEFINE_string('replay_suffix', None, 'Directory from which to load the expert replay data')
# flags.DEFINE_string('supplementary_replay_suffix', None, 'Directory from which to load the expert replay data')
flags.DEFINE_string('init_checkpoint_dir', None, 'Directory from which to load '
                                                 'the initial checkpoint before training starts.')

FLAGS = flags.FLAGS


def create_agent(sess, environment, replay_data_dir, summary_writer=None):
    """Creates a DQN agent.

    Args:
      sess: A `tf.Session`object  for running associated ops.
      environment: An Atari 2600 environment.
      replay_data_dir: Directory to which log the replay buffers periodically.
      summary_writer: A Tensorflow summary writer to pass to the agent
        for in-agent training statistics in Tensorboard.

    Returns:
      A DQN agent with metrics.
    """
    if FLAGS.agent_name == 'bc':
        agent = BehaviourCloningAgent
    elif FLAGS.agent_name == 'nbcu':
        agent = NaiveBehaviourCloningAgent
    elif FLAGS.agent_name == 'iswbc':
        agent = ISWeightedBehaviourCloningAgent
    else:
        raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))

    return agent(sess, num_actions=environment.action_space.n,
                 replay_data_dir=replay_data_dir, summary_writer=summary_writer,
                 # replay_suffix=FLAGS.replay_suffix,
                 # supplementary_replay_suffix=FLAGS.supplementary_replay_suffix,
                 init_checkpoint_dir=FLAGS.init_checkpoint_dir)


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    base_run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)

    set_global_seed(FLAGS.seed)
    configure_logger(FLAGS.base_dir)
    save_code(FLAGS.base_dir)
    config = FLAGS.flag_values_dict()
    config.update(gin.config._CONFIG)
    save_config(config, FLAGS.base_dir)

    replay_data_dir = os.path.join(FLAGS.replay_dir, 'replay_logs')
    create_agent_fn = functools.partial(
        create_agent, replay_data_dir=replay_data_dir)

    def create_environment_fn(*args, **kwargs):
        env = atari_lib.create_atari_environment(*args, **kwargs)
        env.environment.seed(FLAGS.seed)
        return env

    runner = run_experiment.FixedReplayRunner(FLAGS.base_dir, create_agent_fn, create_environment_fn)
    runner.run_experiment()


if __name__ == '__main__':
    flags.mark_flag_as_required('replay_dir')
    flags.mark_flag_as_required('base_dir')
    flags.mark_flag_as_required('env_name')
    app.run(main)
