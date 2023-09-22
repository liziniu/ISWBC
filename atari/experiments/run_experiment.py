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

"""Runner for experiments with a fixed replay buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
from tqdm import tqdm

from absl import logging

import numpy as np
import gin
import tensorflow.compat.v1 as tf
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment


@gin.configurable
class FixedReplayRunner(run_experiment.Runner):
    """Object that handles running Dopamine experiments with fixed replay buffer."""

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        super(FixedReplayRunner, self)._initialize_checkpointer_and_maybe_resume(
            checkpoint_file_prefix)

        # Code for the loading a checkpoint at initialization
        init_checkpoint_dir = self._agent._init_checkpoint_dir  # pylint: disable=protected-access
        if (self._start_iteration == 0) and (init_checkpoint_dir is not None):
            if checkpointer.get_latest_checkpoint_number(self._checkpoint_dir) < 0:
                # No checkpoint loaded yet, read init_checkpoint_dir
                init_checkpointer = checkpointer.Checkpointer(
                    init_checkpoint_dir, checkpoint_file_prefix)
                latest_init_checkpoint = checkpointer.get_latest_checkpoint_number(
                    init_checkpoint_dir)
                if latest_init_checkpoint >= 0:
                    experiment_data = init_checkpointer.load_checkpoint(
                        latest_init_checkpoint)
                    if self._agent.unbundle(
                            init_checkpoint_dir, latest_init_checkpoint, experiment_data):
                        if experiment_data is not None:
                            assert 'logs' in experiment_data
                            assert 'current_iteration' in experiment_data
                            # self._logger.data = experiment_data['logs']
                            # self._start_iteration = experiment_data['current_iteration'] + 1
                        tf.logging.info(
                            'Reloaded checkpoint from %s and will start from iteration %d',
                            init_checkpoint_dir, self._start_iteration)

    def _run_train_phase(self):
        """Run training phase."""
        self._agent.eval_mode = False
        start_time = time.time()
        for _ in tqdm(list(range(self._training_steps))):
            self._agent._train_step()  # pylint: disable=protected-access
        time_delta = time.time() - start_time
        tf.logging.info('Average training steps per second: %.2f',
                        self._training_steps / time_delta)

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction."""
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info('Starting iteration %d', iteration)
        # pylint: disable=protected-access
        # if not self._agent.replay_suffix:
        #     # Reload the replay buffer
        #     self._agent._replay.memory.reload_buffer(num_buffers=5)
        # pylint: enable=protected-access
        self._run_train_phase()

        num_episodes_eval, average_reward_eval, average_clipped_reward_eval, average_reward_event_eval, \
            average_episode_length_eval = self._run_eval_phase(statistics)

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Eval/NumEpisodes',
                             simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Eval/AverageReturns',
                             simple_value=average_reward_eval),
            tf.Summary.Value(tag='Eval/AverageClippedReturns',
                             simple_value=average_clipped_reward_eval),
            tf.Summary.Value(tag='Eval/AverageReturnEvents',
                             simple_value=average_reward_event_eval),
            tf.Summary.Value(tag='Eval/AverageEpisodeLengths',
                             simple_value=average_episode_length_eval)
        ])
        self._summary_writer.add_summary(summary, iteration)
        return statistics.data_lists

    def _run_one_phase(self, min_steps, statistics, run_mode_str):
        """Runs the agent/environment loop until a desired number of steps.

        We follow the Machado et al., 2017 convention of running full episodes,
        and terminating once we've run a minimum number of steps.

        Args:
          min_steps: int, minimum number of steps to generate in this phase.
          statistics: `IterationStatistics` object which records the experimental
            results.
          run_mode_str: str, describes the run mode for this agent.

        Returns:
          Tuple containing the number of steps taken in this phase (int), the sum of
            returns (float), and the number of episodes performed (int).
        """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.
        sum_clipped_returns = 0.
        sum_return_events = 0

        while step_count < min_steps and num_episodes < 10:
            episode_length, episode_return, episode_clipped_return, episode_return_event = self._run_one_episode()
            statistics.append({
                '{}_episode_lengths'.format(run_mode_str): episode_length,
                '{}_episode_returns'.format(run_mode_str): episode_return,
                '{}_episode_clipped_returns'.format(run_mode_str): episode_clipped_return,
                '{}_episode_return_events'.format(run_mode_str): episode_return_event
            })
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1
            sum_clipped_returns += episode_clipped_return
            sum_return_events += episode_return_event
            # We use sys.stdout.write instead of logging so as to flush frequently
            # without generating a line break.
            sys.stdout.write('Steps executed: {} '.format(step_count) +
                             'Episode length: {} '.format(episode_length) +
                             'Return: {}\r'.format(episode_return))
            sys.stdout.flush()
        return step_count, sum_returns, num_episodes, sum_clipped_returns, sum_return_events

    def _run_eval_phase(self, statistics):
        """Run evaluation phase.

        Args:
          statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.

        Returns:
          num_episodes: int, The number of episodes run in this phase.
          average_reward: float, The average reward generated in this phase.
        """
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True
        sum_steps, sum_returns, num_episodes, sum_clipped_returns, sum_return_events = self._run_one_phase(
            self._evaluation_steps, statistics, 'eval')

        average_episode_length = sum_steps / num_episodes if num_episodes > 0 else sum_steps
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        average_clipped_return = sum_clipped_returns / num_episodes if num_episodes > 0 else 0.0
        average_return_event = sum_return_events / num_episodes if num_episodes > 0 else 0.0

        logging.info('Average undiscounted return per evaluation episode: %.2f',
                     average_return)

        statistics.append({'eval_average_episode_length': average_episode_length})
        statistics.append({'eval_average_return': average_return})
        statistics.append({'eval_average_clipped_return': average_clipped_return})
        statistics.append({'eval_average_return_event': average_return_event})
        return num_episodes, average_return, average_clipped_return, average_return_event, average_episode_length

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.

        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.
        total_clipped_reward = 0.
        total_reward_event = 0

        action = self._initialize_episode()
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, is_terminal = self._run_one_step(action)

            total_reward += reward
            step_number += 1
            total_reward_event += float(reward != 0)

            if self._clip_rewards:
                # Perform reward clipping.
                reward = np.clip(reward, -1, 1)
            total_clipped_reward += np.clip(reward, -1, 1)

            if (self._environment.game_over or
                    step_number == self._max_steps_per_episode):
                # Stop the run loop once we reach the true end of episode.
                break
            elif is_terminal:
                # If we lose a life but the episode is not over, signal an artificial
                # end of episode to the agent.
                self._end_episode(reward, is_terminal)
                action = self._agent.begin_episode(observation)
            else:
                action = self._agent.step(reward, observation)

        self._end_episode(reward, is_terminal)

        return step_number, total_reward, total_clipped_reward, total_reward_event

    def _save_tensorboard_summaries(self, iteration,
                                    num_episodes_train,
                                    average_reward_train,
                                    num_episodes_eval,
                                    average_reward_eval,
                                    average_steps_per_second):
        raise NotImplementedError
