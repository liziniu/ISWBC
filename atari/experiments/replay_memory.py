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

"""Logged Replay Buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from concurrent import futures

import time

import gin
import numpy as np
import tensorflow.compat.v1 as tf
from dopamine.replay_memory import circular_replay_buffer
from absl import logging

gfile = tf.gfile

STORE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX


def timeit(f):
    def wrap_f(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        print('Run {} using {} sec'.format(f, end_time - start_time))
        return result
    return wrap_f


class FixedReplayBuffer(object):
    """Object composed of a list of OutofGraphReplayBuffers."""

    def __init__(self, data_dir, replay_suffix, num_actions, noisy_action, *args,
                 **kwargs):  # pylint: disable=keyword-arg-before-vararg
        """Initialize the FixedReplayBuffer class.

        Args:
          data_dir: str, log Directory from which to load the replay buffer.
          replay_suffix: int, If not None, then only load the replay buffer
            corresponding to the specific suffix in data directory.
          *args: Arbitrary extra arguments.
          **kwargs: Arbitrary keyword arguments.
        """
        self._args = args
        self._kwargs = kwargs
        self._data_dir = data_dir
        self._loaded_buffers = False
        self.add_count = np.array(0)
        self._replay_suffix = replay_suffix
        self._num_actions = num_actions
        self._noisy_action = noisy_action
        while not self._loaded_buffers:
            if replay_suffix:
                # assert replay_suffix >= 0, 'Please pass a non-negative replay suffix'
                if isinstance(replay_suffix, int):
                    self._replay_suffix = [replay_suffix]
                else:
                    assert isinstance(replay_suffix, list)
                self._load_replay_buffers(len(self._replay_suffix), self._replay_suffix)
                # self.load_single_buffer(replay_suffix)
            else:
                self._load_replay_buffers(num_buffers=1)

    def load_single_buffer(self, suffix):
        """Load a single replay buffer."""
        replay_buffer = self._load_buffer(suffix)
        if replay_buffer is not None:
            self._replay_buffers = [replay_buffer]
            self.add_count = replay_buffer.add_count
            self._num_replay_buffers = 1
            self._loaded_buffers = True

    def _load_buffer(self, suffix):
        """Loads a OutOfGraphReplayBuffer replay buffer."""
        try:
            # pytype: disable=attribute-error
            tf.logging.info(
                f'Starting to load from ckpt {suffix} from {self._data_dir}')
            replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
                *self._args, **self._kwargs)
            replay_buffer.load(self._data_dir, suffix)
            # pylint:disable=protected-access
            replay_capacity = replay_buffer._replay_capacity
            tf.logging.info(f'Capacity: {replay_buffer._replay_capacity}')
            for name, array in replay_buffer._store.items():
                # This frees unused RAM if replay_capacity is smaller than 1M
                replay_buffer._store[name] = array[:replay_capacity + 4].copy()
                tf.logging.info(f'{name}: {array.shape}')
            tf.logging.info('Loaded replay buffer ckpt {} from {}'.format(
                suffix, self._data_dir))
            total_reward = np.sum(replay_buffer._store['reward'])
            num_trajectory = np.sum(replay_buffer._store['terminal'] == True)
            tf.logging.info('replay buffer ckpt {} average return {:.2f}'.format(
                suffix, total_reward / (num_trajectory + 1e-3)
            ))
            # noisy expert task
            if self._noisy_action and suffix in {45}:
                replay_buffer._store['action'] = np.random.randint(self._num_actions,
                                                                   size=len(replay_buffer._store['action']))
                tf.logging.info(f'replay buffer ckpt {suffix} noisy action')
            # pylint:enable=protected-access
            # pytype: enable=attribute-error
            return replay_buffer
        except tf.errors.NotFoundError:
            return None

    def _load_replay_buffers(self, num_buffers=None, ckpt_suffixes=None):
        """Loads multiple checkpoints into a list of replay buffers."""
        if not self._loaded_buffers:  # pytype: disable=attribute-error
            ckpts = gfile.ListDirectory(self._data_dir)  # pytype: disable=attribute-error
            # Assumes that the checkpoints are saved in a format CKPT_NAME.{SUFFIX}.gz
            ckpt_counters = collections.Counter(
                [name.split('.')[-2] for name in ckpts])
            # Should contain the files for add_count, action, observation, reward,
            # terminal and invalid_range
            if ckpt_suffixes is not None:
                assert len(ckpt_suffixes) == num_buffers
            else:
                ckpt_suffixes = [x for x in ckpt_counters if ckpt_counters[x] in [6, 7]]
                if num_buffers is not None:
                    ckpt_suffixes = np.random.choice(
                        ckpt_suffixes, num_buffers, replace=False)
            self._replay_buffers = []
            # Load the replay buffers in parallel
            with futures.ThreadPoolExecutor(
                    max_workers=num_buffers) as thread_pool_executor:
                replay_futures = [thread_pool_executor.submit(
                    self._load_buffer, suffix) for suffix in ckpt_suffixes]
            for f in replay_futures:
                replay_buffer = f.result()
                if replay_buffer is not None:
                    self._replay_buffers.append(replay_buffer)
                    self.add_count = max(replay_buffer.add_count, self.add_count)
            self._num_replay_buffers = len(self._replay_buffers)
            if self._num_replay_buffers:
                self._loaded_buffers = True

    def get_transition_elements(self):
        return self._replay_buffers[0].get_transition_elements()

    def sample_transition_batch(self, batch_size=None, indices=None):
        # buffer_index = np.random.randint(low=0, high=self._num_replay_buffers)
        # return self._replay_buffers[buffer_index].sample_transition_batch(
        #     batch_size=batch_size, indices=indices)

        assert indices is None
        batch_size = batch_size or 256
        splits = np.linspace(0, batch_size, self._num_replay_buffers + 1).astype(np.int32)

        # (state, action, reward, next_state, next_action, next_reward, terminal, indices)
        batch = [[] for _ in range(8)]
        for i in range(self._num_replay_buffers):
            sub_batch_size = splits[i + 1] - splits[i]
            buffer_index = np.random.randint(low=0, high=self._num_replay_buffers)
            sub_batch = self._replay_buffers[buffer_index].sample_transition_batch(
                batch_size=sub_batch_size, indices=indices)
            for j in range(8):
                batch[j].append(sub_batch[j])

        for j in range(8):
            batch[j] = np.concatenate(batch[j], axis=0)
        return tuple(batch)

    def sample_expert_transition_batch(self, batch_size=None, indices=None):
        buffer_index = 0
        return self._replay_buffers[buffer_index].sample_transition_batch(
            batch_size=batch_size, indices=indices)

    def sample_supplementary_transition_batch(self, batch_size=None, indices=None):
        # buffer_index = np.random.randint(low=1, high=self._num_replay_buffers)
        # return self._replay_buffers[buffer_index].sample_transition_batch(
        #     batch_size=batch_size, indices=indices)
        assert indices is None
        batch_size = batch_size or 256
        splits = np.linspace(0, batch_size, self._num_replay_buffers).astype(np.int32)

        # (state, action, reward, next_state, next_action, next_reward, terminal, indices)
        batch = [[] for _ in range(8)]
        for i in range(self._num_replay_buffers - 1):
            sub_batch_size = splits[i + 1] - splits[i]
            buffer_index = np.random.randint(low=1, high=self._num_replay_buffers)
            sub_batch = self._replay_buffers[buffer_index].sample_transition_batch(
                batch_size=sub_batch_size, indices=indices)
            for j in range(8):
                batch[j].append(sub_batch[j])

        for j in range(8):
            batch[j] = np.concatenate(batch[j], axis=0)
        return tuple(batch)

    def sample_noisy_transition_batch(self, batch_size=None, indices=None):
        # Please make sure the buffer index is correct
        buffer_index = np.random.choice([1])
        return self._replay_buffers[buffer_index].sample_transition_batch(
            batch_size=batch_size, indices=indices)

    def sample_clean_transition_batch(self, batch_size=None, indices=None):
        # Please make sure the buffer index is correct
        buffer_index = np.random.choice([2, 3, 4])
        return self._replay_buffers[buffer_index].sample_transition_batch(
            batch_size=batch_size, indices=indices)

    def load(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass

    def reload_buffer(self, num_buffers=None):
        self._loaded_buffers = False
        self._load_replay_buffers(num_buffers)

    def save(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass

    def add(self, *args, **kwargs):  # pylint: disable=unused-argument
        pass


@gin.configurable(denylist=['observation_shape', 'stack_size',
                            'update_horizon', 'gamma'])
class WrappedFixedReplayBuffer(circular_replay_buffer.WrappedReplayBuffer):
    """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism."""

    def __init__(self,
                 data_dir,
                 replay_suffix,
                 observation_shape,
                 stack_size,
                 use_staging=True,
                 replay_capacity=1000000,
                 batch_size=32,
                 update_horizon=1,
                 gamma=0.99,
                 wrapped_memory=None,
                 max_sample_attempts=1000,
                 extra_storage_types=None,
                 noisy_action=False,
                 num_actions=None,
                 observation_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32):
        """Initializes WrappedFixedReplayBuffer."""

        memory = FixedReplayBuffer(
            data_dir, replay_suffix, num_actions, noisy_action, observation_shape, stack_size, replay_capacity,
            batch_size, update_horizon, gamma, max_sample_attempts,
            extra_storage_types=extra_storage_types,
            observation_dtype=observation_dtype)

        super(WrappedFixedReplayBuffer, self).__init__(
            observation_shape,
            stack_size,
            use_staging=use_staging,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            update_horizon=update_horizon,
            gamma=gamma,
            wrapped_memory=memory,
            max_sample_attempts=max_sample_attempts,
            extra_storage_types=extra_storage_types,
            observation_dtype=observation_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            reward_shape=reward_shape,
            reward_dtype=reward_dtype)


@gin.configurable(denylist=['observation_shape', 'stack_size',
                            'update_horizon', 'gamma'])
class WrappedUnionReplayBuffer(circular_replay_buffer.WrappedReplayBuffer):
    def __init__(self,
                 data_dir,
                 replay_suffix,
                 observation_shape,
                 stack_size,
                 use_staging=True,
                 replay_capacity=1000000,
                 batch_size=32,
                 update_horizon=1,
                 gamma=0.99,
                 wrapped_memory=None,
                 max_sample_attempts=1000,
                 extra_storage_types=None,
                 noisy_action=False,
                 num_actions=None,
                 observation_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32):
        """Initializes WrappedFixedReplayBuffer."""

        memory = FixedReplayBuffer(
            data_dir, replay_suffix, num_actions, noisy_action, observation_shape, stack_size, replay_capacity,
            batch_size, update_horizon, gamma, max_sample_attempts,
            extra_storage_types=extra_storage_types,
            observation_dtype=observation_dtype)

        super(WrappedUnionReplayBuffer, self).__init__(
            observation_shape,
            stack_size,
            use_staging=use_staging,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            update_horizon=update_horizon,
            gamma=gamma,
            wrapped_memory=memory,
            max_sample_attempts=max_sample_attempts,
            extra_storage_types=extra_storage_types,
            observation_dtype=observation_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            reward_shape=reward_shape,
            reward_dtype=reward_dtype)

        self.create_expert_sampling_ops(use_staging)
        self.create_supplementary_sampling_ops(use_staging)
        self.create_clean_sampling_ops(use_staging)
        self.create_noisy_sampling_ops(use_staging)

    def create_expert_sampling_ops(self, use_staging):
        if use_staging:
            logging.warning('use_staging=True is no longer supported')
        with tf.name_scope('sample_replay'):
            with tf.device('/cpu:*'):
                transition_type = self.memory.get_transition_elements()

                # properly set
                transition_tensors = tf.numpy_function(
                    self.memory.sample_expert_transition_batch, [],  # here
                    [return_entry.type for return_entry in transition_type],
                    name='replay_sample_py_func')
                self._set_transition_shape(transition_tensors, transition_type)
                # Unpack sample transition into member variables.
                self.unpack_expert_transition(transition_tensors, transition_type)

    def unpack_expert_transition(self, transition_tensors, transition_type):
        """Unpacks the given transition into member variables.

        Args:
          transition_tensors: tuple of tf.Tensors.
          transition_type: tuple of ReplayElements matching transition_tensors.
        """
        self.expert_transition = collections.OrderedDict()
        for element, element_type in zip(transition_tensors, transition_type):
            self.expert_transition[element_type.name] = element

        # TODO(bellemare): These are legacy and should probably be removed in
        # future versions.
        self.expert_states = self.expert_transition['state']
        self.expert_actions = self.expert_transition['action']
        self.expert_rewards = self.expert_transition['reward']
        self.expert_next_states = self.expert_transition['next_state']
        self.expert_next_actions = self.expert_transition['next_action']
        self.expert_next_rewards = self.expert_transition['next_reward']
        self.expert_terminals = self.expert_transition['terminal']
        self.expert_indices = self.expert_transition['indices']

    def create_supplementary_sampling_ops(self, use_staging):
        if use_staging:
            logging.warning('use_staging=True is no longer supported')
        with tf.name_scope('sample_replay'):
            with tf.device('/cpu:*'):
                transition_type = self.memory.get_transition_elements()

                # properly set
                transition_tensors = tf.numpy_function(
                    self.memory.sample_supplementary_transition_batch, [],  # here
                    [return_entry.type for return_entry in transition_type],
                    name='replay_sample_py_func')
                self._set_transition_shape(transition_tensors, transition_type)
                # Unpack sample transition into member variables.
                self.unpack_supplementary_transition(transition_tensors, transition_type)

    def unpack_supplementary_transition(self, transition_tensors, transition_type):
        """Unpacks the given transition into member variables.

        Args:
          transition_tensors: tuple of tf.Tensors.
          transition_type: tuple of ReplayElements matching transition_tensors.
        """
        self.supply_transition = collections.OrderedDict()
        for element, element_type in zip(transition_tensors, transition_type):
            self.supply_transition[element_type.name] = element

        # TODO(bellemare): These are legacy and should probably be removed in
        # future versions.
        self.supply_states = self.supply_transition['state']
        self.supply_actions = self.supply_transition['action']
        self.supply_rewards = self.supply_transition['reward']
        self.supply_next_states = self.supply_transition['next_state']
        self.supply_next_actions = self.supply_transition['next_action']
        self.supply_next_rewards = self.supply_transition['next_reward']
        self.supply_terminals = self.supply_transition['terminal']
        self.supply_indices = self.supply_transition['indices']

    def create_noisy_sampling_ops(self, use_staging):
        if use_staging:
            logging.warning('use_staging=True is no longer supported')
        with tf.name_scope('sample_replay'):
            with tf.device('/cpu:*'):
                transition_type = self.memory.get_transition_elements()

                # properly set
                transition_tensors = tf.numpy_function(
                    self.memory.sample_noisy_transition_batch, [],  # here
                    [return_entry.type for return_entry in transition_type],
                    name='replay_sample_py_func')
                self._set_transition_shape(transition_tensors, transition_type)
                # Unpack sample transition into member variables.
                self.unpack_noisy_transition(transition_tensors, transition_type)

    def unpack_noisy_transition(self, transition_tensors, transition_type):
        """Unpacks the given transition into member variables.

        Args:
          transition_tensors: tuple of tf.Tensors.
          transition_type: tuple of ReplayElements matching transition_tensors.
        """
        self.noisy_transition = collections.OrderedDict()
        for element, element_type in zip(transition_tensors, transition_type):
            self.noisy_transition[element_type.name] = element

        # TODO(bellemare): These are legacy and should probably be removed in
        # future versions.
        self.noisy_states = self.noisy_transition['state']
        self.noisy_actions = self.noisy_transition['action']
        self.noisy_rewards = self.noisy_transition['reward']
        self.noisy_next_states = self.noisy_transition['next_state']
        self.noisy_next_actions = self.noisy_transition['next_action']
        self.noisy_next_rewards = self.noisy_transition['next_reward']
        self.noisy_terminals = self.noisy_transition['terminal']
        self.noisy_indices = self.noisy_transition['indices']

    def create_clean_sampling_ops(self, use_staging):
        if use_staging:
            logging.warning('use_staging=True is no longer supported')
        with tf.name_scope('sample_replay'):
            with tf.device('/cpu:*'):
                transition_type = self.memory.get_transition_elements()

                # properly set
                transition_tensors = tf.numpy_function(
                    self.memory.sample_clean_transition_batch, [],  # here
                    [return_entry.type for return_entry in transition_type],
                    name='replay_sample_py_func')
                self._set_transition_shape(transition_tensors, transition_type)
                # Unpack sample transition into member variables.
                self.unpack_clean_transition(transition_tensors, transition_type)

    def unpack_clean_transition(self, transition_tensors, transition_type):
        """Unpacks the given transition into member variables.

        Args:
          transition_tensors: tuple of tf.Tensors.
          transition_type: tuple of ReplayElements matching transition_tensors.
        """
        self.clean_transition = collections.OrderedDict()
        for element, element_type in zip(transition_tensors, transition_type):
            self.clean_transition[element_type.name] = element

        # TODO(bellemare): These are legacy and should probably be removed in
        # future versions.
        self.clean_states = self.clean_transition['state']
        self.clean_actions = self.clean_transition['action']
        self.clean_rewards = self.clean_transition['reward']
        self.clean_next_states = self.clean_transition['next_state']
        self.clean_next_actions = self.clean_transition['next_action']
        self.clean_next_rewards = self.clean_transition['next_reward']
        self.clean_terminals = self.clean_transition['terminal']
        self.clean_indices = self.clean_transition['indices']
