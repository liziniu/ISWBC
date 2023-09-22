import os
import numpy as np
import h5py


def concat_trajectories(trajectories):
    return np.concatenate(trajectories, 0)


def get_dataset(dir_name="dataset/rlkit", env_id="Hopper-v2", data_name="full_replay-v2"):
    return get_replay_dataset(dir_name, env_id, data_name)


def get_replay_dataset(dir_name="dataset/rlkit", env_id="Hopper-v2", data_name="full_replay-v2"):
    env_id = env_id.strip("-v2").lower()
    file_name = f'{env_id}_{data_name}'
    file_path = os.path.join(dir_name, file_name + '.hdf5')

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    dataset = h5py.File(file_path, 'r')

    return {
        'observations': np.array(dataset["observations"]).astype(np.float32),
        'actions': np.array(dataset["actions"]).astype(np.float32),
        'next_observations': np.array(dataset["next_observations"]).astype(np.float32),
        'rewards': np.array(dataset["rewards"]).astype(np.float32),
        'terminals': np.array(dataset["terminals"]).astype(np.bool),
    }


def subsample_trajectories(dataset, num_trajectories):
    states, actions, next_states, rewards, dones = (
        dataset["observations"], dataset["actions"], dataset["next_observations"],
        dataset["rewards"], dataset["terminals"])
    states_traj = [[]]
    actions_traj = [[]]
    next_states_traj = [[]]
    rewards_traj = [[]]
    dones_traj = [[]]

    done_indices = [0]
    max_episode_steps = 1000
    for i in range(states.shape[0]):
        states_traj[-1].append(states[i])
        actions_traj[-1].append(actions[i])
        next_states_traj[-1].append(next_states[i])
        rewards_traj[-1].append(rewards[i])
        dones_traj[-1].append(dones[i])

        k = len(dones_traj[-1])
        done = dones[i] or k == max_episode_steps
        if done and i < states.shape[0] - 1:
            print('Subsample trajectory %d (len=%d)' % (len(done_indices), i - done_indices[-1]))
            done_indices.append(i)
            states_traj.append([])
            actions_traj.append([])
            next_states_traj.append([])
            rewards_traj.append([])
            dones_traj.append([])

    shuffle_inds = list(range(len(states_traj)))
    np.random.shuffle(shuffle_inds)
    shuffle_inds = shuffle_inds[:num_trajectories]
    print('Subsample from {}'.format(shuffle_inds))

    states_traj = [states_traj[i] for i in shuffle_inds]
    actions_traj = [actions_traj[i] for i in shuffle_inds]
    next_states_traj = [next_states_traj[i] for i in shuffle_inds]
    rewards_traj = [rewards_traj[i] for i in shuffle_inds]
    dones_traj = [dones_traj[i] for i in shuffle_inds]

    dataset_ = {
        'observations': concat_trajectories(states_traj),
        'actions': concat_trajectories(actions_traj),
        'next_observations': concat_trajectories(next_states_traj),
        'rewards': concat_trajectories(rewards_traj),
        'terminals': concat_trajectories(dones_traj),
    }
    if "init_observations" in dataset:
        dataset_["init_observations"] = dataset["init_observations"][shuffle_inds]

    return dataset_


def add_absorbing_states(expert_states, expert_actions, expert_next_states,
                         expert_dones, env, dtype=np.float32):
    """Adds absorbing states to trajectories.
    Args:
      expert_states: A numpy array with expert states.
      expert_actions: A numpy array with expert states.
      expert_next_states: A numpy array with expert states.
      expert_dones: A numpy array with expert states.
      env: A gym environment.
    Returns:
        Numpy arrays that contain states, actions, next_states and dones.
    """

    # First add 0 indicator to all non-absorbing states.
    expert_states = np.pad(expert_states, ((0, 0), (0, 1)), mode='constant')
    expert_next_states = np.pad(
        expert_next_states, ((0, 0), (0, 1)), mode='constant')

    expert_states = [x for x in expert_states]
    expert_next_states = [x for x in expert_next_states]
    expert_actions = [x for x in expert_actions]
    expert_dones = [x for x in expert_dones]

    # Add absorbing states.
    i = 0
    current_len = 0
    while i < len(expert_states):
        current_len += 1
        if expert_dones[i] and current_len < env._max_episode_steps:  # pylint: disable=protected-access
            current_len = 0
            expert_states.insert(i + 1, env.get_absorbing_state())
            expert_next_states[i] = env.get_absorbing_state()
            expert_next_states.insert(i + 1, env.get_absorbing_state())
            action_dim = env.action_space.n if env.action_space.dtype == int else env.action_space.shape[0]
            expert_actions.insert(i + 1, np.zeros((action_dim,), dtype=dtype))
            expert_dones[i] = 0.0
            expert_dones.insert(i + 1, 1.0)
            i += 1
        i += 1

    expert_states = np.stack(expert_states)
    expert_next_states = np.stack(expert_next_states)
    expert_actions = np.stack(expert_actions)
    expert_dones = np.stack(expert_dones)

    return expert_states.astype(dtype), expert_actions.astype(dtype), expert_next_states.astype(
        dtype), expert_dones.astype(dtype)

