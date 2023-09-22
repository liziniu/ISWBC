import os
import time
import argparse
import yaml
from tqdm import tqdm

import numpy as np
import tensorflow.compat.v2 as tf

import wrappers
import dataset_utils
import io_utils
import iswbc


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, default="iswbc")
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--env-id', type=str, default='Hopper-v2')
    parser.add_argument('--experiment', type=str, default='noisy_expert')

    parser.add_argument('--dataset-dir', type=str, default="dataset/rlkit")
    parser.add_argument('--num-expert-trajectory', type=int, default=1)
    parser.add_argument('--stochastic-policy', type=int, default=0, choices=[0, 1])
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--grad-reg-coef', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--max-time-steps', type=int, default=int(1e6))
    parser.add_argument('--using-absorbing', type=int, default=0, choices=[0, 1])
    parser.add_argument('--batch-size', type=int, default=256)

    parser.add_argument('--eval-interval', type=int, default=int(1e4))

    return parser.parse_args()


def evaluate(env, actor, train_env_id, num_episodes=10):
    """Evaluates the policy.
    Args:
        actor: A policy to evaluate
        env: Environment to evaluate the policy on
        train_env_id: train_env_id to compute normalized score
        num_episodes: A number of episodes to average the policy on
    Returns:
        Averaged reward and a total number of steps.
    """
    total_timesteps = 0
    total_returns = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if 'ant' in train_env_id.lower():
                state = state[:27]
            action = actor.step(state)[0].numpy()

            next_state, reward, done, _ = env.step(action)

            total_returns += reward
            total_timesteps += 1
            state = next_state

    mean_score = total_returns / num_episodes
    mean_timesteps = total_timesteps / num_episodes
    return mean_score, mean_timesteps


def main(args=arg_parse()):
    seed = args.seed
    io_utils.set_global_seed(seed)
    save_dir = os.path.join(
        'log',
        '{}-{}-{}-{}'.format(args.algo, args.env_id, args.seed, time.strftime('%Y-%m-%d-%H-%M-%S'))
    )
    logger = io_utils.configure_logger(save_dir)
    io_utils.configure_gpu()
    io_utils.save_code(save_dir)
    yaml.safe_dump(args.__dict__, open(os.path.join(save_dir, 'config.yml'), 'w'), default_flow_style=False)

    hparam_str_dict = dict(
        seed=args.seed, algo=args.algo, env_name=args.env_id)
    hparam_str = ','.join(['%s=%s' % (k, str(hparam_str_dict[k])) for k in
                           sorted(hparam_str_dict.keys())])
    summary_writer = tf.summary.create_file_writer(
        os.path.join(save_dir, 'tb', hparam_str))

    expert_dataset = dataset_utils.get_dataset(args.dataset_dir, args.env_id, "expert-v2")
    imperfect_dataset1 = dataset_utils.subsample_trajectories(expert_dataset, 5)
    imperfect_dataset2 = dataset_utils.subsample_trajectories(expert_dataset, 10)

    expert_dataset = dataset_utils.subsample_trajectories(expert_dataset, args.num_expert_trajectory)
    expert_states, expert_actions = expert_dataset["observations"], expert_dataset["actions"]
    logger.info('# of expert demonstrations: {}'.format(expert_states.shape[0]))

    imperfect_dataset1["actions"] = np.random.uniform(-1., 1., size=imperfect_dataset1["actions"].shape).astype(
        imperfect_dataset1["actions"].dtype
    )
    imperfect_dataset = {
        "observations": np.concatenate([imperfect_dataset1["observations"], imperfect_dataset2["observations"]]),
        "actions": np.concatenate([imperfect_dataset1["actions"], imperfect_dataset2["actions"]]),
        "next_observations": np.concatenate([imperfect_dataset1["next_observations"],
                                             imperfect_dataset2["next_observations"]])
    }
    imperfect_states, imperfect_actions = imperfect_dataset["observations"], imperfect_dataset["actions"]
    logger.info('# of imperfect demonstrations: {}'.format(imperfect_states.shape[0]))

    union_states = np.concatenate([expert_states, imperfect_states])
    union_actions = np.concatenate([expert_actions, imperfect_actions])

    if 'ant' in args.env_id.lower():
        expert_states = expert_states[:, :27]
        imperfect_states = imperfect_states[:, :27]
        union_states = union_states[:, :27]

    shift = -np.mean(imperfect_states, 0)
    scale = 1.0 / (np.std(imperfect_states, 0) + 1e-3)

    expert_states = (expert_states + shift) * scale
    imperfect_states = (imperfect_states + shift) * scale
    union_states = (union_states + shift) * scale

    if 'ant' in args.env_id.lower():
        shift_env = np.concatenate((shift, np.zeros(84)))
        scale_env = np.concatenate((scale, np.ones(84)))
    else:
        shift_env = shift
        scale_env = scale
    env = wrappers.create_il_env(args.env_id, seed, shift_env, scale_env, normalized_box_actions=False,
                                 use_absorbing=args.using_absorbing)

    if args.using_absorbing:
        raise NotImplementedError("BC does not require absorbing states.")

    if 'ant' in args.env_id.lower():
        observation_dim = 27
    else:
        observation_dim = env.observation_space.shape[0]

    # Create imitator
    is_discrete_action = env.action_space.dtype == int
    action_dim = env.action_space.n if is_discrete_action else env.action_space.shape[0]

    imitator = iswbc.ISWBC(observation_dim, action_dim,
                           actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                           grad_reg_coef=args.grad_reg_coef,
                           tau=args.tau)
    training_time = 0
    num_iter = 0
    with tqdm(total=args.max_time_steps, initial=num_iter, desc='') as pbar:
        while num_iter < args.max_time_steps:
            ts = time.time()

            expert_indices = np.random.randint(0, len(expert_states), size=args.batch_size)
            union_indices = np.random.randint(0, len(union_states), size=args.batch_size)

            info_dict = imitator.update(
                expert_states=tf.convert_to_tensor(expert_states[expert_indices]),
                expert_actions=tf.convert_to_tensor(expert_actions[expert_indices]),
                union_states=tf.convert_to_tensor(union_states[union_indices]),
                union_actions=tf.convert_to_tensor(union_actions[union_indices])
            )

            training_time += time.time() - ts

            if num_iter % 1000 == 0:
                with summary_writer.as_default():
                    for key, val in info_dict.items():
                        tf.summary.scalar(
                            'loss/%s' % key, val, step=num_iter)

            if num_iter % args.eval_interval == 0:
                average_returns, average_lengths = evaluate(env, imitator, args.env_id)

                with summary_writer.as_default():
                    tf.summary.scalar(
                        'eval gym/average returns', average_returns, step=num_iter)
                    tf.summary.scalar(
                        'eval gym/average lengths', average_lengths, step=num_iter)

                logger.info('Eval at %d: ave returns=%.2f, ave episode length=%d',
                            num_iter, average_returns, average_lengths)

            num_iter += 1
            pbar.update(1)

    logger.info(f"Training time: {training_time}")


if __name__ == "__main__":
    main(arg_parse())
