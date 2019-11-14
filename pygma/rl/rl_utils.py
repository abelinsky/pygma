# Copyright (c) 2019 Alexander Belinsky

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ==============================================================================
"""Common utilities for reinforcement learning procedures."""

import numpy as np
import time
import tensorflow as tf


def get_rollout_size(rollout):
    """Gets a rollout's size.

    Args:
      rollout: a rollout (dict, a return from `sample_rollout` func).

    Returns:
      length of the rollout (int)
    """
    return len(rollout["reward"])


def sample_rollout(env, collect_policy, max_rollout_length, render=True):
    """Samples one rollout from the agent's behavior in the environment.

    Args:
      env: Environment to interact with.
      collect_policy: Policy which is used to sample actions (instance of ``BasePolicy``)
      max_rollout_length: Maximum number of steps in the environment for one rollout (int)
      render: Indicates whether to render the environment (boolean). Defaults to True.

    Returns:
        a dict, containing numpy arrays of observations, rewards, actions,
           next observations, terminal signals (under the keys "observation",
           "reward", "action", "next_observation", "terminal")
    """

    # begin new rollout
    ob = env.reset()

    obs, next_obs, acs, rewards, terminals = [], [], [], [], []
    steps = 0
    while True:
        if render:
            env.render()
            time.sleep(0.01)

        obs.append(ob)

        # tf.config.experimental_run_functions_eagerly(True)
        # query the policy
        ac = collect_policy.get_action(ob).numpy()
        ac = ac[0]
        acs.append(ac)

        # performone step in the environment
        ob, rew, done, _ = env.step(ac)

        # record results
        steps += 1
        rewards.append(rew)
        next_obs.append(ob)

        rollout_done = 1 if done or steps >= max_rollout_length else 0
        terminals.append(rollout_done)

        if rollout_done:
            break

    return {'observation': np.array(obs, dtype=np.float32),
            'reward': np.array(rewards, dtype=np.float32),
            'action': np.array(acs, dtype=np.float32),
            'next_observation': np.array(next_obs, dtype=np.float32),
            'terminal': np.array(terminals, dtype=np.float32)}


def sample_rollouts_batch(env, collect_policy, min_batch_size, max_rollout_length, render=True):
    """Samples one batch of the rollouts (trajectories) from the agent's
        behavior in the environment.

    Args:
        env: Environment to interact with.
        collect_policy: Policy which is used to sample actions (instance of ``BasePolicy``)
        min_batch_size: Minimum size of transitions in the batch (int)
        max_rollout_length: Maximum size of each rollout (int)
        render: Indicates whether to render environment (bool). Defaults to True.
    """
    print("\nCollecting data ...")
    batch_size = 0
    batch = []
    while batch_size <= min_batch_size:
        rollout = sample_rollout(
            env, collect_policy, max_rollout_length, render)
        batch.append(rollout)
        batch_size += get_rollout_size(rollout)
    return batch, batch_size


def transform_rollouts_batch(batch):
    """Takes a batch of rollouts and returns separate arrays,
    where each of them is a concatenation of that array from
    across the rollout.

    Args:
      batch: A batch of rollouts (dict).

    Returns:
      arrays of observations, actions, rewards, next_observations, terminals (numpy arrays).
    """
    observations = np.concatenate(
        [rollout['observation'] for rollout in batch])
    actions = np.concatenate([rollout['action'] for rollout in batch])
    concatenated_rewards = np.concatenate(
        [rollout['reward'] for rollout in batch])
    unconcatenated_rewards = np.array(
        [rollout['reward'] for rollout in batch])
    next_observations = np.concatenate(
        [rollout['next_observation'] for rollout in batch])
    terminals = np.concatenate([rollout['terminal'] for rollout in batch])
    return observations, actions, concatenated_rewards, unconcatenated_rewards, next_observations, terminals
