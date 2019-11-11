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
"""Class to represent `agent's` trainer."""
import collections
import abc
import numpy as np
import time
from pygma.policies import base_policy
from pygma.agents import policy_gradient_agent
from pygma.util import logger
import tensorflow as tf


class BaseTrainer(abc.ABC):
    """Base trainer responsible for training RL agents.

    Attributes:
      env: Environment.
      min_batch_size: Minimum size of the batch during training, int.
      max_rollout_length: Maximum size of each rollout, int.
      num_agent_train_steps_per_iter: Number of training steps per each iteration
      logdir: Log directory (is used form tensorboard events)
      render: Indicates whether to render environment, bool.
    """

    def __init__(self, env, min_batch_size=1000, max_rollout_length=100, num_agent_train_steps_per_iter=1,
                 log_metrics=True, logdir=None, render=True):
        self.env = env
        self.min_batch_size = min_batch_size
        self.max_rollout_length = max_rollout_length
        self.num_agent_train_steps_per_iter = num_agent_train_steps_per_iter
        self.render = render
        self.log_metrics = log_metrics
        self.logdir = logdir
        self.log_freq = 10
        # batch size to evaluate policy
        self.eval_batch_size = 400

        if log_metrics:
            self.logger = logger.Logger(logdir)

    @property
    @abc.abstractmethod
    def agent(self):
        pass

    @staticmethod
    def get_rollout_size(rollout):
        """Gets a rollout's size.

        Args:
          rollout: a rollout, a dict (a return from `sample_rollout` func).

        Returns:
          length of the rollout, int
        """
        return len(rollout["reward"])

    def train_agent(self, n_iter):
        for itr in range(n_iter):
            # Generate samples: run current policy :math:`\pi_\theta`
            # and sample a set of trajectories :math:`{\tau^i}`
            # (a sequences of :math:`s_{t}, a_{t}`)
            batch, batch_size = self.sample_rollouts_batch(self.agent.policy,
                                                           self.min_batch_size,
                                                           self.max_rollout_length,
                                                           self.render and itr % 100 == 0)

            # train agent
            obs, acs, conc_rews, unc_rews, next_obs, terminals = BaseTrainer.transform_rollouts_batch(
                batch)

            # perform training
            self.agent.train(obs, acs, unc_rews,
                             self.num_agent_train_steps_per_iter)

            if self.log_metrics and itr % self.log_freq == 0:
                self.log_evaluate(batch, self.env, self.agent.policy, itr)

    def sample_rollouts_batch(self, collect_policy, min_batch_size, max_rollout_length, render=True):
        """Samples one batch of the rollouts (trajectories) from the agent's
           behavior in the environment.

        Args:
          collect_policy: Policy which is used to sample actions,
            instance of `BasePolicy`
          min_batch_size: Minimum size of transitions in the batch, int
          max_rollout_length: Maximum size of each rollout, int
          render: Indicates whether to render environment, bool. Defaults to True.
        """
        batch_size = 0
        batch = []
        while batch_size <= min_batch_size:
            rollout = self.sample_rollout(
                collect_policy, max_rollout_length, render)
            batch.append(rollout)
            batch_size += BaseTrainer.get_rollout_size(rollout)
        return batch, batch_size

    def sample_rollout(self, collect_policy, max_rollout_length, render=True):
        """Samples one rollout from the agent's behavior in the environment.

        Args:
          collect_policy: Policy which is used to sample actions,
            instance of `BasePolicy`
          max_rollout_length: Maximum number of steps in the environment
            for one rollout, it
          render: Indicates whether to render the environment.
            Defaults to True.

        Returns:
            a dict, containing numpy arrays of observations, rewards, actions,
              next observations, terminal signals (under the keys "observation",
              "reward", "action", "next_observation", "terminal")
        """

        if self.agent is None:
            raise AttributeError("Agent in trainer can not be None.")

        # begin new rollout
        ob = self.env.reset()

        obs, next_obs, acs, rewards, terminals = [], [], [], [], []
        steps = 0
        while True:
            if render:
                self.env.render()
                time.sleep(0.01)

            obs.append(ob)

            # query the policy
            ac = collect_policy.get_action(ob).numpy()
            # ac = ac[0]
            acs.append(ac)

            # performone step in the environment
            ob, rew, done, _ = self.env.step(ac)

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

    @staticmethod
    def transform_rollouts_batch(batch):
        """Takes a batch of rollouts and returns separate arrays,
        where each of them is a concatenation of that array from
        across the rollout.

        Args:
          batch: A batch of rollouts, dict (see ``sample_rollout`` func)

        Returns:
           numpy arrays of observations, actions, rewards, next_observations, terminals
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

    def log_evaluate(self, train_batch, env, eval_policy, step):
        """Evaluates the policy and logs train and eval metrics.

        Args:
            train_batch: Batch of rollouts seen in training, numpy array of arrays
            env: Environment
            eval_policy: Policy to use in evaluation, instaince of `BasePolicy`
            step: Current step, int.
        """
        print("\nCollecting data for evaluation...")
        eval_batch, eval_batch_size = self.sample_rollouts_batch(
            collect_policy=eval_policy,
            min_batch_size=self.eval_batch_size,
            max_rollout_length=self.max_rollout_length,
            render=False)

        # save eval metrics
        train_returns = [rollout["reward"].sum() for rollout in train_batch]
        eval_returns = [rollout["reward"].sum() for rollout in eval_batch]

        # episode lengths, for logging
        train_ep_lens = [len(rollout["reward"]) for rollout in train_batch]
        eval_ep_lens = [len(rollout["reward"]) for rollout in eval_batch]

        # decide what to log
        logs = collections.OrderedDict()
        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"] = np.std(eval_returns)
        logs["Eval_MaxReturn"] = np.max(eval_returns)
        logs["Eval_MinReturn"] = np.min(eval_returns)
        logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

        logs["Train_AverageReturn"] = np.mean(train_returns)
        logs["Train_StdReturn"] = np.std(train_returns)
        logs["Train_MaxReturn"] = np.max(train_returns)
        logs["Train_MinReturn"] = np.min(train_returns)
        logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(key, value, step)
        self.logger.flush()


class PolicyGradientTrainer(BaseTrainer):
    """Trainer for `PolicyGradientAgent`.

    Attributes:
        See base class.
    """

    def __init__(self,
                 env,
                 min_batch_size=1000,
                 max_rollout_length=100,
                 n_layers=2,
                 layers_size=64,
                 learning_rate=1e-4,
                 activation_function='relu',
                 is_discrete=False,
                 render=True,
                 gamma=0.99,
                 reward_to_go=False,
                 baseline=False,
                 standardize_advantages=True,
                 logdir=None,
                 **kwargs):
        super().__init__(
            env=env, min_batch_size=min_batch_size, max_rollout_length=max_rollout_length, render=render, logdir=logdir)

        # get dimensions of action and observation spaces
        obs_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if is_discrete else self.env.action_space.shape[0]

        self._agent = policy_gradient_agent.PolicyGradientAgent(env=env,
                                                                action_dim=ac_dim,
                                                                obs_dim=obs_dim,
                                                                n_layers=n_layers,
                                                                layers_size=layers_size,
                                                                is_discrete=is_discrete,
                                                                learning_rate=learning_rate,
                                                                activation=activation_function,
                                                                gamma=gamma,
                                                                reward_to_go=reward_to_go,
                                                                baseline=baseline,
                                                                standardize_advantages=standardize_advantages)

    @property
    def agent(self):
        """See base class."""
        return self._agent
