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
"""Contains the base Reinforcement Learning BaseAgent class, from which all Agents inherit."""

import abc
import collections
import numpy as np
import gym
from pygma.utils import generic_utils, logger
from pygma.rl import rl_utils


class BaseAgent(abc.ABC):
    """Base class for pigma's *agents*.

    Agent refer to RL algorithm.
    This is a class from which all RL agents inherit.

    Note: 
        This is the parent class of all RL agents,
        not an actual agent that can be used for being trained.

    An instance of ``BaseAgent`` class is responsible
    for interacting with environment through it's policy
    and for training policy.

    A ``BaseAgent`` is a class implementing the anatomy of a 
    reinforcement learning algorithm:
    -  generate samples(i.e. run the policy)
    -  fit a model / estimate the return
    -  improve the policy.

    Users will instantiate a ``BaseAgent`` to just add nuances of
    particular agents in particular environments.

    Descendants of ``BaseAgent`` must implement the following 
    abstract methods and properties:

    * ``_train()``: Implement training logic for particular algorithms.
    * ``policy`` property: Returns actual policy for interacting with env.  
    * ``update_policy()``: Run update operation for policy.

    Attributes:
      env: An environment to learn from.
      **kwargs: keyword arguments. Allowed to be {`min_batch_size`, `max_rollout_length`, 
        `num_agent_train_steps_per_iter`, `eval_batch_size`, `render`, `render_freq`,
        `log_metrics`, `logdir`, `log_freq`}. 

        * **min_batch_size** is the minimum size of training batch; 
        * **max_rollout_length** is the maximum number of step per single  rollout; 
        * **num_agent_train_steps_per_iter** is the number of steps of training to be 
          performed within a single batch; 
        * **eval_batch_size** is the size of batch for agent evaluation; 
        * **render** indicates whether to render environment during training; 
        * **render_freq** is the frequency of rendering; 
        * **log_metrics** indicates whether to log events during training; 
        * **logdir** is the path to the directory where to log metrics; 
        * **log_freq** is the frequency of logging metrics.

    """

    def __init__(self, env, **kwargs):

        # TODO: logdir and log_metrics are redundant, logdir (None or not) is suffice, delete log_metrics
        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {
            'min_batch_size',
            'max_rollout_length',
            'num_agent_train_steps_per_iter',
            'eval_batch_size',
            'render',
            'render_freq',
            'log_metrics',
            'logdir',
            'log_freq',
        }
        # Validate optional keyword arguments.
        generic_utils.validate_kwargs(kwargs, allowed_kwargs)

        if env is None:
            raise ValueError("Env to learn from must be set.")

        self._env = env

        # Indicates whether environment is discrete
        self.is_discrete_env = isinstance(
            env.action_space, gym.spaces.Discrete)

        # get dimensions of action and observation spaces
        self.obs_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.n if self.is_discrete_env \
            else self.env.action_space.shape[0]

        # Batchs' parameters for model's training and evaluation.
        self.min_batch_size = kwargs.pop('min_batch_size', 1000)
        self.max_rollout_length = kwargs.pop('max_rollout_length',
                                             env.spec.max_episode_steps)
        self.num_agent_train_steps_per_iter = kwargs.pop(
            'num_agent_train_steps_per_iter', 1)
        self.eval_batch_size = kwargs.pop('eval_batch_size', 400)

        # Rendering and logging settings.
        self.render = kwargs.pop('render', True)
        self.render_freq = kwargs.pop('render_freq', 100)
        self.log_metrics = kwargs.pop('log_metrics', False)
        self.logdir = kwargs.pop('logdir', None)
        self.log_freq = kwargs.pop('log_freq', 1)

        # Create logger for logging events
        if self.log_metrics:
            self.logger = logger.Logger(self.logdir)

    @property
    def env(self):
        return self._env

    @property
    @abc.abstractmethod
    def policy(self):
        """Returns agent's policy.

        Returns:
          ``BasePolicy``: Policy.
        """
        pass

    @property
    def eval_policy(self):
        """Returns policy for evaluation.

        Returns:
          ``BasePolicy``: Policy for evaluation.
        """
        return self.policy

    @abc.abstractmethod
    def _train(self, obs, acs, rews, num_steps):
        """Trains agent's policy.

        Args:
          obs: Observations, numpy array
          acs: Actions, numpy array
          rews: Rewards, numpy array
          num_steps: Number of gradient descent steps in training, int
        """
        pass

    def get_action(self, obs):
        """Returns action for specific observation.

        Args:
          obs: observation of the environment.

        Returns:
          numpy.array: An action which is recommended by agents' policy.
        """
        return self.policy.get_action(obs)

    def run_training_loop(self, n_iter):
        """Performs common loop of training agent. 

        Args:
          n_iter (int): Number of iterations.
        """
        for itr in range(n_iter):
            print(f"\n\n************** Iteration {itr} **************")
            # Generate samples: run current policy :math:`\pi_\theta`
            # and sample a set of trajectories :math:`{\tau^i}`
            # (a sequences of :math:`s_{t}, a_{t}`)
            batch, batch_size = rl_utils.sample_rollouts_batch(
                self.env,
                self.policy,
                self.min_batch_size,
                self.max_rollout_length,
                self.render and itr > 0 and itr % self.render_freq == 0)

            obs, acs, conc_rews, unc_rews, next_obs, terminals = rl_utils.transform_rollouts_batch(
                batch)

            # perform training operations
            self._train(obs, acs, unc_rews,
                        self.num_agent_train_steps_per_iter)

            # evaluate and log results
            if self.log_metrics and itr % self.log_freq == 0:
                self.evaluate_and_log(batch, self.env, self.eval_policy, itr)

    def evaluate_and_log(self, train_batch, env, eval_policy, step):
        """Evaluates the policy and logs train and eval metrics.

        Args:
          train_batch: Batch of rollouts seen in training, numpy array of arrays.
          env: Environment.
          eval_policy (``BasePolicy``): Policy to use in evaluation.
          step (int): Current step.
        """
        print("\nCollecting data for evaluation...")
        eval_batch, eval_batch_size = rl_utils.sample_rollouts_batch(
            env,
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
