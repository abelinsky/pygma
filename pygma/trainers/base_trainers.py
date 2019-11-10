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
import abc
import numpy as np
import time
from pygma.policies import base_policy
from pygma.agents import policy_gradient_agent


class BaseTrainer(abc.ABC):
    """Base trainer responsible for training RL agents.

    Attributes:
      env: Environment.
      min_batch_size: Minimum size of the batch during training, int.
      max_rollout_length: Maximum size of each rollout, int.
      render: Indicates whether to render environment, bool.
    """

    def __init__(self,
                 env,
                 min_batch_size=1000,
                 max_rollout_length=100,
                 render=True):
        self.env = env
        self.min_batch_size = min_batch_size
        self.max_rollout_length = max_rollout_length
        self.render = render

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
            batch, batch_size = self.sample_rollouts_batch(
                self.min_batch_size, self.max_rollout_length, self.render)

            # train agent
            self.agent.train()

    def sample_rollouts_batch(self, min_batch_size, max_rollout_length, render=True):
        """Samples one batch of the rollouts (trajectories) from the agent's 
           behavior in the environment.

        Args:
            min_batch_size: Minimum size of transitions in the batch, int
            max_rollout_length: Maximum size of each rollout, int
            render: Indicates whether to render environment, bool. Defaults to True.
        """
        batch_size = 0
        batch = []
        while batch_size <= min_batch_size:
            rollout = self.sample_rollout(max_rollout_length, render)
            batch.append(rollout)
            batch_size += BaseTrainer.get_rollout_size(rollout)
        return batch, batch_size

    def sample_rollout(self, max_rollout_length, render=True):
        """Samples one rollout from the agent's behavior in the environment.

        Args:
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
            ac = self.agent.policy.get_action(ob)
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

        return {"observation": np.array(obs, dtype=np.float32),
                "reward": np.array(rewards, dtype=np.float32),
                "action": np.array(acs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32)}


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
                 render=True):
        super().__init__(
            env, min_batch_size, max_rollout_length, render)

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
                                                                activation=activation_function)

    @property
    def agent(self):
        """See base class."""
        return self._agent
