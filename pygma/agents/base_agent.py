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
"""Class to represent BaseAgent.

BaseAgent is used to define an abstract class of pigma's ``Agent`` concept.
"""

import abc


class BaseAgent(abc.ABC):
    """Base class for pigma's `agent` concept."""

    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    @property
    @abc.abstractmethod
    def policy(self):
        """Returns agent's policy."""
        pass

    @abc.abstractmethod
    def train(self, obs, acs, rews, num_steps):
        """Trains agent's policy.

        Args:
            obs: Observations, numpy array
            acs: Actions, numpy array
            rews: Rewards, numpy array
            num_steps: Number of gradient descent steps in training, int
        """
        pass

    @abc.abstractmethod
    def update_policy(self, obs, next_obs, acs, rews, terminals):
        """Updates agent's policy.

        Args:
            obs: Observations, numpy array
            next_obs: Next observations, numpy array
            acs: Actions, numpy array
            rews: Rewards, numpy array
            terminals: Terminals, 1 if rollout has been done, 0 otherwise, numpy array
        """
        pass

    def get_action(self, obs):
        """Returns action for specific observation.

        Args:
            obs: observation of the environment.

        Returns:
            An action which is recommended by agents' policy.    
        """
        return self.policy.get_action(obs)
