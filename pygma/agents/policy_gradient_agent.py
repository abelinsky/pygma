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
""" Implementation of policy gradient algorithm."""
from pygma.agents.base_agent import BaseAgent
from pygma.policies.mlp_policy import MLPPolicy


class PolicyGradientAgent(BaseAgent):
    """Policy gradient agent.

    Attributes:
      env: An `Environment` object.
      action_dim: Dimension of actions space, int
      obs_dim: Dimension of observations space, int
      n_layers: Number of hidden layers in policy network, int
      layers_size: Size of hidden layers in policy network, int
      is_discrete: Indicates whether actions are discrete or continuous, bool 
      learning_rate: Learning rate, float
      activation: Activation function in hidden layers of policy network
    """

    def __init__(self, env, action_dim, obs_dim, n_layers, layers_size, is_discrete, learning_rate, activation):
        super(PolicyGradientAgent, self).__init__()
        self.actor = MLPPolicy(
            action_dim=action_dim,
            obs_dim=obs_dim,
            n_layers=n_layers,
            layers_size=layers_size,
            discrete=is_discrete,
            learning_rate=learning_rate,
            activation=activation
        )

    def train(self):
        """See base class."""
        print("Training policy gradient agent ...")

    @property
    def policy(self):
        """See base class."""
        return self.actor
