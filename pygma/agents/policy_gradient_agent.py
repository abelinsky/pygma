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
import numpy as np
import tensorflow as tf


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
      gamma: Discount factor, float
      reward_to_go: Indicates whether to apply reward to go, bool
      baseline: Indicates whether to use baseline for gradient estimation, bool
      standardize_advantages: Indicates whether to normalize the resulting advantages, bool
    """

    def __init__(self, env, action_dim, obs_dim, n_layers, layers_size, is_discrete, learning_rate,
                 activation, gamma, reward_to_go, baseline, standardize_advantages):
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
        self.reward_to_go = reward_to_go
        self.gamma = gamma
        self.baseline = baseline
        self.standardize_advantages = standardize_advantages
        self.learning_rate = learning_rate

    @property
    def policy(self):
        """See base class."""
        return self.actor

    def train(self, obs, acs, rews, num_steps):
        """See base class.

        Training a PolicyGradient agent means updating its policy (actor) with thr given observations/actions
        and the calculated qvals/advantages that come from the seen rewards.

        The expression for the policy gradient is

                PG = E_{tau} [sum_{t=0}^{T-1} grad log pi(a_t|s_t) * (Q_t - b_t )]

            where 
                tau=(s_0, a_0, s_1, a_1, s_2, a_2, ...) is a trajectory,
                Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                b_t is a baseline which may depend on s_t,
                and (Q_t - b_t ) is the advantage.

            Policy gradient update needs (s_t, a_t, q_t, adv_t),
                and that is exactly what this function provides.
        """

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        for step in range(num_steps):
            with tf.GradientTape() as tape:
                # Compute the loss value .
                logprob = self.policy.get_log_prob(acs, obs)
                q_values = self.calculate_q_values(rews)
                adv = self.calculate_advantages(q_values)
                # minus because we wan to maximize cumulative reward
                loss = tf.math.reduce_sum(-tf.multiply(logprob, adv))

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(
                loss, self.policy.model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(
                zip(grads, self.policy.model.trainable_weights))

    def calculate_q_values(self, rews_list):
        """ Estimates the Q-function with Monte Carlo. 

        Args:
            rews_list: Rewards array, length equals to number 
            of rollouts, numpy array. Each element of the array 
            contains list of rewards of a particular rollout.

        Returns:
            Q-values estimates: Array of Q-values estimations, length 
            equals to number of steps across all rollouts. Each element
            corresponds to Q-value of the particular observation/action
            at time step t.
        """

        if not self.reward_to_go:
            # q(s_t, a_t) = \sum_{t=0}^{T-1} gamma^t r_t
            q_values = np.concatenate(
                [self._discounted_rewards(r) for r in rews_list])

        else:
            # TODO: Implement reward-to-go
            raise NotImplementedError

        return q_values

    def calculate_advantages(self, q_values):
        if self.baseline:
            # TODO: Implement baseline
            raise NotImplementedError
        else:
            # just copy q_values
            adv = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

        return adv

    def _discounted_rewards(self, rews):
        """Discounts rewards

        Args:
          rews: a list of rewards for a particular rollout, numpy array.

        Returns:
          a list where each entry corresponds to sum_{t=0}^{T-1} gamma^t r_{t}.

          .. note::
            All entries in return are equivalent, because function doesn't involve `reward-to-go`.
        """
        T = len(rews)
        discounts = self.gamma ** np.arange(T)
        disc_rews = rews * discounts
        disc_rews_sum = np.sum(disc_rews)
        return [disc_rews_sum] * T

    def update_policy(self, obs, next_obs, acs, rews, terminals):
        """See base class."""
        pass
