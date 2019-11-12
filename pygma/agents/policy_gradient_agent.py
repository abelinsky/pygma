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
      discount: Discount factor, float
      reward_to_go: Indicates whether to apply reward to go, bool
      baseline: Indicates whether to use baseline for gradient estimation, bool
      standardize_advantages: Indicates whether to normalize the resulting advantages, bool
    """

    def __init__(self, env, action_dim, obs_dim, n_layers, layers_size, is_discrete, learning_rate,
                 activation, discount, reward_to_go, baseline, standardize_advantages):
        super(PolicyGradientAgent, self).__init__()
        self.actor = MLPPolicy(
            action_dim=action_dim,
            obs_dim=obs_dim,
            n_layers=n_layers,
            layers_size=layers_size,
            discrete=is_discrete,
            learning_rate=learning_rate,
            activation=activation,
            baseline=baseline
        )
        self.reward_to_go = reward_to_go
        self.discount = discount
        self.baseline = baseline
        self.standardize_advantages = standardize_advantages
        self.learning_rate = learning_rate

    @property
    def policy(self):
        """See base class."""
        return self.actor

    def train(self, obs, acs, rews, num_steps):
        r"""See base class.

        Training a PolicyGradient agent means updating its policy (actor) with the given observations/actions
        and the calculated qvals/advantages that come from the seen rewards.

        The expression for the policy gradient is

            .. math::
                \nabla J_\theta = E_{\tau} \Big[\sum_{t=0}^{T-1} \nabla \mathrm{log} \: \pi_\theta(a_t|s_t) * (Q_t - b_t ) \Big]

        where: 
          -  :math:`\tau=(s_0, a_0, s_1, a_1, s_2, a_2, ...)` is a trajectory,
          -  :math:`Q_t` is the *Q*-value at time *t*, :math:`Q^{\pi}(s_t, a_t)`,
          -  :math:`b_t` is a baseline which may depend on :math:`s_t`,
          -  :math:`(Q_t - b_t)` is the advantage.

        Policy gradient update needs :math:`(s_t, a_t, q_t, adv_t)`,
        and that is exactly what this function provides.
        """

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        for _ in range(num_steps):
            with tf.GradientTape(persistent=True) as tape:
                # Compute the loss value for policy gradient
                logprob = self.policy.get_log_prob(acs, obs)
                q_values = self.calculate_q_values(rews)
                adv = self.estimate_advantages(q_values, obs)
                # minus because we wan to maximize cumulative reward
                pg_loss = -tf.math.reduce_sum(tf.multiply(logprob, adv))

                # Compute the loss value for baseline prediction
                baseline_targets = (
                    q_values - np.mean(q_values))/(np.std(q_values)+1e-8)
                baseline_predictions = self.policy.get_baseline_prediction(obs)
                baseline_loss = tf.keras.losses.MSE(
                    baseline_targets, baseline_predictions)

            # 1. Upgrade model
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            pg_grads = tape.gradient(
                pg_loss, self.policy.model.trainable_weights)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(
                zip(pg_grads, self.policy.model.trainable_weights))

            # 2. If countinuous, upgrade std
            if not self.policy.discrete:
                stdgrad = tape.gradient(pg_loss, [self.policy.logstd])
                optimizer.apply_gradients(
                    zip(stdgrad, [self.policy.logstd]))

            # 3. Update baselines
            baseline_grads = tape.gradient(
                baseline_loss, self.policy._baseline_model.trainable_weights)
            optimizer.apply_gradients(
                zip(baseline_grads, self.policy._baseline_model.trainable_weights))

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
            # q(s_t, a_t) = \sum_{t=0}^{T-1} discount^t r_t
            q_values = np.concatenate(
                [self._discounted_rewards(r) for r in rews_list])
        else:
            # q(s_t, a_t) = sum_{t'=t}^{T-1} discount^(t'-t) * r_{t'}
            q_values = np.concatenate(
                [self._discounted_reward_to_go(r) for r in rews_list])

        return q_values

    def estimate_advantages(self, q_values, obs):
        """Estimates advantages by substracting a baseline (if possible) from the sum of the rewards.

        This can be thought as selecting actions that are in some sense better than the mean 
        action in that state.  

        Args:
          q_values: Q-values estimates, length 
            equals to number of steps across all rollouts
          obs: Observations

        Returns:
            Advantages, a numpy array,  length 
            equals to number of steps across all rollouts
        """
        if self.baseline:
            # pass observations into baselint neural network  and get
            # state (observation) - dependent baseline predictions
            # (an estimation of future rewards from this state)
            baselines_unnormalized = self.policy.get_baseline_prediction(obs)
            b_n = baselines_unnormalized * np.std(q_values) + np.mean(q_values)
            adv = q_values - b_n
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
          a list where each entry corresponds to sum_{t=0}^{T-1} discount^t r_{t}.

          .. note::
            All entries in return are equivalent, because function doesn't involve `reward-to-go`.
        """
        T = len(rews)
        discounts = self.discount ** np.arange(T)
        disc_rews = rews * discounts
        disc_rews_sum = np.sum(disc_rews)
        return [disc_rews_sum] * T

    def _discounted_reward_to_go(self, rews):
        """Computes discounted reward to go value

        Args:
          rews: a list of rewards for a single rollout, numpy array of length T.

        Returns:
          a list of length t where the entry in inde t corresponds to q(s_t, a_t) = sum_{t'=t}^{T-1} discount^(t'-t) * r_{t'}.
        """
        T = len(rews)

        all_discounted_cumsums = []

        T = len(rews)
        for start_time_index in range(T):
            indices = np.arange(start_time_index, T)
            discounts = self.discount ** (indices - start_time_index)
            discounted_rtg = discounts * rews[start_time_index:]
            sum_discounted_rtg = np.sum(discounted_rtg)
            all_discounted_cumsums.append(sum_discounted_rtg)

        list_of_discounted_cumsums = np.array(all_discounted_cumsums)
        return list_of_discounted_cumsums

    def update_policy(self, obs, next_obs, acs, rews, terminals):
        """See base class."""
        pass
