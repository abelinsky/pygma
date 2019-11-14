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
import numpy as np
import tensorflow as tf
from tensorflow import keras

from pygma.rl.reinforce import policy
from pygma.rl import base_agent


class ReinforceAgent(base_agent.BaseAgent):
    """Policy gradient algorithm implementation.

    Args:
      env: Environment to learn from.
      actor_n_layers (int): Number of hidden layers in actor policy network.
      actor_layers_size (int): Size of hidden layers in actor policy network.
      activation_function (str): Activation function in hidden layers of actor 
        policy network.
      discount (float): Discount factor.
      learning_rate (float): Learning rate.
      reward_to_go (bool): Indicates whether to apply reward to go.
      standardize_advantages (bool): Indicates whether to normalize the resulting advantages.
      baseline (bool): Indicates whether to use baseline for gradient estimation.
      baseline_n_layers (int): Number of hidden layers in value policy network.
      baseline_layers_size (int): Size of hidden layers in value policy network.
      baseline_nn_activation (int): Activation function in hidden layers of value 
        policy network.
    """

    def __init__(self,
                 env,
                 # MLPPolicy parameters
                 actor_n_layers=2,
                 actor_layers_size=64,
                 activation_function='tanh',
                 discount=1.0,
                 # PolicyGradient algorithm parameters
                 learning_rate=5e-3,
                 reward_to_go=True,
                 standardize_advantages=True,
                 baseline=True,
                 baseline_n_layers=2,
                 baseline_layers_size=64,
                 baseline_nn_activation='tanh',
                 # BaseAgent kwargs
                 **kwargs):

        super(ReinforceAgent, self).__init__(env, **kwargs)

        self.learning_rate = learning_rate
        self.standardize_advantages = standardize_advantages
        self.baseline = baseline
        self.discount = discount
        self.reward_to_go = reward_to_go

        self.baseline_n_layers = baseline_n_layers
        self.baseline_layers_size = baseline_layers_size
        self.baseline_nn_activation = baseline_nn_activation

        # create policy
        self._actor = policy.ReinforcePolicy(action_dim=self.ac_dim,
                                             obs_dim=self.obs_dim,
                                             n_layers=actor_n_layers,
                                             layers_size=actor_layers_size,
                                             is_discrete=self.is_discrete_env,
                                             activation_function=activation_function)

        if self.baseline:
            self._build_baseline_model()

    @property
    def policy(self):
        """See base class."""
        return self._actor

    def _build_baseline_model(self):
        """Builds baseline neural network."""
        self._baseline_model = keras.Sequential()
        for _ in range(self.baseline_n_layers):
            self._baseline_model.add(
                keras.layers.Dense(self.baseline_layers_size, activation=self.baseline_nn_activation))
        # One output for baseline prediction
        self._baseline_model.add(keras.layers.Dense(1))
        self._baseline_model.build((None, self.obs_dim))

    def get_baseline_prediction(self, obs):
        r"""Returns baseline neural network prediction for specified observation.

        This is a *state-dependent* baseline - a sort of *value function* that can be
        trained to approximate the sum of future rewards starting from a
        particular state:

            .. math::

                V_\phi^\pi(s_t) = \sum_{t'=t}^{T} \mathbb{E}_{\pi_\theta} \big[ r(s_{t'}, a_{t'}) | s_t \big]

        Args:
            obs (numpy.array): Observation.

        Raises:
            AttributeError: If `baseline` flag was not set for this policy.

        Returns:
            Baseline prediction, tensor with float value.
        """
        if not self.baseline:
            raise AttributeError(
                "Get baseline prediction failed because baseline flag was not set for this policy.")
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        return tf.squeeze(
            self._baseline_model(observation))

    def _train(self, obs, acs, rews, num_steps):
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
                adv = tf.convert_to_tensor(adv, dtype=tf.float32)

                # minus because we wan to maximize cumulative reward

                # pg_loss = -tf.math.reduce_sum(tf.multiply(logprob, adv))
                pg_loss = -tf.math.reduce_sum(logprob * tf.stop_gradient(adv))

                if self.baseline:
                    # Compute the loss value for baseline prediction
                    baseline_targets = (
                        q_values - np.mean(q_values))/(np.std(q_values)+1e-8)
                    baseline_predictions = self.get_baseline_prediction(obs)
                    baseline_loss = tf.keras.losses.MSE(
                        baseline_targets, baseline_predictions)

            # 1. Upgrade model
            pg_grads = tape.gradient(pg_loss, self.policy.trainable_params)
            optimizer.apply_gradients(
                zip(pg_grads, self.policy.trainable_params))

            if self.baseline:
                # 2. Update baselines
                baseline_grads = tape.gradient(
                    baseline_loss, self._baseline_model.trainable_weights)
                optimizer.apply_gradients(
                    zip(baseline_grads, self._baseline_model.trainable_weights))

    def calculate_q_values(self, rews_list):
        """ Estimates the Q-function with Monte Carlo. 

        Args:
          rews_list (numpy.array): Rewards array, length equals to number 
          of rollouts. Each element of the array contains list of rewards 
          of a particular rollout.

        Returns:
          Q-values estimates (numpy.array): Array of Q-values estimations, length 
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
          q_values (numpy.array): Q-values estimates, length equals to number of steps across all rollouts.
          obs (numpy.array): Observations

        Returns:
            Advantages, a numpy array,  length equals to number of steps across all rollouts.
        """
        if self.baseline:
            # pass observations into baselint neural network  and get
            # state (observation) - dependent baseline predictions
            # (an estimation of future rewards from this state)
            baselines_unnormalized = self.get_baseline_prediction(obs)
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
          rews (numpy.array): A list of rewards for a particular rollout.

        Returns:
          a list where each entry corresponds to sum_{t=0}^{T-1} discount^t r_{t}.

          Note:
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
          rews (numpy.array): A list of rewards for a single rollout, array of length T.

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
