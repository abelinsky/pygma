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
"""Classes to represent types of MLPPolicy."""
from pygma.policies.base_policy import BasePolicy
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


class MLPPolicy(BasePolicy):
    """Multilayer neural network policy.

    Attributes:
      action_dim: action space dimension, int
      obs_dim: observation space dimension, int
      n_layers: number of layers in neural network, int
      layers_size: size of hidden layers, int
      discrete: is discrete environment, bool
      learning_rate: learning rate  
      activation: activation function of hidden layers, default 'relu'
    """

    def __init__(self,
                 action_dim,
                 obs_dim,
                 n_layers,
                 layers_size,
                 discrete,
                 learning_rate=1e-4,
                 activation='relu',
                 baseline=False,
                 **kwargs):
        super(MLPPolicy, self).__init__(**kwargs)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.n_layers = n_layers
        self.layers_size = layers_size
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.activation = activation
        self.baseline = baseline

        self._build_model()

        if self.baseline:
            self._build_baseline_model()

    def _build_model(self):
        """Builds policy (multilayer neural network).

        If the environment is discrete, the outputs from neural network
        refer to logits of categorical distribution from which one can
        sample discrete actions. If it is continuous, the outputs refer 
        to the mean of normal distribution and another variable for 
        standard deviation is involved.    
        """
        self.model = keras.Sequential()
        for _ in range(self.n_layers):
            self.model.add(keras.layers.Dense(self.layers_size))
        self.model.add(keras.layers.Dense(self.action_dim))
        self.model.build((None, self.obs_dim))

        # If environment is continuous then create trainable variable for
        # standard deviation
        if not self.discrete:
            self.logstd = tf.Variable(tf.zeros(self.action_dim), name='logstd')

    def _build_baseline_model(self):
        """Builds baseline neural network."""
        self._baseline_model = keras.Sequential()
        for _ in range(self.n_layers):
            self._baseline_model.add(keras.layers.Dense(self.layers_size))
        # One output for baseline prediction
        self._baseline_model.add(keras.layers.Dense(1))
        self._baseline_model.build((None, self.obs_dim))

    @tf.function
    def get_action(self, obs):
        """See base class."""
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        if self.discrete:
            logits = self.model(observation)
            # Sample action from categorical distribution
            # where logits are outputs from neural network
            sampled_action = tf.squeeze(
                tf.random.categorical(logits, 1))
        else:
            mean = self.model(observation)
            logstd = self.logstd
            sampled_action = tf.squeeze(
                mean + tf.exp(logstd) * tf.random.normal(tf.shape(mean), 0, 1))

        return sampled_action

    def get_baseline_prediction(self, obs):
        r"""Returns baseline neural network prediction for specified observation.

        This is a *state-dependent* baseline - a sort of *value function* that can be
        trained to approximate the sum of future rewards starting from a
        particular state:

            .. math::

                V_\phi^\pi(s_t) = \sum_{t'=t}^{T} \mathbb{E}_{\pi_\theta} \big[ r(s_{t'}, a_{t'}) | s_t \big]

        Args:
            obs: Observation, numpy array

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

    def get_log_prob(self, acs, obs):
        r"""Returns log probabilities of seen actions.

        Args:
            acs: Seen actions (numpy array).
            obs: Observations in which actions were seen (numpy array).

        Returns:
            :math:`\mathrm{log} \: \pi (a_i|o_i)`, log probabilities (numpy array).
        """
        if self.discrete:
            # log probability under categorical distribution
            logits = self.model(obs)
            logprob = tfp.distributions.Categorical(
                logits=logits).log_prob(acs)
        else:
            # log probability under a multivariate gaussian
            mean = self.model(obs)
            logstd = self.logstd
            logprob = tfp.distributions.MultivariateNormalDiag(
                loc=mean, scale_diag=tf.exp(logstd)).log_prob(acs)

        return logprob

    def save(self, filename):
        """See base class."""
        raise NotImplementedError

    def restore(self, filename):
        """See base class."""
        raise NotImplementedError
