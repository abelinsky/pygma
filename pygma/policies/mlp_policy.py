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
                 **kwargs):
        super(MLPPolicy, self).__init__(**kwargs)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.n_layers = n_layers
        self.layers_size = layers_size
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.activation = activation

        self._build_model()

    def _build_model(self):
        """Builds policy (multilayer neural network)."""
        self.model = keras.Sequential()
        for _ in range(self.n_layers):
            self.model.add(keras.layers.Dense(self.layers_size))
        self.model.add(keras.layers.Dense(self.action_dim))
        self.model.build((None, self.obs_dim))

    @tf.function
    def get_action(self, obs):
        """See base class."""
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        if self.discrete:
            logits = self.model(observation)
            sampled_action = tf.squeeze(
                tf.random.categorical(logits, 1))
        else:
            # TODO: implement continuos case
            raise NotImplementedError

        return sampled_action

    def get_log_prob(self, acs, obs):
        r"""Returns log probabilities of seen actions.

        Args:
            acs: Seen actions, numpy array
            obs: Observations in which actions were seen, numpy array

        Returns:
            log \pi (a_i|o_i), naumpy array
        """
        if self.discrete:
            logits = self.model(obs)
            return tfp.distributions.Categorical(logits=logits).log_prob(acs)
        else:
            # TODO: implement continuos case
            raise NotImplementedError

    """
        def define_forward_pass(self):
        if self.discrete:
            logits_na = build_mlp(self.observations_pl, output_size=self.ac_dim,
                                  scope='discrete_logits', n_layers=self.n_layers, size=self.size)
            self.parameters = logits_na
        else:
            mean = build_mlp(self.observations_pl, output_size=self.ac_dim,
                             scope='continuous_logits', n_layers=self.n_layers, size=self.size)
            logstd = tf.Variable(tf.zeros(self.ac_dim), name='logstd')
            self.parameters = (mean, logstd)
    """
    """
        if self.discrete:
            logits_na = self.parameters
            self.sample_ac = tf.squeeze(
                tf.multinomial(logits_na, num_samples=1), axis=1)
        else:
            mean, logstd = self.parameters
            self.sample_ac = mean + \
                tf.exp(logstd) * tf.random_normal(tf.shape(mean), 0, 1)
    """

    def save(self, filename):
        """See base class."""
        raise NotImplementedError

    def restore(self, filename):
        """See base class."""
        raise NotImplementedError
