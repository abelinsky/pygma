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
"""Contains the base Policies, from which all policies inherit."""

import abc
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


class BasePolicy(abc.ABC):
    """Base class for agent's *policy* concept.

    Policy is responsible for producing actions
    from seen observations. 

    This is a class from which all policies inherit. 

    Note: this is the parent class of all policies, 
    not an actual policy that can be trained.

    A BasePolicy is a class implementing common operations of trainable 
    policies.

    Users will instantiate a BasePolicy to just add some nuances of 
    particular policies in particular environments.

    Descendants of ``BasePolicy`` must implement the following abstract methods and properties:

    * ``get_action()``: Return an action for particular observation.
    * ``trainable_params``: Returns trainable params of the policy to use while training.

    """

    def __init__(self):
        super(BasePolicy, self).__init__()

    @property
    @abc.abstractmethod
    def trainable_params(self):
        """Returns trainable params of the policy."""
        pass

    @abc.abstractmethod
    def get_action(self, obs):
        """Returns action for specific observation.

        Args:
          obs: observation of the environment.

        Returns:
          numpy.array: An action which is recommended by policy.    
        """
        pass

    def save(self, filename):
        """Saves policy to disc.

        Args:
          filename: File to save policy to.
        """
        pass

    def restore(self, filename):
        """Restores policy from file.

        Args:
            filename: File wit saved policy.
        """
        pass


class MLPPolicy(BasePolicy):
    """Multilayer neural network policy.

    Args:
      action_dim (int): Action space dimension.
      obs_dim (int): observation space dimension.
      n_layers (int): number of layers in neural network.
      layers_size (int): size of hidden layers.
      is_discrete (bool): indicates whether environment is discrete.
      activation_function (str): activation function of hidden layers.
    """

    def __init__(self,
                 action_dim,
                 obs_dim,
                 n_layers,
                 layers_size,
                 is_discrete,
                 activation_function):

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        self.n_layers = n_layers
        self.layers_size = layers_size
        self.is_discrete = is_discrete
        self.activation_function = activation_function

        self._build_model()

    @property
    def trainable_params(self):
        """See base class."""
        params = self._model.trainable_weights
        if not self.is_discrete:
            params += [self.logstd]
        return params

    def _build_model(self):
        """Builds policy (multilayer neural network).

        If the environment is discrete, the outputs from neural network
        refer to logits of categorical distribution from which one can
        sample discrete actions. If it is continuous, the outputs refer 
        to the mean of normal distribution and another variable for 
        standard deviation is involved.    
        """
        self._model = keras.Sequential()
        for _ in range(self.n_layers):
            self._model.add(keras.layers.Dense(
                self.layers_size, activation=self.activation_function, dtype='float32'))
        self._model.add(keras.layers.Dense(self.action_dim))
        self._model.build((None, self.obs_dim))

        # If environment is continuous then create trainable variable for
        # standard deviation
        if not self.is_discrete:
            self.logstd = tf.Variable(tf.zeros(self.action_dim), name='logstd')

    @tf.function
    def get_action(self, obs):
        """See base class."""
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        if self.is_discrete:
            logits = self._model(observation)
            # Sample action from categorical distribution
            # where logits are outputs from neural network
            sampled_action = tf.random.categorical(logits, 1)
        else:
            mean = self._model(observation)
            logstd = self.logstd
            sigma = tf.exp(logstd)
            # Sample normally distributed action
            # where logits (mean) are outputs from neural network
            # and sigma is trainable variable
            sampled_action = mean + sigma * \
                tf.random.normal(tf.shape(mean), 0, 1)

        return sampled_action
        # return tf.squeeze(sampled_action)

    def get_log_prob(self, acs, obs):
        r"""Returns log probabilities of seen actions.

        Args:
            acs (numpy.array): Seen actions.
            obs (numpy.array): Observations in which actions were seen.

        Returns:
            numpy.array: Log probabilities :math:`\mathrm{log} \: \pi (a_i|o_i)`.
        """
        if self.is_discrete:
            # log probability under categorical distribution
            logits = self._model(obs)
            logprob = tfp.distributions.Categorical(
                logits=logits).log_prob(acs)
        else:
            mean = self._model(obs)
            logprob = tfp.distributions.MultivariateNormalDiag(
                loc=mean, scale_diag=tf.exp(self.logstd)).log_prob(acs)

            # mean = self._model(obs)
            # if self.action_dim > 1:
            #     # log probability under a multivariate gaussian
            #     logprob = tfp.distributions.MultivariateNormalDiag(
            #         loc=mean, scale_diag=tf.exp(self.logstd)).log_prob(acs)
            # elif self.action_dim == 1:
            #     # log probability under a noraml distribution
            #     logprob = tfp.distributions.Normal(
            #         loc=mean, scale=tf.exp(self.logstd)).log_prob(acs)
            # else:
            #     raise ValueError("Action dimension is invalid")

        return logprob
