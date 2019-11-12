.. _vanilla_policy:

Vanilla Policy Gradient
========================

Vanilla Policy Gradient (VPG) Algorithm is the simplest form of the policy
gradient algorithms in reinforcement learning.

Idea
----

The main idea of the algorithm is quite simple - we directly take the gradient
of the objective function of the reinforcement learning (see below) and "climb"
over the gradient. We calculate this gradient through sampling *trajectories* -
sets of environment's states and actions, recommended by our *policy*.
Intuitively, we try to make "good" trajectories (with high *rewards*)
more likely and "bad" trajectories (with low *rewards*) less likely.
We try to increase the probabilities of choosing actions that lead
to higher rewards, and decrease the probabilities of other actions.
The whole math simply formalizes the notion of "trial and error".

Mathematics
-----------

Let denote:

-  :math:`s_{t}` - *state*, :math:`a_{t}` - *action*,
   :math:`r(s_t, a_t)` - *reward*
-  :math:`\pi_{\theta}` - a policy with parameters :math:`\theta`
   (probability distribution over the action space, conditioned
   on the state)
-  :math:`\tau` - a *trajectory* (or *rollout*) of length *T*
   (a sequence of *states*  and *actions*)
-  :math:`p_{\theta}(\tau)` - a joint probability distribution
   of the trajectory
-  :math:`p(s_1)` - probability distribution of the initial state


A joint probability distribution :math:`p_{\theta}(\tau)` of the
trajectory can be written as follows:

.. math::
    p_{\theta}(\tau) = p_{\theta}(s_1, a_1, ..., s_{T}, a_{T}) =
    p(s_1) \prod_{t=1}^{T} p(s_{t+1}|s_{t}, a_{t}) \pi_{\theta} (a_t|s_t)

and cumulative reward of the rollout

.. math::
    r(\tau) = \sum_{t=1}^{T} r(s_t, a_t).


Then the reinforcement learning objective is to learn the parameters
:math:`\theta` of the policy :math:`\pi_{\theta}(\tau)` that maximizes
the **expected reward** of the trajectory:

.. math::
    J(\theta) = \mathbb{E}_{\tau \sim p_\theta (\tau)} \big[ r(\tau) \big] = \int p_\theta (\tau) r(\tau) \mathrm{d} \tau

The policy gradient approach is to directly take the gradient of this objective
function w.r.t. :math:`\theta`:

.. math::

    \nabla_\theta J(\theta) &= \nabla_\theta \int p_\theta(\tau) r(\tau) \mathrm{d} \tau = \\
                            &= \int \nabla_\theta p_\theta(\tau) r(\tau) \mathrm{d} \tau = \\
                            &= \int p_\theta(\tau) \nabla_\theta \mathrm{log} \: p_\theta(\tau) r(\tau) \mathrm{d} \tau = \\
                            &= \mathbb{E}_{\tau \sim p_\theta (\tau)} \big[ \nabla_\theta \mathrm{log} \: p_\theta(\tau) r(\tau) \big]

.. note::
    In derivation we used a convinient identity:

    .. math::

        p_\theta(\tau) \nabla_\theta \mathrm{log} \: p_\theta(\tau) =
            p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = \nabla_\theta p_\theta(\tau)

:math:`p_{\theta}(\tau)` contains an unknown system's dynamics
:math:`p(s_{t+1}|s_{t}, a_{t})`. We can get rid of it by taking
the gradient of the logarithm :math:`\mathrm{log} \: p_\theta(\tau)`:

.. math::

    \nabla_\theta \mathrm{log} \: p_\theta(\tau) &= \nabla_\theta \mathrm{log} \big[ p(s_1) \prod_{t=1}^{T}p(s_{t+1}|s_{t}, a_{t}) \pi_{\theta} (a_t|s_t) \big] = \\
                                                 &= \nabla_\theta \Big[ \mathrm{log} \: p(s_1) + \sum_{t=1}^{T} \big[ \mathrm{log} \:
                                                   \pi_{\theta} (a_t|s_t) + \mathrm{log} \: p(s_{t+1} | s_t, a_t) \big] \Big] = \\
                                                 &= \nabla_\theta \big[ \sum_{t=1}^{T} \mathrm{log} \: \pi_\theta (a_t|s_t) \big]

So, the :math:`\nabla_\theta J(\theta)` can be rewritten as:

.. math::

    \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta (\tau)} \Big[ \nabla_\theta \mathrm{log} \: \pi_\theta (\tau) r(\tau) \Big]

In practice, this can be approximated through a batch of
:math:`N` trajectories:

.. math::

    \nabla_\theta J(\theta) &\approx \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \mathrm{log} \: \pi_\theta (\tau_i) r(\tau_i) = \\
                            &= \frac{1}{N} \sum_{i=1}^{N} \Big( \sum_{t=1}^{T} \nabla_\theta \mathrm{log} \: \pi_\theta (a_{it}|s_{it}) \Big)
                                \Big( \sum_{t=1}^{T} r(s_{it}, a_{it}) \Big)

Given :math:`\nabla_\theta J(\theta)` is calculated,
we can "climb" in the direction of the gradient to maximize
the objective function :math:`J(\theta)`:

.. math::
    \theta_{k+1} = \theta_{k} + \alpha \nabla_\theta J(\theta)

So, the simplest form of policy gradient algorithm, called REINFORCE
algorithm, consists of the following steps:

1.  Generate samples: run current policy :math:`\pi_\theta`
    and sample a set of trajectories :math:`{\tau^i}`
    (a sequences of :math:`s_{t}, a_{t}`)
2.  Estimate returns and compute the gradient of the objective function:

     .. math::

         \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \Big( \sum_{t=1}^{T} \nabla_\theta \mathrm{log}
         \: \pi_\theta (a_{it}|s_{it}) \Big)
         \Big( \sum_{t=1}^{T} r(s_{it}, a_{it}) \Big)

3.  Update the parameters of the policy:
    :math:`\theta_{k+1} = \theta_{k} + \alpha \nabla_\theta J(\theta)`

4.  Iterate through steps 1-3.

Features and obstacles
----------------------
1.  We can use policy gradient algoritm in partially observed Markov
    Decision Process without modification (Markov property isn't used
    in derivation)
2.  But... the gradient :math:`\nabla_\theta J(\theta)` has high variance!
    It's very, very noisy!
3.  It's an online (on-policy) algorithm, that means that we calculate
    the gradient :math:`\nabla_\theta J(\theta)`, change the parameters
    :math:`\theta` and then we have to generate new samples with the new
    policy :math:`\pi_\theta`. For example, if our police is a neural
    network, it changes only a little bit with each gradient step.
    That's why on-policy learning can be extremely inefficient.

4.  Practical considerations: batch size, learning rates, optimizers.


Pygma's example
---------------

.. code-block:: python

   import pygma


Suggested reading
-----------------

-  Williams (1992). Simple statistical gradient-following algorithms
   for connectionist reinforcement learning: introduces REINFORCE algorithm
-  Baxter & Bartlett (2001). Infinite-horizon policy-gradient estimation:
   temporally decomposed policy gradient (not the first paper on this!
   see actor-critic section later)
-  Peters & Schaal (2008). Reinforcement learning of motor skills with
   policy gradients: very accessible overview of optimal baselines and
   natural gradient

