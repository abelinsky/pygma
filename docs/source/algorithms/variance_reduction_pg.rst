.. _variance_reduction_pg:

Policy Gradient Variance Reduction
==================================

:ref:`vanilla_policy` has high variance. Variance Reduction
techniques are intended to solve this problem.

Idea
----

There are several ways to reduce the variance of the policy gradient:

-  to exploit causality: the notion that the policy at time :math:`t'`
   cannot affect rewards in the past (at time :math:`t` when :math:`t < t'`)
   which is also known as "reward-to-go"
-  to apply discounting: multiplying the rewards with a discount factor
   :math:`\gamma`
-  to use baselines - substract a constant from the sum of the reward.

Mathematics
-----------

Reward-to-go
~~~~~~~~~~~~
Applying causality gives a slightly modified objective finction where the
sum of rewards does not include rewards achieved prior to the time step
at which the policy is beeing queried.

.. math::
    \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \mathrm{log} \: \pi_\theta (a_{it}|s_{it})
                                \Big( \sum_{t'=t}^{T} r(s_{it'}, a_{it'}) \Big).

Discounting
~~~~~~~~~~~

Multiplying the rewards with a discount factor :math:`\gamma` can be
thought as encouraging the agent to focus more on the rewards that
are closer in time and less on the rewards that are further in the
future. This also helps to reduce the variance because of the
numerical trick: values of the sum of rewards become smaller.
Applying the discounts on the rewards of the full trajectory
gives an equation:

.. math::

    \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N}
                                \Big( \sum_{t=1}^{T} \nabla_\theta \mathrm{log} \: \pi_\theta (a_{it}|s_{it}) \Big)
                                \Big( \sum_{t'=1}^{T} \gamma^{t'-1} r(s_{it'}, a_{it'}) \Big)


Applying the discounts on the "reward-to-go" gives an equation:

.. math::

    \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \mathrm{log} \: \pi_\theta (a_{it}|s_{it})
                                \Big( \sum_{t'=t}^{T} \gamma^{t'-t} r(s_{it'}, a_{it'}) \Big).


Baseline
~~~~~~~~

Substracting a baseline (some constant *b*) from the sum of the
rewards is intended to reduce the variance:

.. math::

    \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta (\tau)} \Big[ \nabla_\theta \mathrm{log} \: \pi_\theta (\tau) \big[ r(\tau) - b \big] \Big]

.. note::
    Substracting a baseline leaves the policy gradient unbiased in expectation.

There are several choises to select *b*. For example average reward
:math:`b=\frac{1}{N} \sum_{i=1}^{N} r(\tau)` is simple and pretty good.
One can also apply more sophisticated baseline. For example, one can use
a *state-dependent* baseline - a sort of *value function* that can be
trained to approximate the sum of future rewards starting from a
particular state:

.. math::

    V_\phi^\pi(s_t) = \sum_{t'=t}^{T} \mathbb{E}_{\pi_\theta} \big[ r(s_{t'}, a_{t'}) | s_t \big]

and in this case the appriximate policy gradient looks like this:

.. math::

    \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \mathrm{log} \: \pi_\theta (a_{it}|s_{it})
                                \bigg( \Big( \sum_{t'=t}^{T} \gamma^{t'-t} r(s_{it'}, a_{it'}) \Big) - V_\phi^\pi(s_{it}) \bigg).

Features and obstacles
----------------------


Pygma's classes
---------------


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
