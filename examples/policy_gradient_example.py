import pygma
from pygma.trainers import base_trainers as trainers
import gym


def main():
    env = gym.make('CartPole-v0')
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    max_rollout_length = env.spec.max_episode_steps

    pgtrainer = trainers.PolicyGradientTrainer(env, is_discrete=discrete,
                                               max_rollout_length=max_rollout_length)
    pgtrainer.train_agent(1000)

    # tf.config.experimental_run_functions_eagerly(True)
    # tf.config.experimental_run_functions_eagerly(False)


if __name__ == "__main__":
    main()
