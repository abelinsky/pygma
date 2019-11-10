import pygma
from pygma.trainers import base_trainers as trainers
import gym


def main():
    env = gym.make('CartPole-v0')
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    pgtrainer = trainers.PolicyGradientTrainer(env, is_discrete=discrete)
    pgtrainer.train_agent(100)


if __name__ == "__main__":
    main()
