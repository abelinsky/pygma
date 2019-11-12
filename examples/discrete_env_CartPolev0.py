import os
import time
import pygma
from pygma.trainers import base_trainers as trainers
import gym


def main():
    env_name = 'CartPole-v0'

    # create data dir
    data_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '../data')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # create log dir
    log_prefix = 'pg_'
    logdir = log_prefix + env_name + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = os.path.join(data_path, logdir)

    # set up env
    env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    max_rollout_length = env.spec.max_episode_steps

    # create trainer and train
    pgtrainer = trainers.PolicyGradientTrainer(env,
                                               is_discrete=discrete,
                                               max_rollout_length=max_rollout_length,
                                               logdir=logdir,
                                               reward_to_go=True,
                                               baseline=True)
    pgtrainer.train_agent(1000)


if __name__ == "__main__":
    main()
