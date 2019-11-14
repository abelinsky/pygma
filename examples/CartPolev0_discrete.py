import os
import time
import pygma
from pygma.rl.reinforce import agent
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

    # create env
    env = gym.make(env_name)

    # create agent
    agent_ = agent.ReinforceAgent(
        env,
        reward_to_go=True,
        baseline=True,
        n_layers=2,
        layers_size=32,
        min_batch_size=5000,
        log_metrics=True,
        logdir=logdir,
        render_freq=10)

    # train agent
    agent_.run_training_loop(1000)


if __name__ == "__main__":
    main()
