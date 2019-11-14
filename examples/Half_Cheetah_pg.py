import os
import time
import pygma
import gym
import tensorflow as tf
from pygma.rl.reinforce import agent


def main():
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    env_name = 'HalfCheetah-v2'

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
    agent_ = agent.ReinforceAgent(env,
                                  max_rollout_length=150,
                                  discount=0.95,
                                  actor_layers_size=32,
                                  baseline_layers_size=32,
                                  min_batch_size=50000,
                                  learning_rate=0.02,
                                  log_metrics=True,
                                  logdir=logdir,
                                  render_freq=100)

    # train agent
    agent_.run_training_loop(1000)


if __name__ == "__main__":
    main()