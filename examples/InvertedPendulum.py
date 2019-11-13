import os
import time
import pygma
from pygma.trainers import base_trainers as trainers
import gym
import tensorflow as tf

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ab/.mujoco/mjpro150/bin
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so


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

    env_name = 'InvertedPendulum-v2'

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
                                               max_rollout_length=1000,
                                               logdir=logdir,
                                               reward_to_go=True,
                                               discount=0.9,
                                               min_batch_size=2500,
                                               learning_rate=0.005,
                                               n_layers=2,
                                               layers_size=64,
                                               render=True)
    pgtrainer.train_agent(1000)


if __name__ == "__main__":
    main()
