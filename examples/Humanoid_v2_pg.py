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

    env_name = 'Humanoid-v2'

    # python run_hw2_policy_gradient.py --env_name Humanoid-v2 --ep_len 1000 --discount 0.99 -n 1000
    # -l 2 -s 64 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name hu_b40000_r0.005

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

    # create trainer and train
    n_iter = 1000
    pgtrainer = trainers.PolicyGradientTrainer(env,
                                               is_discrete=discrete,
                                               max_rollout_length=1000,
                                               logdir=logdir,
                                               reward_to_go=True,
                                               baseline=True,
                                               discount=0.99,
                                               min_batch_size=10000,
                                               learning_rate=0.005,
                                               render=True,
                                               render_freq=n_iter/10,
                                               log_freq=1)
    pgtrainer.train_agent(n_iter)


if __name__ == "__main__":
    main()
