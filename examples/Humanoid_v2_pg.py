import os
import time
import pygma
import gym
import tensorflow as tf
from pygma.rl.reinforce import agent as reinforce_agent


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

    # create env
    env = gym.make(env_name)

    # create agent
    agent_ = reinforce_agent.ReinforceAgent(
        env,
        max_rollout_length=1000,
        discount=0.99,
        min_batch_size=50000,
        learning_rate=0.01,
        reward_to_go=True,
        baseline=True,
        log_metrics=True,
        logdir=logdir,
        render_freq=100,
        log_freq=1)

    # train agent
    agent_.run_training_loop(1000)


if __name__ == "__main__":
    main()
