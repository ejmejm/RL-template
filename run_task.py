import numpy as np
import gym
import time
from mpi4py import MPI
import heapq
import tensorflow as tf
from model import Model
from utils import *
import multiprocessing
import pickle


def worker(model, max_steps=1000):
    """
    Performs the game simulation, and is called across all processes.
    Returns a list of training data in the shape (n_steps, 4), and
    and the total reward of the rollout.
    """
    train_data = []
    env = make_env()
    obs = env.reset()
    # obs = filter_obs(obs)

    ep_reward = 0
    for _ in range(max_steps):
        act, val = model.gen_actions_and_values([obs])
        act, val = act[0], val[0]

        next_obs, rew, d, _ = env.step(act)
        # next_obs = filter_obs(next_obs)
        train_data.append([obs, act, rew, val, next_obs])
        obs = next_obs
        ep_reward += rew
        if d:
            break

    train_data = np.asarray(train_data)

    ep_reward = np.sum(train_data[:, 2])
    # Calculate GAEs and replace values with the new values.
    train_data[:, 3] = calculate_gaes(train_data[:, 2], train_data[:, 3])

    return train_data, ep_reward


if __name__ == '__main__':
    ### Setp for MPI ###
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_processes = comm.Get_size()
    controller = 0

    ### Define starting parameters ###
    n_epochs = 1000
    n_train_batches = 32
    n_process_batches = int(n_train_batches / n_processes)
    log_freq = 5
    init_logger('training.log')

    ### Enable gpu usage for just the main process for training ###
    if rank == controller:
        device_config = tf.ConfigProto()
    else:
        device_config = tf.ConfigProto(device_count={'GPU': 0})

    model = Model(comm, controller, rank, n_acts=2,
                  obs_shape=(4,), sess_config=device_config)

    all_rewards = []
    for epoch in range(1, n_epochs+1):
        batch_data = []
        train_data = []
        for _ in range(n_process_batches):
            ### Simulate more episodes to gain training data ###
            if rank == controller:
                batch_data = comm.gather(worker(model), controller)

                batch_train_data = [datum[0] for datum in batch_data]
                batch_reward_data = [datum[1] for datum in batch_data]

                train_data.extend(batch_train_data)
                all_rewards.extend(batch_reward_data)
            else:
                comm.gather(worker(model), controller)

        if rank == controller:
            ### Log and print reward ###
            if epoch % log_freq == 0:
                print(
                    f'Epoch: {epoch}, Avg Reward: {np.mean(all_rewards[-n_train_batches:])}')
                log(
                    f'Epoch: {epoch}, Avg Reward: {np.mean(all_rewards[-n_train_batches:])}')

            ### Format training data ###
            train_data = np.concatenate(train_data)
            np.random.shuffle(train_data)

            obs_train_data = np.vstack(train_data[:, 0])
            action_train_data = train_data[:, 1]
            reward_train_data = train_data[:, 2]
            gae_train_data = train_data[:, 3]

            ### Train model and sync weights with all processes ###
            model.train_policy(
                obs_train_data, action_train_data, reward_train_data, gae_train_data)
            model.sync_weights()
        else:
            model.sync_weights()
