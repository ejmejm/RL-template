import cv2
import gym
import numpy as np

def init_logger(lp):
    """
    Call before using the log function
    to initialize the log file.
    """
    global log_path
    log_path = lp

    f = open(log_path, 'w+')
    f.close()

def log(string):
    """
    Log something to the log file created
    by `init_logger`.
    """
    if type(string) != str:
        string = str(string)

    with open(log_path, 'a') as f:
        f.write(string + '\n')

def filter_obs(obs, obs_shape=(42, 42)):
    """
    Used for formatting 3D observations (3rd axis because
    of color channel in this case). Filters an obs to the
    right size, color, and scale. Only works for 1 obs at a time.
    """
    assert(type(obs) == np.ndarray), "The observation must be a numpy array!"
    assert(len(obs.shape) == 3), "The observation must be a 3D array!"

    obs = cv2.resize(obs, obs_shape, interpolation=cv2.INTER_LINEAR)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs / 255.
    
    return obs[:, :, np.newaxis]

def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given
    rewards and gamma factor. While this is not used
    in this implementation, it can be used as an
    alternative to GAEs.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=0.99, decay=0.95):
    """
    Return the General Advantage Estimates from the 
    given reward and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])