import gym


#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math as m
import numpy as np
import baselines

from baselines import logger
from baselines import deepq

global output_dir

best_mean_reward, n_steps = -np.inf, 0
def callback(lcl, glb):
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 100

    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
            mean_reward = np.mean(lcl['episode_rewards'][-100:])
            print(lcl['t'], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                glb['save_variables'](os.path.join(output_dir,'best_model.pkl'))
    n_steps += 1

    if is_solved:
        print("is solved!")
    return is_solved


def train_DQN( env, out_dir, seed=None, **kwargs):

    # Logs will be saved in log_dir/monitor.csv
    global output_dir
    output_dir = out_dir
    log_dir = os.path.join(out_dir,'log')
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir, format_strs=['stdout','log','csv','tensorboard'])

    policy = kwargs['policy']
    n_timesteps = kwargs['total_timesteps']
    del kwargs['policy']
    del kwargs['total_timesteps']


    model = deepq.learn(env=env,
                      network=policy,
                      total_timesteps=n_timesteps,
                      seed=seed,
                      callback=callback,
                      **kwargs,
                      )

    return model
