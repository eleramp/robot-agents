import gym


#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# import RL agent
from stable_baselines.deepq.policies import MlpPolicy as policy
from stable_baselines.deepq import DQN
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from robot_agents.utils import log_callback
#
import numpy as np
import math as m

best_mean_reward, n_steps = -np.inf, 0
global output_dir

def train_DQN( env, out_dir, seed=None, **kwargs):

    # Logs will be saved in log_dir/monitor.csv
    global output_dir
    output_dir = out_dir
    log_dir = os.path.join(out_dir,'log')
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir+'/', allow_early_resets=True)

    global n_steps, best_mean_reward
    best_mean_reward, n_steps = -np.inf, 0

    policy = kwargs['policy']
    n_timesteps = kwargs['n_timesteps']
    del kwargs['policy']
    del kwargs['n_timesteps']

    model = DQN(policy, env,  verbose=1, tensorboard_log=os.path.join(log_dir,'tb'),
                full_tensorboard_log=True, checkpoint_path=log_dir, seed=seed, **kwargs)

    model.learn(total_timesteps=n_timesteps, callback=log_callback)

    return model
