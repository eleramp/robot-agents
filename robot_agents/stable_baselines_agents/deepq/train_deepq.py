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

#
import numpy as np
import math as m

best_mean_reward, n_steps = -np.inf, 0
global output_dir

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(os.path.join(output_dir,'log')), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(os.path.join(output_dir,'best_model.pkl'))
    n_steps += 1
    # Returning False will stop training early
    return True

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
                full_tensorboard_log=True, checkpoint_path=log_dir, **kwargs)

    model.learn(total_timesteps=n_timesteps, seed=seed, callback=callback)

    return model
