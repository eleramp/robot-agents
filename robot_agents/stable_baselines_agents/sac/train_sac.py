import gym

#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


# import RL agent
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.sac.policies import MlpPolicy, LnMlpPolicy
from stable_baselines import SAC
from stable_baselines_agents.sac.sac_residual import SAC_residual

from stable_baselines.ppo2.ppo2 import constfn
from robot_agents.utils import linear_schedule

#
import numpy as np
import math as m
global output_dir

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if 'num_episodes' in _locals:
        if _locals['num_episodes'] % 10 == 0:
            # Evaluate policy training performance
            if len(_locals['episode_rewards'][-101:-1]) == 0:
                mean_reward = -np.inf
            else:
                mean_reward = round(float(np.mean(_locals['episode_rewards'][-101:-1])), 1)

            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
            if _locals['num_episodes'] % 100 == 0:
                # Save model
                print("Saving model at iter {}".format(_locals['step']))
                locals['self'].save(os.path.join(output_dir, str(_locals['step'])+'model.pkl'))
    n_steps += 1
    # Returning False will stop training early
    return True


def train_SAC( env, out_dir, seed=None, **kwargs):

    # Logs will be saved in log_dir/monitor.csv
    global output_dir
    output_dir = out_dir
    log_dir = os.path.join(out_dir,'log')
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir+'/', allow_early_resets=True)

    # Delete keys so the dict can be pass to the model constructor
    policy = kwargs['policy']
    n_timesteps = kwargs['n_timesteps']
    noise_type = None
    if 'noise_type' in kwargs:
        noise_type = kwargs['noise_type']
        del kwargs['noise_type']
    del kwargs['policy']
    del kwargs['n_timesteps']

    # the noise objects - usually not necessary for SAC but can help for hard exploration tasks
    nb_actions = env.action_space.shape[-1]
    action_noise = None
    if not noise_type is None:

        for current_noise_type in noise_type.split(','):

            current_noise_type = current_noise_type.strip()

            if 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))

            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(nb_actions),
                sigma=float(stddev) * np.ones(nb_actions))

            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Create learning rate schedule
    for key in ['learning_rate', 'learning_rate_pi', 'cliprange']:
        if key in kwargs:
            if isinstance(kwargs[key], str):
                schedule, initial_value = kwargs[key].split('_')
                initial_value = float(initial_value)
                kwargs[key] = linear_schedule(initial_value)
            elif isinstance(kwargs[key], float):
                kwargs[key] = constfn(kwargs[key])
            else:
                raise ValueError('Invalid valid for {}: {}'.format(key, hyperparams[key]))

    if 'continue' in kwargs and kwargs['continue'] is True:
        # Continue training
        print("Loading pretrained agent")
        model = SAC.load(os.path.join(out_dir,'final_model.pkl'), env=env,
                         tensorboard_log=os.path.join(log_dir,'tb'), verbose=1, **kwargs)
    else:
        if 'continue' in kwargs:
            del kwargs['continue']
        model = SAC_residual(policy, env, action_noise=action_noise,
                    verbose=1, tensorboard_log=os.path.join(log_dir,'tb'), full_tensorboard_log=False, **kwargs)

    model.learn(total_timesteps=n_timesteps, seed=seed, callback=callback, log_interval=10)

    return model
