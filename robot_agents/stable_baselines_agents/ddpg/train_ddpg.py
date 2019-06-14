import gym


#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


# import RL agent
from stable_baselines.ddpg.policies import LnMlpPolicy as policy # MlpPolicy or LnMlpPolicy (with layer normalization)
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

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

def train_DDPG( env, out_dir, seed=None, **kwargs):

    # Logs will be saved in log_dir/monitor.csv
    global output_dir
    output_dir = out_dir
    log_dir = os.path.join(out_dir,'log')
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir+'/', allow_early_resets=True)

    policy = kwargs['policy']
    n_timesteps = kwargs['n_timesteps']
    noise_type = kwargs['noise_type']
    del kwargs['policy']
    del kwargs['n_timesteps']
    del kwargs['noise_type']

    ''' Parameter space noise:
    injects randomness directly into the parameters of the agent, altering the types of decisions it makes
    such that they always fully depend on what the agent currently senses. '''

    # the noise objects for DDPG
    nb_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = None
    if not noise_type is None:

        for current_noise_type in noise_type.split(','):

            current_noise_type = current_noise_type.strip()

            if 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))

            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))

            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(nb_actions),
                sigma=float(stddev) * np.ones(nb_actions))

            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))


    model = DDPG(policy, env, param_noise=param_noise, action_noise=action_noise,
                 verbose=1, tensorboard_log=os.path.join(log_dir,'tb'),full_tensorboard_log=True, **kwargs)

    model.learn(total_timesteps=n_timesteps, seed=seed, callback=callback)

    return model
