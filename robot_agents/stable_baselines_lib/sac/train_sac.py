#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


# import RL agent
from stable_baselines.bench import Monitor

from stable_baselines import SAC
from robot_agents.stable_baselines_lib.sac.sac_residual import SAC_residual

from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines.common.schedules import constfn
from robot_agents.utils import linear_schedule
from robot_agents.stable_baselines_lib.common.common import get_train_callback

#
import numpy as np
import math as m


def train_SAC( env, eval_env, out_dir, seed=None, **kwargs):

    # Delete keys so the dict can be pass to the model constructor
    policy = kwargs['policy']
    n_timesteps = kwargs['n_timesteps']
    noise_type = None
    if 'noise_type' in kwargs:
        noise_type = kwargs['noise_type']
        del kwargs['noise_type']
    del kwargs['policy']
    del kwargs['n_timesteps']


    save_frequency = 10000
    eval_frequency = 50000
    eval_episodes = 1000
    if 'save_freq' in kwargs:
        save_frequency = kwargs['save_freq']
        del kwargs['save_freq']

    if 'eval_freq' in kwargs:
        eval_frequency = kwargs['eval_freq']
        del kwargs['eval_freq']

    if 'eval_episides' in kwargs:
        eval_episodes = kwargs['eval_episides']
        del kwargs['eval_episides']

    # the noise objects - usually not necessary for SAC but can help for hard exploration tasks
    nb_actions = env.action_space.shape[-1]
    action_noise = None
    if noise_type:

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
                raise ValueError('Invalid valid for {}: {}'.format(key, kwargs[key]))

    if 'continue' in kwargs and kwargs['continue'] is True:
        # Continue training
        print("Loading pretrained agent")
        model = SAC.load(os.path.join(out_dir, 'final_model.zip'), env=env,
                         tensorboard_log=os.path.join(out_dir, 'tb'), verbose=1, **kwargs)
        reset_num_timesteps=False
    else:
        if 'continue' in kwargs:
            del kwargs['continue']
        # create model
        model = SAC(policy, env, action_noise=action_noise, seed=seed,
                    verbose=1, tensorboard_log=os.path.join(out_dir, 'tb'), full_tensorboard_log=False, **kwargs)
        reset_num_timesteps=True

    # start training
    train_callback = get_train_callback(eval_env, seed, out_dir,
                        save_f=save_frequency, eval_f=eval_frequency, eval_ep=eval_episodes)
    model.learn(total_timesteps=n_timesteps, callback=train_callback, log_interval=10, reset_num_timesteps=reset_num_timesteps)

    return model


def train_SAC_residual( env, eval_env, out_dir, seed=None, **kwargs):

    # Delete keys so the dict can be pass to the model constructor
    policy = kwargs['policy']
    n_timesteps = kwargs['n_timesteps']
    noise_type = None
    if 'noise_type' in kwargs:
        noise_type = kwargs['noise_type']
        del kwargs['noise_type']
    del kwargs['policy']
    del kwargs['n_timesteps']

    save_frequency = 10000
    eval_frequency = 50000
    eval_episodes = 1000
    if 'save_freq' in kwargs:
        save_frequency = kwargs['save_freq']
        del kwargs['save_freq']

    if 'eval_freq' in kwargs:
        eval_frequency = kwargs['eval_freq']
        del kwargs['eval_freq']

    if 'eval_episides' in kwargs:
        eval_episodes = kwargs['eval_episides']
        del kwargs['eval_episides']

    # the noise objects - usually not necessary for SAC but can help for hard exploration tasks
    nb_actions = env.action_space.shape[-1]
    action_noise = None
    if noise_type:

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
                raise ValueError('Invalid valid for {}: {}'.format(key, kwargs[key]))

    if 'continue' in kwargs and kwargs['continue'] is True:
        # Continue training
        print("Loading pretrained agent")
        model = SAC_residual.load(os.path.join(out_dir,'final_model.zip'), env=env,
                         tensorboard_log=os.path.join(out_dir, 'tb'), verbose=1, **kwargs)

        reset_num_timesteps = False
    else:
        if 'continue' in kwargs:
            del kwargs['continue']

        # create model
        model = SAC_residual(policy, env, action_noise=action_noise, seed=seed,
                    verbose=1, tensorboard_log=os.path.join(out_dir, 'tb'), full_tensorboard_log=False, **kwargs)

        reset_num_timesteps = True

    # start training
    train_callback = get_train_callback(eval_env, seed, out_dir,
                        save_f=save_frequency, eval_f=eval_frequency, eval_ep=eval_episodes)

    model.learn(total_timesteps=n_timesteps, callback=train_callback, log_interval=10, reset_num_timesteps=reset_num_timesteps)

    return model
