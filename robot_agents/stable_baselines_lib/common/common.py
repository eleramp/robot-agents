import os
import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.bench import Monitor
import numpy as np
from stable_baselines import results_plotter


def make_env(env_id, env_args, seed, is_train):

    monitor_dir = os.path.join(env_args['log_file'], 'log')

    if is_train:
        # env for training
        env = make_vec_env(env_id=lambda: gym.make(env_id, **env_args),
                           seed=seed, monitor_dir=monitor_dir, n_envs=1)

        # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

        # env for evaluation during training
        env_args['renders'] = False
        eval_env = make_vec_env(env_id=lambda: gym.make(env_id, **env_args),
                           seed=seed+1, monitor_dir=monitor_dir+'/eval', n_envs=1)

        # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    else:
        env = gym.make(env_id, **env_args)
        eval_env = None

    return env, eval_env

def load_evaluation_results(file_path):
    data = np.load(os.path.join(file_path, 'evaluations.npz'))
    results_plotter.plot_curves([(data.f.timesteps, data.f.results)], 'timesteps', 'evaluation_results')

load_evaluation_results('/home/erampone/workspace/phd/robot-agents')