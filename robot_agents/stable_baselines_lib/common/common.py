import os
import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.bench import Monitor
import numpy as np
from stable_baselines import results_plotter
import matplotlib
import matplotlib.pyplot as plt

EPISODES_WINDOW = 80


def make_env(env_id, env_args, seed, is_train, with_vecnorm):

    monitor_dir = os.path.join(env_args['log_file'], 'log')

    if is_train:
        # env for training
        env = make_vec_env(env_id=lambda: gym.make(env_id, **env_args),
                           seed=seed, monitor_dir=monitor_dir, n_envs=1)

        if with_vecnorm:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

        # env for evaluation during training
        env_args['renders'] = False
        eval_env = make_vec_env(env_id=lambda: gym.make(env_id, **env_args),
                           seed=seed+1, monitor_dir=monitor_dir+'/eval', n_envs=1)

        if with_vecnorm:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    else:
        env = gym.make(env_id, **env_args)
        eval_env = None

    return env, eval_env

def plot_curves(xy_list, xaxis, title):
    """
    plot the curves

    :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: (str) the title of the plot
    """

    plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = results_plotter.COLORS[i]
        plt.plot(x, y, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()

def load_evaluation_results(file_path):
    data = np.load(os.path.join(file_path, 'evaluations.npz'))
    res = data.f.results
    avg_res = []
    for ep in res:
        avg = 0
        for r in ep:
            avg += r
        avg /= len(ep)
        avg_res.append(avg)

    plot_curves([(data.f.timesteps, np.array(avg_res))], 'timesteps', 'evaluation_results')
    a = 1

# load_evaluation_results('/home/erampone/workspace/phd/pybullet_robot_agents_logs/2020_04/SQ_DRL_sisq/panda_grasp_1obj_v0/1_control_pts/obj_1-sac_residual/evaluation_results')