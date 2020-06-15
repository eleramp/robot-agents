import os
import gym
from typing import Union, Optional
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecEnv, VecNormalize, sync_envs_normalization
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines import results_plotter
import matplotlib.pyplot as plt
import tensorflow as tf

EPISODES_WINDOW = 80


class EvalTensorboardCallback(EvalCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 100,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 seed: int = 1,
                 verbose: int = 1):

        self.is_tb_set = False
        self.seed = seed

        super(EvalTensorboardCallback, self).__init__(eval_env=eval_env,
                                                  callback_on_new_best=callback_on_new_best,
                                                  n_eval_episodes=n_eval_episodes,
                                                  eval_freq=eval_freq,
                                                  log_path=log_path,
                                                  best_model_save_path=best_model_save_path,
                                                  deterministic=deterministic,
                                                  render=render,
                                                  verbose=verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        if not self.is_tb_set:
            with self.model.graph.as_default():
                tf.summary.scalar('eval_episode_reward', self.last_mean_reward)
                self.model.summary = tf.summary.merge_all()
            self.is_tb_set = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # self.eval_env.seed(self.seed)

            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        # Log episode mean reward
        summary = tf.Summary(value=[tf.Summary.Value(tag='eval_episode_reward', simple_value=self.last_mean_reward)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)

        return True


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
        if 'dset' in env_args:
            env_args['dset'] = 'eval'
        eval_env = make_vec_env(env_id=lambda: gym.make(env_id, **env_args),
                           seed=seed+1, monitor_dir=monitor_dir+'/eval', n_envs=1)

        if with_vecnorm:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    else:
        env = gym.make(env_id, **env_args)
        eval_env = None

    return env, eval_env


def get_train_callback(eval_env, seed, log_dir, save_f=10000, eval_f=50000, eval_ep=1000):
    checkpoint_callback = CheckpointCallback(save_freq=save_f, save_path=log_dir)

    # Separate evaluation env
    eval_callback = EvalTensorboardCallback(eval_env, best_model_save_path=os.path.join(log_dir, 'best_model'),
                                            log_path=os.path.join(log_dir, 'evaluation_results'), eval_freq=eval_f,
                                            n_eval_episodes=eval_ep, deterministic=True, render=False, seed=seed)

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    return callback


def plot_curves(xyz_list, xaxis, title):
    """
    plot the curves

    :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: (str) the title of the plot
    """

    plt.figure(figsize=(8, 2))
    maxx = max(xyz[0][-1] for xyz in xyz_list)
    minx = 0
    for (i, (x, y, z)) in enumerate(xyz_list):
        color = results_plotter.COLORS[i]
        # plt.plot(x, y, color=color)
        plt.errorbar(x, y, z, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()
    plt.show()


def load_evaluation_results(file_path):
    data = np.load(os.path.join(file_path, 'evaluations.npz'))
    res = data.f.results
    avg_res = [np.mean(ep) for ep in res]
    std_res = [np.std(ep) for ep in res]

    plot_curves([(data.f.timesteps, avg_res, std_res)], 'timesteps', 'evaluation_results')
    a = 1


def load_actions(file_path):
    data = np.load(file_path)
    x_vals = data.f.x
    y_vals = data.f.y
    z_vals = data.f.z
    roll_vals = data.f.roll
    pitch_vals = data.f.pitch
    yaw_vals = data.f.yaw

    df_x = pd.DataFrame(x_vals, columns=['x'])
    df_y = pd.DataFrame(y_vals, columns=['y'])
    df_z = pd.DataFrame(z_vals, columns=['z'])
    df_roll = pd.DataFrame(roll_vals, columns=['roll'])
    df_pitch = pd.DataFrame(pitch_vals, columns=['pitch'])
    df_yaw = pd.DataFrame(yaw_vals, columns=['yaw'])

    # Make the plot
    df_x.plot.hist(bins=100, alpha=0.5)
    x_min, x_max = min(x_vals), max(x_vals)
    x_mean, x_std = np.mean(x_vals), np.std(x_vals)
    title = "min: " + str(round(float(x_min), 2)) + ", max: " + str(round(float(x_max), 2)) + ", mean: " + str(
        round(float(x_mean), 2)) + ", std: " + str(round(float(x_std), 2))
    plt.title(title, fontsize=12, fontweight=0, color='black')

    df_y.plot.hist(bins=100, alpha=0.5)
    y_min, y_max = min(y_vals), max(y_vals)
    y_mean, y_std = np.mean(y_vals), np.std(y_vals)
    title = "min: " + str(round(float(y_min), 2)) + ", max: " + str(round(float(y_max), 2)) + ", mean: " + str(
        round(float(y_mean), 2)) + ", std: " + str(round(float(y_std), 2))
    plt.title(title, fontsize=12, fontweight=0, color='black')

    df_z.plot.hist(bins=100, alpha=0.5)
    z_min, z_max = min(z_vals), max(z_vals)
    z_mean, z_std = np.mean(z_vals), np.std(z_vals)
    title = "min: " + str(round(float(z_min), 2)) + ", max: " + str(round(float(z_max), 2)) + ", mean: " + str(
        round(float(z_mean), 2)) + ", std: " + str(round(float(z_std), 2))
    plt.title(title, fontsize=12, fontweight=0, color='black')

    df_roll.plot.hist(bins=100, alpha=0.5)
    roll_min, roll_max = min(roll_vals), max(roll_vals)
    roll_mean, roll_std = np.mean(roll_vals), np.std(roll_vals)
    title = "min: " + str(round(float(roll_min), 2)) + ", max: " + str(round(float(roll_max), 2)) + ", mean: " + str(
        round(float(roll_mean), 2)) + ", std: " + str(round(float(roll_std), 2))
    plt.title(title, fontsize=12, fontweight=0, color='black')

    df_pitch.plot.hist(bins=100, alpha=0.5)
    pitch_min, pitch_max = min(pitch_vals), max(pitch_vals)
    pitch_mean, pitch_std = np.mean(pitch_vals), np.std(pitch_vals)
    title = "min: " + str(round(float(pitch_min), 2)) + ", max: " + str(round(float(pitch_max), 2)) + ", mean: " + str(
        round(float(pitch_mean), 2)) + ", std: " + str(round(float(pitch_std), 2))
    plt.title(title, fontsize=12, fontweight=0, color='black')


    df_yaw.plot.hist(bins=100, alpha=0.5)
    yaw_min, yaw_max = min(yaw_vals), max(yaw_vals)
    yaw_mean, yaw_std = np.mean(yaw_vals), np.std(yaw_vals)
    title = "min: " + str(round(float(yaw_min), 2)) + ", max: " + str(round(float(yaw_max), 2)) + ", mean: " + str(
        round(float(yaw_mean), 2)) + ", std: " + str(round(float(yaw_std), 2))
    plt.title(title, fontsize=12, fontweight=0, color='black')
