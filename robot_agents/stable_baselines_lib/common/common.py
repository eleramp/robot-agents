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
                 n_eval_episodes: int = 10,
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

            self.eval_env.seed(self.seed)

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
        eval_env = make_vec_env(env_id=lambda: gym.make(env_id, **env_args),
                           seed=seed+1, monitor_dir=monitor_dir+'/eval', n_envs=1)

        if with_vecnorm:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

    else:
        env = gym.make(env_id, **env_args)
        eval_env = None

    return env, eval_env


def get_train_callback(eval_env, seed, log_dir):
    checkpoint_callback = CheckpointCallback(save_freq=3000, save_path=log_dir)

    # Separate evaluation env
    eval_callback = EvalTensorboardCallback(eval_env, best_model_save_path=os.path.join(log_dir, 'best_model'),
                                            log_path=os.path.join(log_dir, 'evaluation_results'), eval_freq=3000,
                                            deterministic=True, render=False, seed=seed)

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    return callback

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
    plt.show()

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


path = '/home/erampone/workspace/phd/pybullet_robot_agents_logs/2020_04_02/no_terminal_obs/basic_exp/sisq/panda_grasp_1obj/obj_1-sac_residual/evaluation_results'
#load_evaluation_results(path)