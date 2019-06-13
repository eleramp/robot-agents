import gym


#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# import RL agent
from stable_baselines.deepq.policies import MlpPolicy as policy
from stable_baselines import DDPG

import numpy as np

def evaluate(env, model, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  for i in range(num_steps):
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)
      obs, reward, done, info = env.step(action)

      # Stats
      episode_rewards[-1] += reward
      if done:
          obs = env.reset()
          episode_rewards.append(0.0)
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

  return mean_100ep_reward


def test_DDPG( env, out_dir, seed=None, **kwargs):

  model = DDPG.load(os.path.join(out_dir,'best_model.pkl'))

  # Evaluate the trained agent
  mean_reward = evaluate(env, model, num_steps=5000)

  return
