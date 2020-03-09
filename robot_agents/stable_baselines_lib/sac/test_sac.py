import gym


#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# import RL agent
from stable_baselines.sac.policies import MlpPolicy, LnMlpPolicy
from stable_baselines import SAC
from robot_agents.stable_baselines_lib.sac.sac_residual import SAC_residual

import numpy as np
import time

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
      action, _states = model.predict(obs, deterministic=False)

      obs, reward, done, info = env.step(action)

      # Stats
      episode_rewards[-1] += reward
      if done:
          print("Episode reward: ", episode_rewards[-1])
          time.sleep(1)
          obs = env.reset()
          episode_rewards.append(0.0)




  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

  return mean_100ep_reward


def test_SAC( env, out_dir, seed=None, **kwargs):

  model = SAC.load(os.path.join(out_dir,'final_model.pkl'), env=env)

  #model.learn(total_timesteps=10000)
  # Evaluate the trained agent
  mean_reward = evaluate(env, model, num_steps=5000)

  return

def test_SAC_residual( env, out_dir, seed=None, **kwargs):

  model = SAC_residual.load(os.path.join(out_dir,'final_model.pkl'), env=env)

  #model.learn(total_timesteps=10000)
  # Evaluate the trained agent
  mean_reward = evaluate(env, model, num_steps=5000)

  return
