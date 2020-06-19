import os
import gym
from ray.tune.registry import register_env

def make_env(env_id, env_args, seed=1, is_train=True, with_vecnorm=False):

    if is_train:
        env = gym.make(env_id, **env_args)
        # TODO: how to define eval env in rllib?
        eval_env = None
    else:
        env = gym.make(env_id, **env_args)
        eval_env = None

    # Register the environment in the Ray database
    register_env("env_id", env)

    return env, eval_env

# TODO: define here callbacks
