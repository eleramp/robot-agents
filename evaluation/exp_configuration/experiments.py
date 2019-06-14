import re
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import robot_agents

_EXPERIMENTS = []

def register_experiment(experiment):
    for e in _EXPERIMENTS:
        if e['name'] == experiment['name']:
            raise ValueError('Experiment with name {} already registered!' .format( e['name']) )

    if not experiment['algo']['RLlibrary'] in robot_agents.ALGOS:
        raise ValueError('Library {} not found! Known libraries {}' \
                    .format( experiment['algo']['RLlibrary'] , robot_agents.ALGOS.keys()))

    if not experiment['algo']['name'] in robot_agents.ALGOS[experiment['algo']['RLlibrary']]:
        raise ValueError('Algorithm {} not found! Known algorithms in {} are {}' \
                    .format( experiment['algo']['name'], experiment['algo']['RLlibrary'] , robot_agents.ALGOS[experiment['algo']['RLlibrary']]))

    _EXPERIMENTS.append(experiment)


def list_experiments():
    return [e['name'] for e in _EXPERIMENTS]


def get_experiment(experiment_name):
    for e in _EXPERIMENTS:
        if e['name'] == experiment_name:
            return e
    raise ValueError('{} not found! Known experiments: {}' .format(experiment_name, list_experiments()))


## ---- icub reach exps ---- ##
_env_prefix = 'pybullet_robot_envs:'

register_experiment({
    'name': 'icub_push_deepq/stable_baselines',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DQN algorithm',
    'tasks': [  {'sub_name':'fixed_obj_3dim', 'env_id':_env_prefix+'iCubPush-v0', 'seed': 1,
                'env_params':{'isDiscrete':1, 'control_arm':'l','rnd_obj_pose':0, 'maxSteps':1000, 'useOrientation':0, 'renders':False}, #'trials': 1,
                },
                #{'sub_name':'random_obj_3dim', 'env_id':_env_prefix+'iCubReach-v0', 'seed': 1,
                #            'env_params':{'isDiscrete':1, 'control_arm':'l','rnd_obj_pose':1, 'maxSteps':1000, 'renders':False},
                #},
                {'sub_name':'fixed_obj_6dim', 'env_id':_env_prefix+'iCubReach-v0', 'seed': 1,
                'env_params':{'isDiscrete':1, 'control_arm':'l','rnd_obj_pose':0, 'maxSteps':1000, 'useOrientation':1, 'renders':False}, #'trials': 1,
                },
                #{'sub_name':'random_obj_6dim', 'env_id':_env_prefix+'iCubReach-v0', 'seed': 1,
                #'env_params':{'isDiscrete':1, 'control_arm':'l','rnd_obj_pose':1, 'maxSteps':1000, 'useOrientation':1, 'renders':False},
                #},
            ],
    'algo': {   'name' : 'deepq',
                'RLlibrary': 'stable_baselines_agents',
                'description': 'DQN algorithm from stable_baselines library',
                'params' : {'policy': 'MlpPolicy',
                            'n_timesteps': 500000,
                            'learning_rate':1e-3,
                            'buffer_size':100000,
                            'batch_size': 16,
                            #the algorithm explore for exploration_fraction*n_timesteps,
                            # with e linearly annealed from 1.0 to exploration_final_eps
                            'exploration_fraction':0.2,
                            'exploration_final_eps':0.02,
                            'param_noise':False,
                            'policy_kwargs':dict(layers=[64])},
                            'prioritized_replay': True,
                            },
})

register_experiment({
    'name': 'icub_reach_ddpg/stable_baselines/random_obj_3dim',
    'description': 'Reach task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'', 'env_id':_env_prefix+'iCubReach-v0', 'seed': 1,
                'env_params':{'useIK':1,'isDiscrete':0, 'control_arm':'l','rnd_obj_pose':1, 'maxSteps':1000, 'useOrientation':0, 'renders':False},
                },
                ],
    'algo': {   'name' : 'ddpg',
                'RLlibrary': 'stable_baselines_agents',
                'description': 'DDPG algorithm from stable_baselines library',
                'params' : {'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                            'policy': 'LnMlpPolicy', #options: MlpPolicy e cnns ones
                            'n_timesteps': 1000000,
                            'batch_size': 16,
                            'gamma': 0.99,
                            'normalize_observations': True,
                            'normalize_returns': False,
                            'memory_limit': 100000},
            },
})

register_experiment({
    'name': 'icub_push_ddpg/stable_baselines/fixed_obj_3dim',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'', 'env_id':_env_prefix+'iCubPush-v0', 'seed': 1,
                'env_params':{'useIK':1,'isDiscrete':0, 'control_arm':'l','rnd_obj_pose':0, 'maxSteps':1000, 'useOrientation':0, 'renders':False},
                },
                ],
    'algo': {   'name' : 'ddpg',
                'RLlibrary': 'stable_baselines_agents',
                'description': 'DDPG algorithm from stable_baselines library',
                'params' : {'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                            'policy': 'LnMlpPolicy', #options: MlpPolicy e cnns ones
                            'n_timesteps': 1000000,
                            'batch_size': 16,
                            'gamma': 0.99,
                            'normalize_observations': True,
                            'normalize_returns': False,
                            'memory_limit': 100000},
            },

})

register_experiment({
    'name': 'icub_push_ddpg/stable_baselines/fixed_obj_6dim',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'', 'env_id':_env_prefix+'iCubPush-v0', 'seed': 1,
                'env_params':{'useIK':1,'isDiscrete':0, 'control_arm':'l','rnd_obj_pose':0, 'maxSteps':1000, 'useOrientation':1, 'renders':False},
                },
                ],
    'algo': {   'name' : 'ddpg',
                'RLlibrary': 'stable_baselines_agents',
                'description': 'DDPG algorithm from stable_baselines library',
                'params' : {'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                            'policy': 'LnMlpPolicy', #options: MlpPolicy e cnns ones
                            'n_timesteps': 1000000,
                            'batch_size': 16,
                            'gamma': 0.99,
                            'normalize_observations': True,
                            'normalize_returns': False,
                            'memory_limit': 100000},
            },

})
