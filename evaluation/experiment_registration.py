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
    'name': 'icub_reach_ddpg/stable_baselines',
    'description': 'Reach task with iCub robot, simulated in PyBullet by using DQN algorithm',
    'tasks': [  #{'sub_name':'fixed_obj_6dim', 'env_id':_env_prefix+'iCubReach-v0', 'seed': 1,
                #'env_params':{'isDiscrete':1, 'control_arm':'l','rnd_obj_pose':0, 'maxSteps':1000, 'useOrientation':1, 'renders':False}
                #},
                {'sub_name':'random_obj_6dim', 'env_id':_env_prefix+'iCubReach-v0', 'seed': 1,
                'env_params':{'useIK':1, 'isDiscrete':0, 'control_arm':'l','rnd_obj_pose':1, 'maxSteps':1000, 'useOrientation':1, 'renders':True},
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
                'env_params':{'useIK':1,'isDiscrete':0, 'control_arm':'l','rnd_obj_pose':0,
                              'maxSteps':2000, 'useOrientation':1, 'reward_type':1, 'renders':False},
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
    'name': 'icub_push_ddpg/stable_baselines/random_tg_6dim',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'', 'env_id':_env_prefix+'iCubPush-v0', 'seed': 1,
                'env_params':{'useIK':1,'isDiscrete':0, 'control_arm':'l','rnd_obj_pose':1,
                              'maxSteps':2000, 'useOrientation':1, 'reward_type':1, 'renders':True},
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
    'name': 'icub_push_sac/stable_baselines/random_tg_6dim_rew_0',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'', 'env_id':_env_prefix+'iCubPush-v0', 'seed': 1,
                'env_params':{'useIK':1,'isDiscrete':0, 'control_arm':'l','rnd_obj_pose':1,
                              'maxSteps':2000, 'useOrientation':1, 'reward_type':0, 'renders':True},
                },
                ],
    'algo': {   'name' : 'sac',
                'RLlibrary': 'stable_baselines_agents',
                'description': 'sac algorithm from stable_baselines library',
                'params' : {#'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                            'policy': 'LnMlpPolicy', #options: MlpPolicy e cnns ones
                            'n_timesteps': 1000000,
                            'learning_rate': 'lin_3e-4',
                            'batch_size': 16,
                            'gamma': 0.99,
                            'train_freq': 1,
                            'gradient_steps': 1,
                            'learning_starts': 1000,
                            'buffer_size': 1000000,
                            'continue': True},
            },

})

register_experiment({
    'name': 'icub_push_sac/stable_baselines/random_tg_6dim_rew_2',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'', 'env_id':_env_prefix+'iCubPush-v0', 'seed': 1,
                'env_params':{'useIK':1,'isDiscrete':0, 'control_arm':'l','rnd_obj_pose':1,
                              'maxSteps':2000, 'useOrientation':1, 'reward_type':2, 'renders':True},
                },
                ],
    'algo': {   'name' : 'sac',
                'RLlibrary': 'stable_baselines_agents',
                'description': 'sac algorithm from stable_baselines library',
                'params' : {#'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                            'policy': 'LnMlpPolicy', #options: MlpPolicy e cnns ones
                            'n_timesteps': 9000000,
                            'learning_rate': 'lin_3e-4',
                            'batch_size': 16,
                            'gamma': 0.99,
                            'train_freq': 1,
                            'gradient_steps': 1,
                            'learning_starts': 1000,
                            'buffer_size': 1000000,
                            'continue': True},
            },

})
