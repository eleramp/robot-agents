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
    'name': 'SQ_DRL',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'icub_grasp_residual_trial_0', 'env_id':_env_prefix+'iCubGraspResidual-v0', 'seed': 1,
                'env_params':{'control_arm':'l', 'rnd_obj_pose':0.05, 'noise_pcl': 0.005,
                              'maxSteps':3000, 'useOrientation':1, 'renders':True, 'terminal_failure': True},
                },
                ],
    'algo': {   'name' : 'residual_sac',
                'RLlibrary': 'stable_baselines_lib',
                'description': 'sac algorithm from stable_baselines library',
                'params' : {#'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                            'policy': 'LnMlpInitZeroPolicy', #options: MlpPolicy e cnns ones
                            'n_timesteps': 1000000,
                            'learning_rate': 'lin_3e-4',
                            'learning_rate_pi': 'lin_3e-4',
                            'batch_size': 256,
                            'gamma': 0.99,
                            'train_freq': 10,
                            'gradient_steps': 1,
                            'learning_starts': 1000,
                            'buffer_size': 1000000,
                            'continue': True},
            },

})

register_experiment({
    'name': 'icub_push/fixed_obj_6dim',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name': '', 'env_id': _env_prefix+'iCubPushGoal-v0', 'seed': 1,
                'env_params':{'useIK': 1, 'isDiscrete': 0, 'control_arm': 'l', 'rnd_obj_pose': 0,
                              'maxSteps': 1000, 'useOrientation': 1, 'renders': False},
                },
                ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'td3',
                        'random_exploration': 0.3,
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'policy': 'LnMlpPolicy',  # options: MlpPolicy e cnns ones
                        'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'n_timesteps': 1000000,
                        'gradient_steps': 40,
                        'batch_size': 128,
                        'gamma': 0.95,
                        'buffer_size': 1000000,
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [64, 64, 64]}},
             },

})

register_experiment({
    'name': 'icub_push/fixed_obj_6dim_00',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name': '', 'env_id': _env_prefix+'iCubPushGoal-v0', 'seed': 1,
                'env_params':{'useIK': 1, 'isDiscrete': 0, 'control_arm': 'l', 'rnd_obj_pose': 0,
                              'maxSteps': 1000, 'useOrientation': 1, 'renders': False},
                },
                ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'ddpg',
                        'random_exploration': 0.3,
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'policy': 'LnMlpPolicy',  # options: MlpPolicy e cnns ones
                        'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'n_timesteps': 1000000,
                        'batch_size': 16,
                        'gamma': 0.95,
                        'normalize_observations': False,
                        'normalize_returns': False,
                        'buffer_size': 100000,
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [64, 64, 64]}},
             },

})

register_experiment({
    'name': 'icub_push/fixed_obj_6dim_01',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name': '', 'env_id': _env_prefix+'iCubPushGoal-v0', 'seed': 1,
                'env_params':{'useIK': 1, 'isDiscrete': 0, 'control_arm': 'l', 'rnd_obj_pose': 0,
                              'maxSteps': 1000, 'useOrientation': 1, 'renders': False},
                },
                ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'ddpg',
                        'random_exploration': 0.3,
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'policy': 'LnMlpPolicy',  # options: MlpPolicy e cnns ones
                        'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'n_timesteps': 1000000,
                        'batch_size': 16,
                        'gamma': 0.95,
                        'normalize_observations': False,
                        'normalize_returns': True,
                        'buffer_size': 100000,
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [64, 64, 64]}},
             },

})

register_experiment({
    'name': 'icub_push/fixed_obj_6dim_10',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name': '', 'env_id': _env_prefix+'iCubPushGoal-v0', 'seed': 1,
                'env_params':{'useIK': 1, 'isDiscrete': 0, 'control_arm': 'l', 'rnd_obj_pose': 0,
                              'maxSteps': 1000, 'useOrientation': 1, 'renders': False},
                },
                ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'ddpg',
                        'random_exploration': 0.3,
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'policy': 'LnMlpPolicy',  # options: MlpPolicy e cnns ones
                        'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'n_timesteps': 1000000,
                        'batch_size': 16,
                        'gamma': 0.95,
                        'normalize_observations': True,
                        'normalize_returns': False,
                        'buffer_size': 100000,
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [64, 64, 64]}},
             },

})

register_experiment({
    'name': 'icub_push/fixed_obj_6dim_11',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name': '', 'env_id': _env_prefix+'iCubPushGoal-v0', 'seed': 1,
                'env_params':{'useIK': 1, 'isDiscrete': 0, 'control_arm': 'l', 'rnd_obj_pose': 0,
                              'maxSteps': 1000, 'useOrientation': 1, 'renders': False},
                },
                ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'ddpg',
                        'random_exploration': 0.3,
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'policy': 'LnMlpPolicy',  # options: MlpPolicy e cnns ones
                        'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'n_timesteps': 1000000,
                        'batch_size': 16,
                        'gamma': 0.95,
                        'normalize_observations': True,
                        'normalize_returns': True,
                        'buffer_size': 100000,
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [64, 64, 64]}},
             },

})

register_experiment({
    'name': 'icub_push/random_tg_6dim',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'', 'env_id':_env_prefix+'iCubPushGoal-v0', 'seed': 1,
                'env_params':{'useIK':1,'isDiscrete':0, 'control_arm':'l','rnd_obj_pose':0.03,
                              'maxSteps':2000, 'useOrientation':1, 'renders':False},
                },
                ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'ddpg',
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'policy': 'LnMlpPolicy',  # options: MlpPolicy e cnns ones
                        'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'n_timesteps': 1000000,
                        'batch_size': 16,
                        'gamma': 0.99,
                        'normalize_observations': True,
                        'normalize_returns': False,
                        'buffer_size': 100000},
             },
})

register_experiment({
    'name': 'SQ_DRL_HER',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [{'sub_name': 'icub_grasp_residual', 'env_id': _env_prefix+'iCubGraspResidualGoal-v0', 'seed': 1,
               'env_params': {'control_arm': 'r', 'rnd_obj_pose': 0.06, 'noise_pcl': 0.012, 'maxSteps': 3000,
                               'useOrientation': 1, 'renders': True, 'terminal_failure': True},
              },
             ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'sac_residual',
                        #'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'policy': 'LnMlpPolicy', #options: MlpPolicy e cnns ones
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'n_timesteps': 1000000,
                        'learning_rate': 3e-4,
                        'learning_rate_pi': 3e-4,
                        'batch_size': 64,
                        'gamma': 0.99,
                        'train_freq': 10,
                        'gradient_steps': 1,
                        'learning_starts': 1,
                        'buffer_size': 1000000,
                        'continue': False},
            },

})

