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
                'RLlibrary': 'stable_baselines_agents',
                'description': 'sac algorithm from stable_baselines library',
                'params' : {#'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                            'policy': 'LnMlpPolicy', #options: MlpPolicy e cnns ones
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
    'name': 'SQ_DRL_1',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'icub_grasp_residual/terminal_failure', 'env_id':_env_prefix+'iCubGraspResidual-v0', 'seed': 1,
                'env_params':{'control_arm':'r', 'rnd_obj_pose':0.06, 'noise_pcl': 0.012,
                              'maxSteps':3000, 'useOrientation':1, 'renders':True, 'terminal_failure': True},
                },
                ],
    'algo': {   'name' : 'residual_sac',
                'RLlibrary': 'stable_baselines_agents',
                'description': 'sac algorithm from stable_baselines library',
                'params' : {#'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                            'policy': 'LnMlpPolicy', #options: MlpPolicy e cnns ones
                            'n_timesteps': 1000000,
                            'learning_rate': 3e-4,
                            'learning_rate_pi': 3e-4,
                            'batch_size': 64,
                            'gamma': 0.99,
                            'train_freq': 10,
                            'gradient_steps': 1,
                            'learning_starts': 1000,
                            'buffer_size': 1000000,
                            'continue': False},
            },

})

register_experiment({
    'name': 'SQ_DRL_2',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'icub_grasp_residual/NO_terminal_failure', 'env_id':_env_prefix+'iCubGraspResidual-v0', 'seed': 1,
                'env_params':{'control_arm':'r', 'rnd_obj_pose':0.06, 'noise_pcl': 0.012,
                              'maxSteps':3000, 'useOrientation':1, 'renders':True, 'terminal_failure': False},
                },
                ],
    'algo': {   'name' : 'residual_sac',
                'RLlibrary': 'stable_baselines_agents',
                'description': 'sac algorithm from stable_baselines library',
                'params' : {#'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                            'policy': 'LnMlpPolicy', #options: MlpPolicy e cnns ones
                            'n_timesteps': 1000000,
                            'learning_rate': 3e-4,
                            'learning_rate_pi': 3e-4,
                            'batch_size': 64,
                            'gamma': 0.99,
                            'train_freq': 10,
                            'gradient_steps': 1,
                            'learning_starts': 1000,
                            'buffer_size': 1000000,
                            'continue': False},
                },
})

register_experiment({
    'name': 'SQ_DRL_3_detactions',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [  { 'sub_name':'icub_grasp_residual/terminal_failure', 'env_id':_env_prefix+'iCubGraspResidual-v0', 'seed': 1,
                'env_params':{'control_arm':'r', 'rnd_obj_pose':0.06, 'noise_pcl': 0.012,
                              'maxSteps':3000, 'useOrientation':1, 'renders':True, 'terminal_failure': True},
                },
                ],
    'algo': {   'name' : 'residual_sac',
                'RLlibrary': 'stable_baselines_agents',
                'description': 'sac algorithm from stable_baselines library',
                'params' : {#'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                            'policy': 'LnMlpPolicy', #options: MlpPolicy e cnns ones
                            'n_timesteps': 1000000,
                            'learning_rate': 3e-4,
                            'learning_rate_pi': 3e-4,
                            'batch_size': 64,
                            'gamma': 0.99,
                            'train_freq': 10,
                            'gradient_steps': 1,
                            'learning_starts': 50,
                            'buffer_size': 1000000,
                            'continue': False},
                },
})
