import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import robot_agents
import os

_EXPERIMENTS = []

def register_experiment(experiment):
    for e in _EXPERIMENTS:
        if e['name']+e['algo']['name'] == experiment['name']+experiment['algo']['name']:
            raise ValueError('Experiment with name {} already registered!' .format( e['name']) )

    if not experiment['algo']['RLlibrary'] in robot_agents.ALGOS:
        raise ValueError('Library {} not found! Known libraries {}' \
                    .format( experiment['algo']['RLlibrary'] , robot_agents.ALGOS.keys()))

    if not experiment['algo']['name'] in robot_agents.ALGOS[experiment['algo']['RLlibrary']]:
        raise ValueError('Algorithm {} not found! Known algorithms in {} are {}' \
                    .format( experiment['algo']['name'], experiment['algo']['RLlibrary'] , robot_agents.ALGOS[experiment['algo']['RLlibrary']]))

    _EXPERIMENTS.append(experiment)


def list_experiments():
    return [e['name']+'-'+e['algo']['name'] for e in _EXPERIMENTS]


def get_experiment(experiment_name):
    for e in _EXPERIMENTS:
        if e['name']+'-'+e['algo']['name'] == experiment_name:
            return e
    raise ValueError('{} not found! Known experiments: {}' .format(experiment_name, list_experiments()))


## ---- icub reach exps ---- ##

_env_prefix = 'pybullet_robot_envs:'

# 1
n_control_pt = 2
low_r, med_r, high_r = -2, -5, -8


register_experiment({
    'name': 'superquadric_dset/sisq/panda_grasp',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'PandaGraspResidualGymEnvSqObj-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'dset': 'train',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'obj_orn_rnd': 1.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 50000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess': 12,
                        'policy_kwargs': {'layers': [256, 256]},
                        },
             },
})

register_experiment({
    'name': 'superquadric_dset/nosq/panda_grasp',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'PandaGraspResidualGymEnvSqObj-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'dset': 'train',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'obj_orn_rnd': 1.0,
                            'noise_pcl': 0.00,
                            'use_superq': 0,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 50000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess': 12,
                        'policy_kwargs': {'layers': [256, 256]},
                        },
             },
})

