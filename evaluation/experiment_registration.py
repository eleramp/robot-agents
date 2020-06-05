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


# icub grasp
register_experiment({
    'name': 'icub_grasp_3objs_fixed/2_control_pts',
    'description': '3 objects in fixed position.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'iCubGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 600,
                            'n_control_pt': n_control_pt,
                            'obj_name': None,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 500000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':12,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})


# panda grasp

register_experiment({
    'name': 'SQ_DRL_sisq/panda_grasp_1obj_v0/2_control_pts/obj_0',
    'description': '1 obj (mustard). random position and orientation. reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.07,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
                            'n_control_pt': n_control_pt,
                            'obj_name': 0,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 500000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':12,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq/panda_grasp_1obj_v0/2_control_pts/obj_1',
    'description': '1 obj (mustard). random position and orientation. reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.07,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
                            'n_control_pt': n_control_pt,
                            'obj_name': 1,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 500000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':12,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'SQ_DRL_nosq/panda_grasp_1obj_v0/1_control_pts/obj_1',
    'description': '1 obj (mustard). random position and orientation. reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.07,
                            'noise_pcl': 0.00,
                            'use_superq': 0,
                            'max_steps': 500,
                            'n_control_pt': n_control_pt,
                            'obj_name': 1,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 500000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':12,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq/panda_grasp_1obj_v0/2_control_pts/obj_2',
    'description': '1 obj (mustard). random position and orientation. reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.07,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
                            'n_control_pt': n_control_pt,
                            'obj_name': 2,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 500000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':12,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq/panda_grasp_1obj_v0/2_control_pts/obj_2',
    'description': '1 obj (mustard). random position and orientation. reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.07,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
                            'n_control_pt': n_control_pt,
                            'obj_name': 2,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 500000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':12,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})



'''
BASELINE EXPERIMENTS
'''

n_control_pt = 2
register_experiment({
    'name': 'SQ_DRL_sisq/panda_grasp_1obj/2_control_pts/obj_0-baseline',
    'description': '1 obj (mustard). random position and orientation. reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.07,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
                            'n_control_pt': n_control_pt,
                            'obj_name': 0,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 200000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 100000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':12,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq/panda_grasp_1obj/2_control_pts/obj_1-baseline',
    'description': '1 obj (mustard). random position and orientation. reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.07,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
                            'n_control_pt': n_control_pt,
                            'obj_name': 1,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 100000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 100000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':12,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq/panda_grasp_1obj/2_control_pts/obj_2-baseline',
    'description': '1 obj (mustard). random position and orientation. reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.07,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
                            'n_control_pt': n_control_pt,
                            'obj_name': 2,
                            'r_weights': [med_r, -10, 10],
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 100000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 100000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':12,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

# BASIC EXP
register_experiment({
    'name': 'basic_exp/sisq/panda_grasp_1obj/obj_0',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': 0,
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
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [256, 256]},
                        },
             },
})

register_experiment({
    'name': 'basic_exp/sisq/panda_grasp_1obj/obj_1',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': 1,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 30000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [256, 256]},
                        },
             },
})

register_experiment({
    'name': 'basic_exp/sisq/panda_grasp_1obj/obj_1_auto_ent',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': 1,
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
                        'ent_coef': 'auto_0.4',
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [256, 256]},
                        },
             },
})

register_experiment({
    'name': 'basic_exp/sisq/panda_grasp_1obj/obj_2',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
                            'n_control_pt': n_control_pt,
                            'obj_name': 2,
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
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [256, 256]},
                        },
             },
})

register_experiment({
    'name': 'basic_exp/nosq/panda_grasp_1obj/obj_1',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 0,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': 1,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 30000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [256, 256]},
                        },
             },
})

register_experiment({
    'name': 'basic_exp/sisq_novel/panda_grasp_1obj/obj_1',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': 1,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 30000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':1,
                        'policy_kwargs': {'layers': [256, 256]},
                        },
             },
})

# MULTI OBJECTS
register_experiment({
    'name': 'multi_obj/sisq/panda_grasp_4objs',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.1,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': None,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 500000,
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
    'name': 'multi_obj/nosq/panda_grasp_4objs',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.1,
                            'noise_pcl': 0.00,
                            'use_superq': 0,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': None,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 500000,
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
    'name': 'multi_obj/cv/panda_grasp_8cylinders',
    'description': 'train on 8 cylinders of similar dim to see if it can generalize on unseen similar shapes in test',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidualCvCyl-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.1,
                            'noise_pcl': 0.001,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': None,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 200000,
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
    'name': 'multi_obj/cv_nosq/panda_grasp_8cylinders',
    'description': 'train on 8 cylinders of similar dim to see if it can generalize on unseen similar shapes in test',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidualCvCyl-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.1,
                            'noise_pcl': 0.001,
                            'use_superq': 0,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': None,
                            'renders': False}
              },
             ],
    'algo': {'name': 'sac_residual',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 200000,
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
    'name': 'multi_obj_fixed/sisq_r_bonus/panda_grasp_4objs',
    'description': 'add superquadric context related info in reward',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': None,
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
    'name': 'multi_obj_fixed/sisq/panda_grasp_4objs',
    'description': 'add superquadric context related info in reward',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': None,
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
    'name': 'multi_obj_fixed/nosq/panda_grasp_4objs',
    'description': '1 obj. same pose',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'pandaGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 0,
                            'max_steps': 1000,
                            'n_control_pt': n_control_pt,
                            'obj_name': None,
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

