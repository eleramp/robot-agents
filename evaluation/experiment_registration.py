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

register_experiment({
    'name': 'SQ_DRL_HER_nosq',
    'description': 'Push task with iCub robot, simulated in PyBullet by using DDPG algorithm',
    'tasks': [{'sub_name': 'icub_grasp_residual_1obj', 'env_id': _env_prefix+'iCubGraspResidualGoal-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.0,
                            'use_superq': 0,
                            'max_steps': 1000,
                            'renders': False}
              },
             ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'td3',
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'n_timesteps': 1000000,
                        'gradient_steps': 40,
                        'batch_size': 256,
                        'gamma': 0.95,
                        'buffer_size': 1000000,
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256, 256, 256]}},
             },
})

register_experiment({
    'name': 'SQ_DRL_HER_sisq',
    'description': 'grasping',
    'tasks': [{'sub_name': 'icub_grasp_residual_1obj', 'env_id': _env_prefix+'iCubGraspResidualGoal-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.0,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'renders': False}
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
                        'batch_size': 256,
                        'gamma': 0.95,
                        'buffer_size': 1000000,
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256,256]}},
             },
})

register_experiment({
    'name': 'SQ_DRL_HER_sisq/pos_orn_constrained_pos_rew',
    'description': 'only reaching. reward is different: (0,1) instead of (-1,0). reward -10 with contact also under distance of 0.1. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.05)',
    'tasks': [{'sub_name': 'icub_reach_residual_1obj', 'env_id': _env_prefix+'iCubReachResidualGoal-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'renders': False}
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
                        'batch_size': 256,
                        'gamma': 0.95,
                        'buffer_size': 1000000,
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256,256]}},
             },
})

register_experiment({
    'name': 'SQ_DRL_HER_sisq/pos_orn_constrained_neg_rew',
    'description': 'only reaching. reward is (-1,0). reward -10 with contact also under distance of 0.1. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.05)',
    'tasks': [{'sub_name': 'icub_reach_residual_1obj', 'env_id': _env_prefix+'iCubReachResidualGoal-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'renders': False}
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
                        'batch_size': 256,
                        'gamma': 0.95,
                        'buffer_size': 1000000,
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256,256]}},
             },
})

register_experiment({
    'name': 'SQ_DRL_HER_sisq/residual_td3_pos_orn_constrained_neg_rew',
    'description': 'only reaching. action repeat lower (20). reward is (-1,10). reward -10 with contact also under distance of 0.1. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.08)',
    'tasks': [{'sub_name': 'icub_reach_residual_1obj', 'env_id': _env_prefix+'iCubReachResidualGoal-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 1000,
                            'renders': False}
              },
             ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'td3_residual',
                        'random_exploration': 0.3,
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'learning_starts': 300,
                        'policy': 'LnMlpPolicy',  # options: MlpPolicy e cnns ones
                        'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'n_timesteps': 500000,
                        'gradient_steps': 40,
                        'batch_size': 256,
                        'gamma': 0.95,
                        'buffer_size': 1000000,
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256,256]}},
             },
})

register_experiment({
    'name': 'SQ_DRL_HER_nosq/residual_td3_pos_orn_constrained_neg_rew',
    'description': 'only reaching. action repeat lower (20). reward is (-1,10). reward -10 with contact also under distance of 0.1. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.05)',
    'tasks': [{'sub_name': 'icub_reach_residual_1obj', 'env_id': _env_prefix+'iCubReachResidualGoal-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 0,
                            'max_steps': 1000,
                            'renders': False}
              },
             ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'td3_residual',
                        'random_exploration': 0.3,
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'learning_starts': 300,
                        'policy': 'LnMlpPolicy',  # options: MlpPolicy e cnns ones
                        'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'n_timesteps': 500000,
                        'gradient_steps': 40,
                        'batch_size': 256,
                        'gamma': 0.95,
                        'buffer_size': 1000000,
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256,256]}},
             },
})

register_experiment({
    'name': 'SQ_DRL_HER_sisq/residual_td3_pos_orn_constrained_pos_rew_summed',
    'description': 'only reaching. action repeat lower (20). reward is: -10 fall, -1 contact, 10 success, 0 otherwise. And all components are summed. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.08)',
    'tasks': [{'sub_name': 'icub_reach_residual_1obj', 'env_id': _env_prefix+'iCubReachResidualGoal-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
                            'renders': False}
              },
             ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'td3_residual',
                        'random_exploration': 0.3,
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'learning_starts': 300,
                        'policy': 'LnMlpPolicy',  # options: MlpPolicy e cnns ones
                        'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                        'n_timesteps': 500000,
                        'gradient_steps': 40,
                        'batch_size': 256,
                        'gamma': 0.95,
                        'buffer_size': 1000000,
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256,256]}},
             },
})

register_experiment({
    'name': 'SQ_DRL_HER_sisq_sac/residual_td3_pos_orn_constrained_pos_rew_summed',
    'description': 'only reaching. action repeat lower (20). reward is: -10 fall, -1 contact, 10 success, 0 otherwise. And all components are summed. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.08)',
    'tasks': [{'sub_name': 'icub_reach_residual_1obj', 'env_id': _env_prefix+'iCubReachResidualGoal-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
                            'renders': False}
              },
             ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'sac',
                        'n_timesteps': 500000,
                        'random_exploration':  0.31616809563805226,
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256]}},
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq_sac/pos_orn_constrained_pos_rew_summed',
    'description': 'only reaching.not goal env. action repeat lower (20). reward is: -10 fall, -1 contact, 10 success, 0 otherwise. And all components are summed. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.08)',
    'tasks': [{'sub_name': 'icub_grasp_notgoal_residual_1obj', 'env_id': _env_prefix+'iCubGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 500,
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
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256]}},
             },
})


register_experiment({
    'name': 'SQ_DRL_sisq/pos_orn_constrained_pos_rew_for_1sec_summed',
    'description': 'only reaching.not goal env. action repeat lower (20). reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise. And all components are summed. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.08)',
    'tasks': [{'sub_name': 'icub_grasp_notgoal_residual_1obj', 'env_id': _env_prefix+'iCubGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 800,
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
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256]},
                        'continue': True},
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq/pos_orn_constrained_pos_rew_for_1sec_summed',
    'description': 'only reaching.not goal env. action repeat lower (20). reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise. And all components are summed. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.08)',
    'tasks': [{'sub_name': 'icub_grasp_notgoal_residual_1obj', 'env_id': _env_prefix+'iCubGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 800,
                            'renders': False,
                            }
              },
             ],
    'algo': {'name': 'td3',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'n_timesteps': 500000,
                       'learning_starts': 1000,
                       'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                       'noise_type' : 'normal_0.5373', #options: normal, adaptive-param (can be multiple)
                       'n_timesteps': 500000,
                       'gradient_steps': 40,
                       'batch_size': 256,
                       'gamma': 0.99,
                       'buffer_size': 1000000,
                       'n_cpu_tf_sess':4,
                       'policy_kwargs': {'layers': [256,256]},
                       'continue': True},
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq/pos_orn_constrained_pos_rew_for_1sec_summed_+_vel_tipspos',
    'description': 'only reaching.not goal env. robot observation augmented with linear and angular velocities and pos/orn of all fingertips.action repeat lower (20). reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise. And all components are summed. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.08)',
    'tasks': [{'sub_name': 'icub_grasp_notgoal_residual_1obj', 'env_id': _env_prefix+'iCubGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 800,
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
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq/pos_orn_constrained_pos_rew_for_1sec_summed_+_vel',
    'description': 'only reaching.not goal env. robot observation augmented with linear and angular velocities and pos/orn of all fingertips.action repeat lower (20). reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise. And all components are summed. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.08)',
    'tasks': [{'sub_name': 'icub_grasp_notgoal_residual_1obj', 'env_id': _env_prefix+'iCubGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 800,
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
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'SQ_DRL_HER_sisq/pos_orn_constrained_pos_rew_for_1sec_summed_+_vel',
    'description': 'only reaching.different goal: on dist(hand_pos, gp_pos) and dist(current_obj_orn, init_obj_orn). robot observation augmented with linear and angular velocities. action repeat lower (20). reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise. And all components are summed. add check in success (if d<=threshold and object not fallen). also the position is more constrained (from 0.1 to 0.08)',
    'tasks': [{'sub_name': 'icub_reach_residual_1obj', 'env_id': _env_prefix+'iCubReachResidualGoal-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 800,
                            'renders': False}
              },
             ],
    'algo': {'name': 'her',
             'RLlibrary': 'stable_baselines_lib',
             'description': 'HER from stable_baselines library',
             'params': {'algo_name': 'sac',
                        'random_exploration': 0.3,
                        'goal_selection_strategy': 'future',
                        'n_sampled_goal': 4,
                        'n_timesteps': 500000,
                        'policy': 'MlpPolicy',  # options: MlpPolicy e cnns ones
                        'gamma': 0.99,
                        'learning_rate': 0.0007224206139165605,
                        'batch_size': 256,
                        'buffer_size': 10000,
                        'learning_starts': 1000,
                        'train_freq': 10,
                        'ent_coef': 0.1,
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq/pos_orn_constrained_pos_rew_for_1sec_summed_+_vel/icub_reach_notgoal_residual_3objs_NOISE_PCL',
    'description': 'only reaching.not goal env. robot observation augmented with linear and angular velocities and pos/orn of all fingertips.action repeat lower (20). reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'iCubReachResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'noise_pcl': 0.003,
                            'use_superq': 1,
                            'max_steps': 800,
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
                        'n_cpu_tf_sess':32,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'SQ_DRL_sisq/pos_orn_constrained_pos_rew_for_1sec_summed_+_vel/icub_reach_notgoal_residual_3objs_ROTATED',
    'description': 'only reaching.not goal env. 1 obj (mustard) rotated in range (pi/4, 3/4*pi). robot observation augmented with linear and angular velocities. action repeat lower (20). reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise. And all components are summed.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'iCubReachResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.07, # change only position
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 800,
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

register_experiment({
    'name': 'SQ_DRL_sisq/icub_grasp_lift_+reachingReward_only_stop_grasp_early_stop_lift',
    'description': 'grasping 1 object. action repeat 15 (+5.) counter step giusto. early stop for lift senza finger contact. reward for reaching gp: -10 fall, -1 contact, +2 reach gp, +1 for each fingertip contact, +20 lift',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'iCubGraspResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'max_steps': 800,
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

register_experiment({
    'name': 'SQ_DRL_sisq/icub_reach_forced_grasp_1obj',
    'description': 'only reaching.not goal env. 1 obj (mustard) rotated in range (pi/4, 3/4*pi). robot observation augmented with linear and angular velocities. action repeat lower (20). reward is: -10 fall, -1 contact, 10*1sec+100 success, 0 otherwise. And all components are summed.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'iCubReachResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0, # change only position
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 800,
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

register_experiment({
    'name': 'SQ_DRL_sisq/icub_reach_forced_grasp_3objs',
    'description': 'reaching and forced grasp lift. 3 objs (mustard), static. robot observation augmented with linear and angular velocities. reward is: -10 fall, -1 contact, 10 grasp pose reaching, +100 lifting with success, 0 otherwise.',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'iCubReachResidual-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0, # change only position
                            'noise_pcl': 0.00,
                            'use_superq': 1,
                            'max_steps': 800,
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

register_experiment({
    'name': 'icub_grasp',
    'description': 'grasping and lift of 1 object. not goal',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'iCubGrasp-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.0,
                            'max_steps': 1000,
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
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})

register_experiment({
    'name': 'icub_reach_3objs_rnd_pos_orn',
    'description': 'safe reaching of 3 objects. not goal',
    'tasks': [{'sub_name': '', 'env_id': _env_prefix+'iCubReach-v0', 'seed': 1,
               'env_params': {
                            'log_file': '',
                            'control_arm': 'r',
                            'control_orientation': 1,
                            'control_eu_or_quat': 0,
                            'obj_pose_rnd_std': 0.07,
                            'max_steps': 500,
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
                        'n_cpu_tf_sess':4,
                        'policy_kwargs': {'layers': [256,256]},
                        },
             },
})
#######

register_experiment({
    'name': 'pushing',
    'description': 'pushing environment, with low randomization of obj position and fixed target position',
    'tasks': [{'sub_name': 'obj_rnd_tg_fixed', 'env_id': _env_prefix+'iCubPushGoal-v0', 'seed': 1,
               'env_params': {
                            'use_IK': 1,
                            'discrete_action': 0,
                            'control_arm': 'l',
                            'control_orientation': 1,
                            'obj_pose_rnd_std': 0.02,
                            'tg_pose_rnd_std': 0.0,
                            'max_steps': 1000,
                            'renders': False}
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
                        'n_timesteps': 10000000,
                        'gradient_steps': 40,
                        'batch_size': 256,
                        'gamma': 0.95,
                        'buffer_size': 1000000,
                        'n_cpu_tf_sess':4,
                        'continue': True,
                        'policy_kwargs': {'layers': [256,256,256]}},
             },
})
