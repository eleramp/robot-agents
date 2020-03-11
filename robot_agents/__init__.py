import robot_agents
from robot_agents import stable_baselines_lib
from robot_agents import baselines_lib
from robot_agents.utils import linear_schedule
from robot_agents.stable_baselines_lib.sac.sac_residual import SAC_residual

ALGOS = {
    'stable_baselines_lib': {'ddpg': robot_agents.stable_baselines_lib.train_DDPG,
                             'td3': robot_agents.stable_baselines_lib.train_TD3,
                             'td3_residual': robot_agents.stable_baselines_lib.train_TD3_residual,
                             'deepq': robot_agents.stable_baselines_lib.train_DQN,
                             'sac': robot_agents.stable_baselines_lib.train_SAC,
                             'sac_residual': robot_agents.stable_baselines_lib.train_SAC_residual,
                             'her': robot_agents.stable_baselines_lib.train_HER,
                             'ddpg_test': robot_agents.stable_baselines_lib.test_DDPG,
                             'deepq_test': robot_agents.stable_baselines_lib.test_DQN,
                             'sac_residual_test': robot_agents.stable_baselines_lib.test_SAC_residual,
                             'sac_test': robot_agents.stable_baselines_lib.test_SAC,
                             'td3_test': robot_agents.stable_baselines_lib.test_TD3,
                             'her_test': robot_agents.stable_baselines_lib.test_HER,
                             },
    'baselines_lib': {'deepq': robot_agents.baselines_lib.train_DQN,
                     },
}
