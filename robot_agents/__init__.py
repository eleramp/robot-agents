import robot_agents
from robot_agents import stable_baselines_lib
from robot_agents import baselines_lib
from robot_agents.utils import linear_schedule
from robot_agents.stable_baselines_lib.sac.sac_residual import SAC_residual

ALGOS = {
    'stable_baselines_lib': {'ddpg': robot_agents.stable_baselines_lib.train_DDPG,
                             'deepq': robot_agents.stable_baselines_lib.train_DQN,
                             'residual_sac': robot_agents.stable_baselines_lib.train_SAC,
                             'her': robot_agents.stable_baselines_lib.train_HER,
                             'ddpg_test': robot_agents.stable_baselines_lib.test_DDPG,
                             'deepq_test': robot_agents.stable_baselines_lib.test_DQN,
                             'residual_sac_test': robot_agents.stable_baselines_lib.test_SAC,
                             },
    'baselines_lib': {'deepq': robot_agents.baselines_lib.train_DQN,
                     },
}

