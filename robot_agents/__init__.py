import robot_agents
from robot_agents import stable_baselines_agents
from robot_agents import openai_baselines
from robot_agents.utils import linear_schedule

ALGOS = {
    'stable_baselines_agents': {'ddpg': robot_agents.stable_baselines_agents.train_DDPG,
                                'deepq': robot_agents.stable_baselines_agents.train_DQN,
                                'ddpg_test': robot_agents.stable_baselines_agents.test_DDPG,
                                'deepq_test': robot_agents.stable_baselines_agents.test_DQN,
                                },
    'openai_baselines': {'deepq': robot_agents.openai_baselines.train_DQN,
                        },
}
