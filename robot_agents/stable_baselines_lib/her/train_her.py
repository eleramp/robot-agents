#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np

from stable_baselines import HER, DQN, SAC, DDPG, TD3
from robot_agents.stable_baselines_lib.sac.sac_residual import SAC_residual
from robot_agents.stable_baselines_lib.td3.td3_residual import TD3_residual

from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.ppo2.ppo2 import constfn
from robot_agents.utils import linear_schedule

best_mean_reward, n_steps = -np.inf, 0
def log_callback(_locals, _globals):

    global n_steps, best_mean_reward, log_dir
    # Print stats every 1000 calls
    if n_steps % 3000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
            print("Saving new model")
            _locals['self'].save(os.path.join(output_dir, str(n_steps)+'_model_r_'+str(best_mean_reward)+'.pkl'))
    n_steps += 1
    return True

def set_agent(algo_name):
    agent = None
    agent = DQN if algo_name is 'deepq' else agent
    agent = DDPG if algo_name is 'ddpg' else agent
    agent = TD3 if algo_name is 'td3' else agent
    agent = SAC if algo_name is 'sac' else agent
    agent = TD3_residual if algo_name is 'td3_residual' else agent
    agent = SAC_residual if algo_name is 'sac_residual' else agent
    return agent

def train_HER(env, out_dir, seed=None, **kwargs):
    # Logs will be saved in log_dir/monitor.csv
    global output_dir,log_dir
    output_dir = out_dir
    log_dir = os.path.join(out_dir,'log')
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir+'/', allow_early_resets=True)

    policy = kwargs['policy']
    algo_name = kwargs['algo_name']
    n_timesteps = kwargs['n_timesteps']
    noise_type = None
    if 'noise_type' in kwargs:
        noise_type = kwargs['noise_type']
        del kwargs['noise_type']

    # HER Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = kwargs['goal_selection_strategy']
    n_sampled_goal = kwargs['n_sampled_goal']

    del kwargs['policy']
    del kwargs['algo_name']
    del kwargs['n_timesteps']
    del kwargs['goal_selection_strategy']
    del kwargs['n_sampled_goal']

    # Set agent algorithm
    agent = set_agent(algo_name)
    if not agent:
        print("invalid algorithm for HER")
        return

    # the noise objects
    nb_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = None

    if noise_type:

        for current_noise_type in noise_type.split(','):

            current_noise_type = current_noise_type.strip()

            if 'adaptive-param' in current_noise_type and algo_name is 'ddpg':
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))

            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))

            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(nb_actions),
                sigma=float(stddev) * np.ones(nb_actions))

            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Create learning rate schedule
    for key in ['learning_rate', 'learning_rate_pi', 'cliprange']:
        if key in kwargs:
            if isinstance(kwargs[key], str):
                schedule, initial_value = kwargs[key].split('_')
                initial_value = float(initial_value)
                kwargs[key] = linear_schedule(initial_value)
            elif isinstance(kwargs[key], float):
                kwargs[key] = constfn(kwargs[key])
            else:
                raise ValueError('Invalid valid for {}: {}'.format(key, kwargs[key]))

    kwargs['tensorboard_log'] = os.path.join(log_dir, 'tb')
    kwargs['full_tensorboard_log'] = False
    kwargs['seed'] = seed
    kwargs['action_noise'] = action_noise
    if algo_name is 'ddpg':
        kwargs['param_noise'] = param_noise

    if 'continue' in kwargs and kwargs['continue'] is True:
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del kwargs['policy']
        model = HER.load(os.path.join(out_dir, 'final_model.pkl'), env=env, verbose=1, **kwargs)
    else:
        if 'continue' in kwargs:
            del kwargs['continue']
        model = HER(policy, env, agent,
                    goal_selection_strategy=goal_selection_strategy, n_sampled_goal=n_sampled_goal, verbose=1, **kwargs)

    model.learn(total_timesteps=n_timesteps, callback=log_callback)

    return model