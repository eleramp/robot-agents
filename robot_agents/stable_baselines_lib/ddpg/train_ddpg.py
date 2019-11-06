import gym

#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


# import RL agent
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.bench import Monitor
from robot_agents.utils import log_callback

# Fix for breaking change for DDPG buffer in v2.6.0
#if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
#    os.sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
#    stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

#
import numpy as np
global output_dir

def train_DDPG( env, out_dir, seed=None, **kwargs):

    # Logs will be saved in log_dir/monitor.csv
    global output_dir
    output_dir = out_dir
    log_dir = os.path.join(out_dir,'log')
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir+'/', allow_early_resets=True)

    policy = kwargs['policy']
    n_timesteps = kwargs['n_timesteps']
    noise_type = kwargs['noise_type']
    del kwargs['policy']
    del kwargs['n_timesteps']
    del kwargs['noise_type']

    ''' Parameter space noise:
    injects randomness directly into the parameters of the agent, altering the types of decisions it makes
    such that they always fully depend on what the agent currently senses. '''

    # the noise objects for DDPG
    nb_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = None
    if not noise_type is None:

        for current_noise_type in noise_type.split(','):

            current_noise_type = current_noise_type.strip()

            if 'adaptive-param' in current_noise_type:
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


    if 'continue' in kwargs and kwargs['continue'] is True:
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del kwargs['policy']
        model = DDPG.load(os.path.join(out_dir,'final_model.pkl'), env=env,
                         tensorboard_log=os.path.join(log_dir,'tb'), verbose=1, **kwargs)
    else:
        if 'continue' in kwargs:
            del kwargs['continue']
        model = DDPG(policy, env, param_noise=param_noise, action_noise=action_noise, seed=seed,
                verbose=1, tensorboard_log=os.path.join(log_dir,'tb'),full_tensorboard_log=False, **kwargs)

    model.learn(total_timesteps=n_timesteps, callback=log_callback)

    return model
