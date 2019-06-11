#add parent dir to find package.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
import tensorflow as tf
from utils import set_global_seed
from evaluation.exp_configuration import experiments
import robot_agents

import argparse

def main(exp_name, output_dir):

    if exp_name is None:
        raise ValueError("Please specify the experiment name. Run '$ experiment_wrapper -h' for info")

    exp = experiments.get_experiment(exp_name)

    for task in exp['tasks']:

        # Get Gym environment
        env = env = gym.make(task['env_id'], **task['env_params'])
        output_exp_dir = os.path.join(output_dir,exp_name, task['sub_name'])

        # Seed everything to make things reproducible.
        seed = task['seed']
        tf.reset_default_graph()
        set_global_seed(seed)
        env.seed(seed)

        # Should set a logger somehow
        #

        #run algorithm
        rl_library, algo_name, algo_params = exp['algo']['RLlibrary'], exp['algo']['name'],exp['algo']['params']

        model = robot_agents.ALGOS[rl_library][algo_name](env, output_exp_dir, seed, **algo_params)

        if not model is None:
            print("Saving model.pkl to ",output_exp_dir)
            model.save(os.path.join(output_exp_dir,"final_model.pkl"))

        del env
        del model


def parser_args():
    """
    parse the arguments for running the experiment
    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', action='store', dest='exp_name', type=str,
                        help='name of the experiment to run')

    parser.add_argument('--dir', type=str, action='store', dest='output_dir', default = '$HOME/robot_agents_log/',
                        help='directory where trained model, params and logs should be stored')

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parser_args()
    main(**args)
