#add parent dir to find package.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import tensorflow as tf
import numpy as np
from utils import set_global_seed
from evaluation import experiment_registration

import robot_agents

import argparse
import csv


def main(exp_name, output_dir, do_train, do_test, n_seeds, seed_val):

    if exp_name is None:
        raise ValueError("Please specify the experiment name. Run '$ experiment_wrapper -h' for info")
    if not (do_train or do_test):
        raise ValueError("Please specify if you want to do training or testing. Run '$ experiment_wrapper -h' for info")

    exp = experiment_registration.get_experiment(exp_name)

    for task in exp['tasks']:
        # decide seed
        if seed_val is not None and n_seeds > 1:
            raise ValueError("You cannot both provide a specific seed value {} and require n_seeds={} random values".format(seed_val, n_seeds))

        # override seed value with the one provided as arg
        if seed_val is not None:
            task['seed'] = seed_val

        if n_seeds > 1 or 'seed' not in task:
            np.random.seed(2)
            seeds = np.random.randint(0, 20000, size=n_seeds)
        else:
            seeds = np.array([task['seed']])

        # a different training for each seed
        for ns in range(n_seeds):

            seed = int(seeds[ns])

            # Seed everything to make things reproducible.
            tf.compat.v1.reset_default_graph()
            set_global_seed(seed)

            # Read experiment conf variables
            rl_library, algo_name, algo_params = exp['algo']['RLlibrary'], exp['algo']['name'], exp['algo']['params']

            # Set path for outputdata
            output_exp_dir = os.path.join(output_dir, exp_name, task['sub_name'], 'seed_' + str(seed))
            os.makedirs(output_exp_dir, exist_ok=True)

            # Set Gym environment
            renders = True if do_test else False

            task['env_params']['renders'] = renders
            if 'log_file' in task['env_params']:
                task['env_params']['log_file'] = output_exp_dir

            # Create environment as normalized vectorized environment
            with_vecnorm = False
            env, eval_env = robot_agents.ALGOS[rl_library]['make_env'](task['env_id'], task['env_params'], seed, do_train, with_vecnorm)

            # Run algorithm
            csv_file = os.path.join(output_exp_dir, "exp_param.csv")
            try:
                with open(csv_file, 'w') as f:
                    for key in exp.keys():
                        f.write("%s,%s\n"%(key,exp[key]))
            except IOError:
                print("I/O error")

            if do_train:
                model = robot_agents.ALGOS[rl_library][algo_name](env, eval_env, output_exp_dir, seed, **algo_params)

                if not model is None:
                    print("Saving model to ", output_exp_dir)
                    model.save(os.path.join(output_exp_dir, "final_model"))

            elif do_test:
                algo_name = algo_name+'_test'
                model = robot_agents.ALGOS[rl_library][algo_name](env, output_exp_dir, seed, **algo_params)

            del env
            del eval_env
            del model


def parser_args():
    """
    parse the arguments for running the experiment
    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', action='store', dest='exp_name', type=str,
                        help='name of the experiment to run')

    parser.add_argument('--train', action='store_const', const=1, dest="do_train",
                        help='do training')

    parser.add_argument('--test', action='store_const', const=1, dest="do_test",
                        help='do testing')

    parser.add_argument('--n_seeds', action='store', dest="n_seeds", type=int, default=1,
                        help='number of seeds to use for training')

    parser.add_argument('--seed', action='store', dest="seed_val", type=int, default=None,
                        help='seed value')

    parser.add_argument('--dir', type=str, action='store', dest='output_dir', default = '~/robot_agents_log/',
                        help='directory where trained model, params and logs should be stored')

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parser_args()
    print('args')
    print(args)
    main(**args)
