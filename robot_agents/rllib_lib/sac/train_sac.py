import os, inspect
import pprint as pp

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import ray.rllib

from ray.tune.experiment import convert_to_experiment_list
# from robot_agents.rllib_lib.common.common import train_callback


def train_SAC(env, eval_env, out_dir, seed=None, **kwargs):

    ray.init(
        local_mode=kwargs['local'],
        address=(kwargs['ray_address'] if 'ray_address' in kwargs else None),
        ignore_reinit_error=True,
        log_to_driver=False,
        webui_host="0.0.0.0",
    )

    # Get the experiments from the configuration file
    experiments = convert_to_experiment_list(kwargs)

    if len(experiments) == 0:
        raise ValueError("No experiments found")
    elif len(experiments) > 1:
        raise ValueError("Multiple experiments not yet supported")

    # Get the first experiment
    experiment = experiments[0]

    # TODO: define callbacks
    # Create the callback field if it does not exist
    if "callbacks" not in experiment.spec["config"]:
        experiment.spec["config"]["callbacks"] = {}

    callbacks = experiment.spec["config"]["callbacks"]

    checkpoint = None
    if 'checkpoint' in kwargs:
        checkpoint = kwargs['checkpoint']

    print(f"Running experiment:")
    pp.pprint(experiment.spec)

    trials = ray.tune.run(
        experiment,
        resume=kwargs['continue'] if 'continue' in kwargs else False,
        restore=checkpoint,
        return_trials=True,
    )


    return trials
