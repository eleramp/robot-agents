import os
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def get_train_callback(eval_env, log_dir):
    checkpoint_callback = CheckpointCallback(save_freq=3000, save_path=log_dir)

    # Separate evaluation env
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(log_dir, 'best_model'),
                                 log_path=os.path.join(log_dir, 'evaluation_results'), eval_freq=500,
                                 deterministic=True, render=False)

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    return callback
