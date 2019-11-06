import numpy as np

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


best_mean_reward, n_steps = -np.inf, 0
def log_callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if 'num_episodes' in _locals:
        if _locals['num_episodes'] % 10 == 0:
            # Evaluate policy training performance
            if len(_locals['episode_rewards'][-101:-1]) == 0:
                mean_reward = -np.inf
            else:
                mean_reward = round(float(np.mean(_locals['episode_rewards'][-101:-1])), 1)

            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward or _locals['num_episodes'] % 1000 == 0:
                best_mean_reward = mean_reward
                # Save model
                print("Saving model at iter {}".format(_locals['step']))
                locals['self'].save(os.path.join(output_dir, str(_locals['step'])+'model' +
                                                 'rew_' + str(best_mean_reward)+'.pkl'))
    n_steps += 1
    # Returning False will stop training early
    return True
