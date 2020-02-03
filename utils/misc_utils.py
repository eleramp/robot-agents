import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed):
    """
    set the seed for python random, numpy and tensorflow
    :param seed: (int) the seed
    """
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
