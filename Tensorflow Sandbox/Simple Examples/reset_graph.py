import tensorflow.compat.v1 as reset
import numpy.random as np

def reset_graph(seed=42):
    reset.reset_default_graph()
    reset.set_random_seed(seed)
    np.seed(seed)