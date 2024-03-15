import contextlib

import numpy as np


@contextlib.contextmanager
def numpy_seed(seed):
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)