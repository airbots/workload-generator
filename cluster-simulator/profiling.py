# %%
import cProfile
from simulate import scale_workload
from utils import load_workload, load_config

data = load_workload('workload')
config = load_config()

def profile():
    """
    A function to profile the scale_workload function.

    Args:
    None.

    Returns:
    None.
    """
    scale_workload(data, config)

if __name__ == '__main__':
    cProfile.run('profile()')
