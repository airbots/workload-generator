# %%
import cProfile
from simulate import scale_workload
from utils import load_workload, load_config

data = load_workload('workload')
config = load_config()

def profile():
    scale_workload(data, config)

cProfile.run('profile()')
