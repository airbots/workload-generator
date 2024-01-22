# %%
import json
from utils import load_workload, load_config

#load config json file
def load_config():
    with open('data/config.json', 'r') as f:
        config = json.load(f)
    return config

data = load_workload()
config = load_config()

print(data[0])
print(config)
# %%
def scale_workload(data, config, cluster):
    pass