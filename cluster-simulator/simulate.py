# %%
import json
from utils import load_workload, load_config
from cluster import Cluster

#load config json file
def load_config():
    with open('data/config.json', 'r') as f:
        config = json.load(f)
    return config

data = load_workload()
config = load_config()

cluster = Cluster()
# %%
def scale_workload(data, config):
    result = {}
    max_time = max([task.time for task in data])
    processed_task = 0
    rejected_task = 0

    for i in range(max_time):
        acceptace_rate = int((processed_task / processed_task + rejected_task) * 100)
        curr_task = []
        # check if it is time to bring in new task
        if i == data[0].arr_tme:
            
            print(f'Processing task: {data[0]} at time {i}')
            curr_task.append(data.pop(0))
            parallization = get_num_task_per_node(task, config)



    for task in data:
        parallization = get_num_task_per_node(task, config)
        time_to_finish = task.time / parallization
        
    


# %%
def get_num_task_per_node(task, config):
    cpu_cap = config['Max_CPU_Utilization']/task.cpu
    mem_cap = config['Max_Memory_Utilization']/task.memory
    net_cap = config['Max_Network_Utilization']/task.network
    return int(min(cpu_cap, mem_cap, net_cap))

print(get_num_task_per_node(data[0], config))
# %%
