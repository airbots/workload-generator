# %%
from utils import load_workload, load_config
import math

data = load_workload('workload3')
config = load_config()


def scale_workload(data, config):
    # This function is still work in progress
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
            parallization = get_num_task_per_node(curr_task, config)
    pass
        

def get_min_nodes_lst(data, config):
    lst = []
    for workload in data:
        lst.append(calc_min_nodes(workload, config))
    return lst

# find the minimum number of nodes needed to complete the workload
def calc_min_nodes(task, config):
    # load initial variables
    node_count = 1
    parallelization = get_num_task_per_node(task, config)
    execution_time = config['Time_To_Complete_Task']
    while (task.num_tasks * execution_time)/(node_count * parallelization) > task.deadline:
        node_count += 1
    return node_count

def get_num_task_per_node(task, config):
    cpu_cap = config['Max_CPU_Utilization']/task.cpu
    mem_cap = config['Max_Memory_Utilization']/task.memory
    net_cap = config['Max_Network_Utilization']/task.network
    return math.ceil(min(cpu_cap, mem_cap, net_cap))
