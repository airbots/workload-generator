# %%
from utils import load_workload, load_config, get_min_nodes_lst, get_num_task_per_node
import math

data = load_workload('workload3')
config = load_config()

# %%
# This function is used to simulate the cluster scaling process
def scale_workload(data, config):
    result = []

    # get the maximum time needed to complete the entire workload
    max_time = [task.arrival_time + task.deadline for task in data]
    
    processed_task = 0
    rejected_task = 0

    # Initialize the cluster, where each node is empty for every second of the timeline
    default_cluster = [0] * 500
    cluster_state = [default_cluster for _ in range(config['Initial_Node_Count'])]

    # load the constant time to complete a task
    execution_time = config['Time_To_Complete_Task']

    # Start the simulation, where each iteration represents one second
    for i in range(max_time):
        # Calculate the acceptance rate
        if processed_task + rejected_task == 0:
            acceptace_rate = 0
        else:
            acceptace_rate = int((processed_task / processed_task + rejected_task) * 100)
        
        task_queue = []
        # check if it is time to bring in new task
        if i == data[0].arrival_time:
            print(f'Processing task: {data[0]} at time {i}')
            task_queue.append(data.pop(0))
            parallization = get_num_task_per_node(task_queue, config)
    pass
        
def task_can_be_processed(task, cluster_state, parallelization, execution_time):
    # check if there is enough nodes to process the task
    available_space = sum([node[task.arrival_time:task.deadline].count(0) for node in cluster_state])
    required_space = math.ceil(task.num_tasks/parallelization) * execution_time
    return available_space >= required_space

# this is the task schedule function, which marks occupied space in the cluster as 1
def process_task(task, cluster_state, parallelization, execution_time):
    required_space = math.ceil((task.num_tasks/parallelization) * execution_time)
    print(f'The required space is: {required_space} slots')
    
    # Use round robin to schedule the task
    node_idx = 0
    num_nodes = len(cluster_state)
    while required_space > 0:
        print(f'Processing node {node_idx}', 'the current node state is: ', cluster_state[node_idx])
        node = cluster_state[node_idx].copy()
        for i in range(task.arrival_time, task.arrival_time + task.deadline):
            if node[i] == 0:
                node[i] = 1
                required_space -= 1
                if required_space == 0:
                    break
        cluster_state[node_idx] = node
        node_idx = (node_idx + 1) % num_nodes
        if required_space == 0:
            return cluster_state
    return cluster_state

# %%
'''

This section is for testing

'''
def task_can_be_processed(task, cluster_state, parallelization, execution_time): 
    # check if there is enough nodes to process the task
    available_space = sum([node[task.arrival_time:(task.arrival_time+ task.deadline)].count(0) for node in cluster_state])
    required_space = math.ceil((task.num_tasks/parallelization) * execution_time)
    return available_space >= required_space

def process_task(task, cluster_state, parallelization, execution_time):
    required_space = math.ceil((task.num_tasks/parallelization) * execution_time)
    print(f'The required space is: {required_space} slots')
    
    # Use round robin to schedule the task
    node_idx = 0
    num_nodes = len(cluster_state)
    while required_space > 0:
        print(f'Processing node {node_idx}', 'the current node state is: ', cluster_state[node_idx])
        node = cluster_state[node_idx].copy()
        for i in range(task.arrival_time, task.arrival_time + task.deadline):
            if node[i] == 0:
                node[i] = 1
                required_space -= 1
                if required_space == 0:
                    break
        cluster_state[node_idx] = node
        node_idx = (node_idx + 1) % num_nodes
        if required_space == 0:
            return cluster_state
    return cluster_state



tast_task = data[0]
default_cluster = [0] * 500
tast_cluster_state = [default_cluster for _ in range(2)]

print(task_can_be_processed(tast_task, tast_cluster_state, 2, 5))
#print(process_task(tast_task, tast_cluster_state, 2, 5))

output = process_task(tast_task, tast_cluster_state, 2, 5)
print([node.count(1) for node in output])
# %%
