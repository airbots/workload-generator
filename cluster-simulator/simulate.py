# %%
from utils import load_workload, load_config, get_min_nodes_lst, get_num_task_per_node
import math
import logging

data = load_workload('workload3')
config = load_config()

logging.getLogger().setLevel(logging.INFO)
# %%
# This function is used to simulate the cluster scaling process
def scale_workload(data, config):
    result = []

    # get the maximum time needed to complete the entire workload
    max_time = [task.arrival_time + task.deadline for task in data]
    
    processed_task = 0
    rejected_task = 0

    # Initialize the cluster, where each node is empty for every second of the timeline
    default_cluster = [0] * max_time
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
    """
    Checks if a task can be processed by the current cluster state considering the available space, parallelization level, and execution time.

    Parameters:
    - task: A Task object representing the task to be processed. The Task object should have attributes like num_tasks, arrival_time, and deadline.
    - cluster_state: A list of lists representing the current state of the cluster. Each inner list represents a node and its elements represent the state of each slot in the node.
    - parallelization: The level of parallelization for the task.
    - execution_time: The execution time for the task.

    Returns:
    - A boolean value indicating whether the task can be processed. Returns True if there is enough space to process the task, and False otherwise.
    """
    # check if there is enough nodes to process the task
    available_space = sum([node[task.arrival_time:task.deadline].count(0) for node in cluster_state])
    required_space = math.ceil(task.num_tasks/parallelization) * execution_time
    return available_space >= required_space

def process_task(task, cluster_state, parallelization, execution_time):
    """
    This function processes a task using a round-robin scheduling algorithm.

    Parameters:
    task: The task to be processed.
    cluster_state: The current state of the cluster.
    parallelization: The level of parallelization for the task.
    execution_time: The execution time for the task.

    Returns:
    The updated state of the cluster after processing the task.
    """

    num_nodes = len(cluster_state)
    required_space = math.ceil((task.num_tasks/parallelization) * execution_time)
    upper_bound = math.ceil(required_space/num_nodes) + task.arrival_time

    logging.debug(f'The upper bound is: {upper_bound}')
    logging.info(f'The required space is: {required_space} slots')

    # Use round robin to schedule the task
    current_node_idx = 0
    logging.debug(f'The number of nodes is: {num_nodes}')
    while required_space > 0:
        logging.debug(f'Processing node {current_node_idx}, the current node state is: {cluster_state[current_node_idx]}')
        node = cluster_state[current_node_idx].copy()
        for i in range(task.arrival_time, upper_bound):
            if node[i] == 0:
                node[i] = 1
                required_space -= 1
                if required_space == 0:
                    break
        cluster_state[current_node_idx] = node
        logging.debug(f'The node state after assignment is: {node}')
        current_node_idx = (current_node_idx + 1) % num_nodes
 
    logging.info(f'The cluster task count after assignment is: {[node.count(1) for node in cluster_state]}')
    return cluster_state

def scale_down(cluster_state, target_node_count, timestamp):
    """
    Scales down the cluster by removing nodes and reassigning their tasks.

    Parameters:
    - cluster_state: A list of lists representing the current state of the cluster.
    - target_node_count: The target number of nodes in the cluster after scaling down.
    - timestamp: The current timestamp.

    Returns:
    - The updated state of the cluster after scaling down and reassigning tasks.
    """
    # concentrate all workload to the nodes that are not going to be removed
    # calculate the total number of tasks scheduled in to be removed nodes after the timestamp
    logging.info(f'********** Scaling down the cluster from {len(cluster_state)} to {target_node_count} nodes **********')
    nodes_to_be_removed = cluster_state[target_node_count:]
    total_tasks = sum([node[timestamp:].count(1) for node in nodes_to_be_removed])
    logging.info(f'The total number of tasks to be reassigned is: {total_tasks}')
    # calculate the number of tasks that should be assigned to each node
    remaining_nodes = cluster_state[:target_node_count]

    return reassign_task(remaining_nodes, total_tasks, timestamp)


def scale_up(cluster_state, target_node_count, timestamp):
    """
    Scales up the cluster by adding new nodes.

    Parameters:
    - cluster_state: A list of lists representing the current state of the cluster. Each inner list represents a node and its elements represent the state of each slot in the node.
    - target_node_count: The target number of nodes in the cluster after scaling up.

    Returns:
    - A new list of lists representing the state of the cluster after scaling up. The new nodes are added at the beginning of the list and are initialized with all slots being 0.
    """
    # add new nodes to the cluster
    logging.info(f'********** Scaling up the cluster from {len(cluster_state)} to {target_node_count} nodes **********')
    node_diff = target_node_count - len(cluster_state)
    new_cluster = [[0] * len(cluster_state[0]) for _ in range(node_diff)] + cluster_state
    total_tasks_to_assign = sum([node[timestamp:].count(1) for node in new_cluster])
    logging.info(f'The total number of tasks to be reassigned is: {total_tasks_to_assign}')
    return reassign_task(new_cluster, total_tasks_to_assign, timestamp)


def reassign_task(cluster_state, total_tasks_to_assign, timestamp):
    """
    Reassigns tasks to nodes in the cluster.

    Parameters:
    - cluster_state: A list of lists representing the current state of the cluster.
    - total_tasks_to_assign: The total number of tasks to be reassigned.
    - timestamp: The current timestamp.

    Returns:
    - The updated state of the cluster after reassigning the tasks.
    """
    tasks_per_node = math.ceil(total_tasks_to_assign / len(cluster_state))
    updated_cluster_state = cluster_state
    logging.info(f'There are {total_tasks_to_assign} tasks unfinished and to be assigned to {len(cluster_state)} nodes, each node will be assigned {tasks_per_node} tasks')
    current_node_idx = 0
    while total_tasks_to_assign > 0:
        node = updated_cluster_state[current_node_idx].copy()
        try:
            first_nonzero_index_after_timestamp = node[timestamp:].index(0) + timestamp
        except ValueError:
            logging.error(f"No available slots in node {current_node_idx} after timestamp {timestamp}")
            break
        logging.debug(f'Processing node {current_node_idx}, the current node state is: {cluster_state[current_node_idx]}')
        for i in range(timestamp, first_nonzero_index_after_timestamp + tasks_per_node):
            if node[i] == 0:
                node[i] = 1
                total_tasks_to_assign -= 1
                if total_tasks_to_assign == 0:
                    break
        updated_cluster_state[current_node_idx] = node
        logging.debug(f'The node state after assignment is: {node}')
        current_node_idx = (current_node_idx + 1) % len(updated_cluster_state)
    logging.info(f'The new cluster state after scaling is: {[node[timestamp:].count(1) for node in cluster_state]}')
    return updated_cluster_state


# %%
'''

This section is for testing

'''

from utils import load_workload, load_config, get_min_nodes_lst, get_num_task_per_node
import math
import logging

data = load_workload('workload3')
config = load_config()

logging.getLogger().setLevel(logging.INFO)

max_time = 30

default_cluster = [0] * max_time
cluster_state = [default_cluster for _ in range(5)]
cluster_state = process_task(data[0], cluster_state, 1, 2)

logging.info(f'The cluster state is: {cluster_state}')
# %%

