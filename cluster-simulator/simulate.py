# %%
from utils import load_workload, load_config, get_num_task_per_node
import math
import logging

data = load_workload('workload1')
config = load_config()

logging.getLogger().setLevel(logging.INFO)


# This function is used to simulate the cluster scaling process
def scale_workload(data, config):
    result = {}
    data_copy = data.copy()

    # get the maximum time needed to complete the entire workload
    max_time = max([task.arrival_time + task.deadline for task in data])
    logging.info(f'The maximum time needed to complete the entire workload is: {max_time}')
    
    processed_task = 0
    rejected_task = 0

    # Initialize the cluster, where each node is empty for every second of the timeline
    default_cluster = [0] * max_time
    cluster_state = [default_cluster for _ in range(config['Initial_Node_Count'])]

    # load the constant time to complete a task
    execution_time = config['Time_To_Complete_Task']

    # Start the simulation, where each iteration represents one second
    for i in range(max_time):
        task_queue = []
        # check if it is time to bring in new task
        if len(data) == 0:
            break
        if i == data[0].arrival_time:
            logging.info(f'----- Processing task: {data[0]} at time {i} -----')
            task_queue.append(data.pop(0))
            parallization = get_num_task_per_node(task_queue[0], config)
            if task_can_be_processed(task_queue[0], cluster_state, parallization, execution_time):
                current_task = task_queue.pop(0)
                temp_cluster_state = process_task(current_task, cluster_state, parallization, execution_time)
                processed_task += 1
                cluster_state = temp_cluster_state
                logging.info(f'The task {current_task} is processed, the cluster state after processing the task is: {[node.count(1) for node in cluster_state]}')
                logging.info(f'The cluster state after processing the task is: {[node.count(1) for node in cluster_state]}')
            else:
                logging.info(f'The task cannot be processed, rejecting the task')
                rejected_task += 1
        else:
            logging.info(f'No task is coming in at time {i}')
    
    # Calculate the acceptance rate
    if processed_task + rejected_task == 0:
        acceptace_rate = 0
    else:
        acceptace_rate = int((processed_task / (processed_task + rejected_task)) * 100)

    completion_time = max(i for sublist in cluster_state for i, x in enumerate(sublist) if x == 1)
    utilization = int(sum([node.count(1) for node in cluster_state]) / (len(cluster_state) * completion_time)*100)

    result['Acceptance_Rate'] = acceptace_rate
    result['Overall_Utilization'] = utilization
    result['Task_Completion_Time'] = completion_time
    result['No_of_Completed_Jobs'] = processed_task
    result['No_of_Rejected_Jobs'] = rejected_task
    result['Node_Count'] = len(cluster_state)
    result['Total_Num_Tasks'] = sum([task.num_tasks for task in data_copy])
        
    return result


# This function is a wrapper of the origimal scale_workload function with custom node count
def scale_workload_dynamic(data, config, node_lst):
    result = {}
    data_copy = data.copy()

    # get the maximum time needed to complete the entire workload
    max_time = max([task.arrival_time + task.deadline for task in data])
    logging.info(f'The maximum time needed to complete the entire workload is: {max_time}')
    
    processed_task = 0
    rejected_task = 0

    # Initialize the cluster, where each node is empty for every second of the timeline
    default_cluster = [0] * max_time
    cluster_state = [default_cluster for _ in range(node_lst[0])]
    # load the constant time to complete a task
    execution_time = config['Time_To_Complete_Task']

    # Start the simulation, where each iteration represents one second
    for i in range(max_time):
        task_queue = []
        node_queue = []
        # check if it is time to bring in new task
        if len(data) == 0:
            break
        if i == data[0].arrival_time:
            logging.info(f'----- Processing task: {data[0]} at time {i} -----')
            task_queue.append(data.pop(0))
            node_queue.append(node_lst.pop(0))
            parallization = get_num_task_per_node(task_queue[0], config)
            cluster_state = scale_workload(cluster_state, node_queue[0], i)
            if task_can_be_processed(task_queue[0], cluster_state, parallization, execution_time):
                current_task = task_queue.pop(0)
                temp_cluster_state = process_task(current_task, cluster_state, parallization, execution_time)
                processed_task += 1
                cluster_state = temp_cluster_state
                logging.info(f'The task {current_task} is processed, the cluster state after processing the task is: {[node.count(1) for node in cluster_state]}')
                logging.info(f'The cluster state after processing the task is: {[node.count(1) for node in cluster_state]}')
            else:
                logging.info(f'The task cannot be processed, rejecting the task')
                rejected_task += 1
        else:
            logging.info(f'No task is coming in at time {i}')
    
    # Calculate the acceptance rate
    if processed_task + rejected_task == 0:
        acceptace_rate = 0
    else:
        acceptace_rate = int((processed_task / (processed_task + rejected_task)) * 100)

    completion_time = max(i for sublist in cluster_state for i, x in enumerate(sublist) if x == 1)
    utilization = int(sum([node.count(1) for node in cluster_state]) / (len(cluster_state) * completion_time)*100)

    result['Acceptance_Rate'] = acceptace_rate
    result['Overall_Utilization'] = utilization
    result['Task_Completion_Time'] = completion_time
    result['No_of_Completed_Jobs'] = processed_task
    result['No_of_Rejected_Jobs'] = rejected_task
    result['Node_Count'] = len(cluster_state)
    result['Total_Num_Tasks'] = sum([task.num_tasks for task in data_copy])
        
    return result


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
    available_space = sum([node[task.arrival_time:(task.arrival_time + task.deadline)].count(0) for node in cluster_state])
    required_space = math.ceil(task.num_tasks/parallelization) * execution_time
    logging.info(f'The available space is: {available_space} slots, the required space is: {required_space} slots')
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
    logging.info(f'The required space is: {required_space} slots')

    # Use round robin to schedule the task
    current_node_idx = 0
    logging.debug(f'The number of nodes is: {num_nodes}')
    while required_space > 0:
        logging.debug(f'Processing node {current_node_idx}, the current node state is: {cluster_state[current_node_idx]}')
        node = cluster_state[current_node_idx].copy()
        try:
            first_nonzero_index_after_timestamp = node[task.arrival_time:].index(0) + task.arrival_time
        except ValueError:
            logging.error(f"No available slots in node {current_node_idx} after timestamp {task.arrival_time}")
            break
        upper_bound = math.ceil(required_space/num_nodes) + first_nonzero_index_after_timestamp
        if upper_bound > len(node): 
            upper_bound = len(node)
        logging.debug(f'The upper bound is: {upper_bound}')
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


def scale_workload(cluster_state, target_node_count, timestamp):
    """
    This is a wrapper of Scales the cluster to process the workload.

    Parameters:
    - cluster_state: A list of lists representing the current state of the cluster.
    - target_node_count: The target number of nodes in the cluster after scaling.
    - timestamp: The current timestamp.

    Returns:
    - The updated state of the cluster after scaling.
    """

    if target_node_count > len(cluster_state):
        return scale_up(cluster_state, target_node_count, timestamp)
    elif target_node_count < len(cluster_state):
        return scale_down(cluster_state, target_node_count, timestamp)
    else:
        logging.info(f'No scaling is needed')
        return cluster_state


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


def calc_min_nodes(task, cluster_state, execution_time, parallelization):
    """
    Calculates the minimum number of nodes required to process a task.

    Parameters:
    - task: The task to be processed.
    - cluster_state: The current state of the cluster nodes.
    - execution_time: The time required to execute the task.
    - parallelization: The degree of parallelization for the task execution.

    Returns:
    - The minimum number of nodes required to process the task.
    """

    if len(cluster_state) == 1:
        logging.info(f'No scaling is needed')
        return len(cluster_state)
    
    # Calculate the total number of tasks
    total_tasks = task.num_tasks * execution_time / parallelization
    logging.info(f'The total number of tasks is: {total_tasks}')

    # Calculate the available space in the cluster nodes
    available_space = sum([node[task.arrival_time:task.arrival_time+task.deadline].count(0) 
                           for node in cluster_state])

    # Calculate the difference between the total number of tasks and the available space
    diff = total_tasks - available_space

    # Resource provided by one node
    resource = task.deadline

    # Calculate the node needed or can be removed
    count = (diff / resource)

    # no scaling is needed
    if count == 0:
        logging.info(f'No scaling is needed')
        return len(cluster_state)
    # scaling up is needed
    elif count > 0:
        logging.info(f'Scaling up is needed')
        return len(cluster_state) + math.ceil(count)
    # scaling down is needed
    else:
        logging.info(f'Scaling down is needed')
        return len(cluster_state) + math.floor(resource / diff)
    

'''

This section is for testing

'''
min_lst = [1, 1, 1, 1, 1]
node_lst = [5, 3, 3, 5, 3]
max_lst = [9, 9, 9, 9, 9]
second_algo_lst = [1, 1, 9, 1, 1]

if __name__ == "__main__":
    print(scale_workload_dynamic(data, config, second_algo_lst))


# %%
