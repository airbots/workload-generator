from task import Task
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import math
import logging

STEP_SIZE = 10

logging.getLogger().setLevel(logging.INFO)

def print_utilization(time, cpu_percent, mem_percent, net_percent, node_count, step=STEP_SIZE):
    """
    Prints the utilization of the cluster.

    Parameters:
    - time: The current time.
    - cpu_percent: The CPU utilization percentage.
    - mem_percent: The memory utilization percentage.
    - net_percent: The network utilization percentage.
    - node_count: The number of nodes.
    - step: The step size for the utilization bar (default is 10).
    """

    cpu_string = ":" * (cpu_percent//step) + " " * (step - cpu_percent//step)
    mem_string = ":" * (mem_percent//step) + " " * (step - mem_percent//step)
    net_string = ":" * (net_percent//step) + " " * (step - net_percent//step)

    print(f"{time} | CPU: {cpu_string} {cpu_percent}% | Memory: {mem_string} {mem_percent}% | Network: {net_string} {net_percent}% | Nodes: {node_count}")


def load_workload(fname):
    """
    This function loads a workload from a text file and returns a list of Task objects.

    Parameters:
    - fname: The name of the file (without extension) from which to load the workload.

    Returns:
    - A list of Task objects. Each Task object represents a single task in the workload.
      The attributes of a Task object are loaded from the corresponding values in the file.
      The file is expected to have six columns, which correspond to the six parameters of the Task constructor.
    """
    with open(f'data/{fname}.txt', 'r') as f:
        line = [line.strip() for line in f.readlines()]
        # remove first line because it is the header
        line.pop(0)
        data = []
        for i in range(len(line)):
            line[i] = line[i].split(',')
            data.append(Task(int(line[i][0]), int(line[i][1]), int(line[i][2]), int(line[i][3]), int(line[i][4]), int(line[i][5])))
    return data


def load_config():
    """
    Loads the configuration settings from a JSON file.

    The JSON file is expected to be located in the 'data' directory and named 'config.json'.

    Returns:
    - A dictionary containing the configuration settings. The keys are the setting names and the values are the setting values.
    """
    with open('data/config.json', 'r') as f:
        config = json.load(f)
    return config


def random_forest(data, node_lst):
    """
    This function trains a RandomForestRegressor model using the given data and node list. 
    It also performs grid search to find the best hyperparameters for the model.

    Parameters:
    - data: A list of Task objects. Each Task object represents a task with attributes like arrival_time, num_tasks, cpu, memory, network, and deadline.
    - node_lst: A list of node counts associated with each task in the data.

    Returns:
    - The best RandomForestRegressor model trained with the optimal hyperparameters found during grid search.
    """

    X = [[task.arrival_time, task.num_tasks, task.cpu, task.memory, task.network, task.deadline] for task in data]
    Y = node_lst
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, Y_train)

    param_grid = {
        'n_estimators': [25, 50, 100, 200, 300, 400, 500],
        'max_depth': [None, 1, 2, 3, 6, 10, 20, 30]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, Y_train)

    best_model = grid_search.best_estimator_

    # Make predictions
    Y_pred = best_model.predict(X_test)

    # Evaluate the best model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Mean Squared Error (MSE): {mse}")
    logging.info(f"R^2 Score: {r2}")

    return best_model


def get_min_nodes_lst(data, config):
    """
    This function calculates the minimum number of nodes required for each task in the given data.

    Parameters:
    - data: A list of Task objects. Each Task object represents a task with attributes like arrival_time, num_tasks, cpu, memory, network, and deadline.
    - config: A dictionary containing the configuration settings, such as 'Max_CPU_Utilization', 'Max_Memory_Utilization', and 'Max_Network_Utilization'.

    Returns:
    - A list of integers. Each integer represents the minimum number of nodes required for the corresponding task in the data.
    """
    lst = []
    for workload in data:
        lst.append(calc_min_nodes(workload, config))
    return lst


def calc_min_nodes(task, config):
    """
    Calculates the minimum number of nodes required to complete a given task within its deadline.

    Parameters:
    - task: A Task object representing the task to be completed. The Task object should have attributes like num_tasks, cpu, memory, network, and deadline.
    - config: A dictionary containing the configuration settings, such as 'Time_To_Complete_Task'.

    Returns:
    - An integer representing the minimum number of nodes required to complete the task within its deadline.
    """
    # load initial variables
    node_count = 1
    parallelization = get_num_task_per_node(task, config)
    execution_time = config['Time_To_Complete_Task']
    while (task.num_tasks * execution_time)/(node_count * parallelization) > task.deadline:
        node_count += 1
    return node_count


def get_num_task_per_node(task, config):
    """
    Calculates the maximum number of tasks that can be run on a single node based on the resource capacities and task requirements.

    Parameters:
    - task: A Task object representing the task to be completed. The Task object should have attributes like cpu, memory, and network which represent the resources required by the task.
    - config: A dictionary containing the configuration settings, such as 'Max_CPU_Utilization', 'Max_Memory_Utilization', and 'Max_Network_Utilization'.

    Returns:
    - An integer representing the maximum number of tasks that can be run on a single node.
    """
    cpu_cap = config['Max_CPU_Utilization']/task.cpu
    mem_cap = config['Max_Memory_Utilization']/task.memory
    net_cap = config['Max_Network_Utilization']/task.network
    return math.ceil(min(cpu_cap, mem_cap, net_cap))
