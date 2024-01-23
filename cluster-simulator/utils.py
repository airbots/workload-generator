from task import Task
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# This helper function is used to print the utilization of the cluster
def print_utilization(time, cpu_percent, mem_percent, net_percent, node_count, step=10):
    cpu_string = ":" * int(cpu_percent//step) + " " * int(step - (cpu_percent)//step)
    mem_string = ":" * int(mem_percent/step) + " " * int(step - (mem_percent)//step)
    net_string = ":" * int(net_percent//step) + " " * int(step - (net_percent)//step)

    print(f"{time} | CPU: {cpu_string} {cpu_percent}% | Memory: {mem_string} {mem_percent}% | Network: {net_string} {net_percent}% | Nodes: {node_count}")

def load_workload(fname):
    with open(f'data/{fname}.txt', 'r') as f:
        line = [line.strip() for line in f.readlines()]
        # remove first line because it is the header
        line.pop(0)
        data = []
        for i in range(len(line)):
            line[i] = line[i].split(',')
            data.append(Task(int(line[i][0]), int(line[i][1]), int(line[i][2]), int(line[i][3]), int(line[i][4]), int(line[i][5])))
    return data

#load config json file
def load_config():
    with open('data/config.json', 'r') as f:
        config = json.load(f)
    return config

def random_forest(data, node_lst):
    X = [[task.arrival_time, task.num_tasks, task.cpu, task.memory, task.network, task.deadline] for task in data]
    Y = node_lst
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # %%
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

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R^2 Score: {r2}")

    return best_model

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