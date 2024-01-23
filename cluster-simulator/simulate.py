# %%
from utils import load_workload, load_config
from cluster import Cluster
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

data = load_workload('workload3')
config = load_config()

cluster = Cluster(config['Initial_Node_Count'])

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

print(get_min_nodes_lst(data, config))
# %%
X = [[task.arrival_time, task.num_tasks, task.cpu, task.memory, task.network, task.deadline] for task in data]
Y = get_min_nodes_lst(data, config)
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

# %%
