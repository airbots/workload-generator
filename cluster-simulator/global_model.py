# %%
from utils import load_workload, load_config, get_min_nodes_lst, get_num_task_per_node
import matplotlib.pyplot as plt
import pandas as pd
data = load_workload('workload')
config = load_config()
# %%

# Exploratory Data Analysis
arrival_time = [task.arrival_time for task in data]
no_of_tasks = [task.num_tasks for task in data]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(arrival_time, no_of_tasks)

# Set the labels
ax.set_xlabel('Arrival Time')
ax.set_ylabel('No. of Tasks in job')

# Display the plot
plt.show()

# %%
# Model Controller
from utils import load_workload, load_config
from simulate import scale_workload
import pandas as pd

iteration_count = 16
workload_count = 8

df = pd.DataFrame()

config = load_config()

for i in range(1,workload_count):
    for j in range(1, iteration_count):
        data = load_workload(f'workload{i}')
        config['Initial_Node_Count'] = j
        result = scale_workload(data, config)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)

df.to_csv('ml_model/model_output.csv', index=False)
# %%
# Model Training
import pandas as pd
from utils import random_forest, logistic_regression
import logging
import joblib

df = pd.read_csv('ml_model/model_output.csv')
# %%
Y = df['Node_Count']
X = df.drop(['Node_Count', 'No_of_Completed_Tasks', 'No_of_Rejected_Tasks'], axis=1)

X_lst = X.values.tolist()
Y_lst = Y.values.tolist()

logging.info(f"Lenght of X: {len(X_lst)}")
logging.info(f"Lenght of Y: {len(Y_lst)}")

model_lr = logistic_regression(X_lst, Y_lst)
model_rf = random_forest(X_lst, Y_lst)
# %%
# Save Model
joblib.dump(model_lr, 'ml_model/logistic_regression_model.joblib')
joblib.dump(model_rf, 'ml_model/random_forest_model.joblib')

# %%
# Load
from utils import load_workload, load_config, get_num_task_per_node
import joblib

config = load_config()


def calc_completion_time(task):
    return task.num_tasks * config['Time_To_Complete_Task'] / get_num_task_per_node(task, config)

def calc_overall_utilization(task):
    return max(task.cpu, task.memory, task.network) * get_num_task_per_node(task, config)

X_data = load_workload('workload1')

input_lst = [[80, 80, calc_completion_time(data), data.num_tasks] for data in X_data]

loaded_model = joblib.load('ml_model/random_forest_model.joblib')
prd_node_lst = loaded_model.predict(input_lst)
prd_node_lst = [int(i) for i in prd_node_lst]

print(prd_node_lst)
# %%
