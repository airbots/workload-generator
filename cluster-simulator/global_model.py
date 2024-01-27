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
# Visualize the Results

import matplotlib.pyplot as plt
import numpy as np
import math

# Data
scenarios = ['Small Economic Cluster', 'Abundantly Provisioned Cluster', 'Smart Agent Cluster']
acceptance_rate = [24, 100, 100]
overall_utilization = [77, 17, 34]
task_completion_time = [850, 851, 850]
completed_jobs = [37, 150, 150]
rejected_jobs = [113, 0, 0]
node_count_avg = [1, 10, 5]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Function to add labels
def add_labels(ax, data):
    for i, v in enumerate(data):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom')

# Plot data with labels
axs[0, 0].bar(scenarios, acceptance_rate, color=colors)
axs[0, 0].set_title('Acceptance Rate')
add_labels(axs[0, 0], acceptance_rate)

axs[0, 1].bar(scenarios, overall_utilization, color=colors)
axs[0, 1].set_title('Overall Utilization')
add_labels(axs[0, 1], overall_utilization)

# axs[0, 2].bar(scenarios, task_completion_time, color=colors)
# axs[0, 2].set_title('Task Completion Time')
# add_labels(axs[0, 2], task_completion_time)

axs[1, 0].bar(scenarios, completed_jobs, color=colors)
axs[1, 0].set_title('Number of Completed Jobs')
add_labels(axs[1, 0], completed_jobs)
axs[1, 0].set_ylim(0, max(completed_jobs) + 8)

# axs[1, 1].bar(scenarios, rejected_jobs, color=colors)
# axs[1, 1].set_title('Number of Rejected Jobs')
# add_labels(axs[1, 1], rejected_jobs)
# axs[1, 1].set_ylim(0, max(rejected_jobs) + 1)

axs[1, 1].bar(scenarios, node_count_avg, color=colors)
axs[1, 1].set_title('Average Node Count')
add_labels(axs[1, 1], node_count_avg)
axs[1, 1].set_ylim(0, max(node_count_avg) + 8)


# Rotate scenario labels
for ax in axs.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Improve layout
plt.tight_layout()
plt.show()
plt.savefig('ml_model/results.png', dpi=300)

# %%
import matplotlib.pyplot as plt
import numpy as np
from math import pi

scenarios = ['Small Economic Cluster', 'Worry-free Cluster', 'Smart Agent Cluster']
data = {
    'Acceptance Rate': [24, 100, 100],
    'Overall Utilization': [77, 17, 34],
    'Task Completion Time Score': [100, 100, 100],
    'Completed Jobs Score': [25, 100, 100],
    'Rejected Jobs Score': [25, 100, 100],
    'Node Utilization Score': [100, 0, 50]
}

categories = list(data.keys())
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # ensure the plot is a closed circle

plt.figure(figsize=(10, 10))
plt.subplot(polar=True)

for i, scenario in enumerate(scenarios):
    values = data.values()
    values = [x[i] for x in values]
    values += values[:1]  # ensure the plot is a closed circle
    plt.plot(angles, values, linewidth=1, linestyle='solid', label=scenario)
    plt.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], categories, color='black', size=15)
plt.yticks(color="grey", size=10)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()
plt.savefig('ml_model/radar.png', dpi=300)
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

X_data = load_workload('workload7')

input_lst = [[80, 80, calc_completion_time(data), data.num_tasks] for data in X_data]

loaded_model = joblib.load('ml_model/random_forest_model.joblib')
prd_node_lst = loaded_model.predict(input_lst)
prd_node_lst = [int(i) for i in prd_node_lst]

print(prd_node_lst)
# %%

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

# Load the model
loaded_model = joblib.load('ml_model/random_forest_model.joblib')

# Extract single tree
estimator = loaded_model.estimators_[5]

# Visualize the tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(estimator,
               filled = True)
fig.savefig('rf_individualtree.png')

# %%
