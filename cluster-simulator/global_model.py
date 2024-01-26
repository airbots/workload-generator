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

df.to_csv('data/model_output.csv', index=False)
# %%
df

# %%
