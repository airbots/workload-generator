# %%
from utils import load_workload, load_config, get_min_nodes_lst, get_num_task_per_node
import matplotlib.pyplot as plt
data = load_workload('workload')
config = load_config()
# %%

# Data
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
