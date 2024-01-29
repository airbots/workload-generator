# %%
# Visualize the Results

import matplotlib.pyplot as plt
import numpy as np
from math import pi

scenarios = ['Small Economic Cluster', 'Abundantly Provisioned Cluster', 'Smart Agent Cluster']

data = {
    'Acceptance_Rate': [24, 100, 100],
    'Overall_Utilization': [77, 17, 34],
    'Task_Completion_Time_Score': [100, 100, 100],
    'Completed_Jobs_Score': [25, 100, 100],
    'Rejected_Jobs_Score': [25, 100, 100],
    'Node_Utilization_Score': [100, 0, 50]
}

acceptance_rate = data['Acceptance_Rate']
overall_utilization = data['Overall_Utilization']
completed_jobs = [37, 150, 150]
node_count_avg = [1, 10, 5]


def bar_plot():
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

    axs[1, 0].bar(scenarios, completed_jobs, color=colors)
    axs[1, 0].set_title('Number of Completed Jobs')
    add_labels(axs[1, 0], completed_jobs)
    axs[1, 0].set_ylim(0, max(completed_jobs) + 8)

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


def radar_plot():

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

