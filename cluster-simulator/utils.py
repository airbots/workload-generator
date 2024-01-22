from task import Task
import json
# This helper function is used to print the utilization of the cluster
def print_utilization(time, cpu_percent, mem_percent, net_percent, node_count, step=10):
    cpu_string = ":" * int(cpu_percent//step) + " " * int(step - (cpu_percent)//step)
    mem_string = ":" * int(mem_percent/step) + " " * int(step - (mem_percent)//step)
    net_string = ":" * int(net_percent//step) + " " * int(step - (net_percent)//step)

    print(f"{time} | CPU: {cpu_string} {cpu_percent}% | Memory: {mem_string} {mem_percent}% | Network: {net_string} {net_percent}% | Nodes: {node_count}")

def load_workload():
    with open('data/workload.txt', 'r') as f:
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