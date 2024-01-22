# %%
import time

class WorkLoad:
    def __init__(self, arr_tme, task_num, cpu, mem, network, ddl):
        self.arrival_time = arr_tme
        self.num_tasks = task_num
        self.cpu = cpu
        self.memory = mem
        self.network = network
        self.deadline = ddl
    
    def __str__(self):
        return f"arrival_time: {self.arrival_time}, num_tasks: {self.num_tasks}, cpu: {self.cpu}, memory: {self.memory}, network: {self.network}, deadline: {self.deadline}"

def load_workload():
    with open('data/workload.txt', 'r') as f:
        line = [line.strip() for line in f.readlines()]
        # remove first line
        line.pop(0)
        data = []
        for i in range(len(line)):
            line[i] = line[i].split()
            data.append(WorkLoad(int(line[i][0]), int(line[i][1]), int(line[i][2]), int(line[i][3]), int(line[i][4]), int(line[i][5])))
    return data
    
print(load_workload())