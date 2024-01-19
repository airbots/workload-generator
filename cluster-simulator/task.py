class Task:
    """
    Define the structure of the task data.

    A Task class consists of the following attributes:
    - load (int): The load of the task
    - cpu (float): The CPU utilization of the task
    - memory (float): The memory utilization of the task
    - network (float): The network bandwidth utilization of the task
    - arrival_time (int): The arrival time of the task
    - deadline (int): The deadline of the task
    """

    def __init__(self, load, cpu, memory, network, arrival_time, deadline):
        if cpu < 0 or memory < 0 or network < 0 or arrival_time < 0:
            raise Exception("Invalid parameters")
        
        self.load = load
        self.cpu = cpu
        self.memory = memory
        self.network = network
        self.arrival_time = arrival_time
        self.deadline = deadline

    def __lt__(self, other):
        return self.arrival_time < other.arrival_time

    def __str__(self):
        return f"Task({self.load}, {self.cpu}, {self.memory}, {self.network}, {self.arrival_time}, {self.deadline})"
    
    def get_avg_metrics(self, node_count):
        return self.cpu/node_count, self.memory/node_count, self.network/node_count
