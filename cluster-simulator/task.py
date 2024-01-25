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

    def __init__(self, arrival_time, num_tasks, cpu, mem, network, ddl):
        if cpu < 0 or mem < 0 or network < 0 or num_tasks < 0:
            raise Exception("Invalid parameters")
        
        self.arrival_time = arrival_time
        self.num_tasks = num_tasks
        self.cpu = cpu
        self.memory = mem
        self.network = network
        self.deadline = ddl

    def __lt__(self, other):
        return self.arrival_time < other.arrival_time

    def __str__(self):
        return f"arrival_time: {self.arrival_time}, num_tasks: {self.num_tasks}, cpu: {self.cpu}, memory: {self.memory}, network: {self.network}, deadline: {self.deadline}"
    
    def get_avg_metrics(self, node_count):
        return self.cpu/node_count, self.memory/node_count, self.network/node_count
