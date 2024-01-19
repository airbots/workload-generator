import logging

class Cluster:
    def __init__(self, node_count=10):
        self.nodes = [Node(0, 0, 0) for _ in range(node_count)]

    def scale(self, node_count):
        node_diff = node_count - len(self.nodes)
        # No scaling
        if node_diff == 0:
            logging.info("No scaling needed")
        # Scale in
        elif node_diff > 0: 
            self.nodes = self.nodes.append([Node(0, 0, 0) for _ in range(node_diff)])
            self.load_rebalance(status="add", node_count=node_count)
        # Scale out
        elif node_diff < 0:
            self.load_rebalance(status="remove", node_count=node_count)
            # keep only desired number of nodes
            self.nodes = self.nodes[:node_count]
    
    def add_task(self, task):
        total_cpu, total_memory, total_network = self.get_total_utilization()
        if task.cpu > total_cpu or task.memory > total_memory or task.network > total_network:
            raise Exception("Task resources exceed total available resources")
        
        avg_cpu, avg_memory, avg_network = task.get_avg_metrics(len(self.nodes))
        for node in self.nodes:
            node.cpu += avg_cpu
            node.memory += avg_memory
            node.network += avg_network

    def remove_task(self, task):
        avg_cpu, avg_memory, avg_network = task.get_avg_metrics(len(self.nodes))

        for node in self.nodes:
            node.cpu -= avg_cpu
            node.memory -= avg_memory
            node.network -= avg_network

    def load_rebalance(self, status, node_count):
        total_cpu, total_memory, total_network = self.get_total_utilization()
        if status == "add":
            node_count = node_count
        elif status == "remove":
            node_count = node_count -1

        for node in self.nodes:
            node.cpu = total_cpu / node_count
            node.memory = total_memory / node_count
            node.network = total_network / node_count


    def get_total_utilization(self):
        cpu = sum([node.cpu for node in self.nodes])
        memory = sum([node.memory for node in self.nodes])
        network = sum([node.network for node in self.nodes])
        return round(cpu), round(memory), round(network)

    def get_avg_cluster_utilization(self):
        cpu, memory, network = self.get_total_utilization()
        return round(cpu/len(self.nodes)), round(memory/len(self.nodes)), round(network/len(self.nodes))
    
class Node:
    def __init__(self, cpu, memory, network):
        self.cpu = cpu
        self.memory = memory
        self.network = network

    def __eq__(self, other):
        return self.cpu == other.cpu and self.memory == other.memory and self.network == other.network