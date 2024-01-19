# This helper function is used to print the utilization of the cluster
def print_utilization(time, cpu_percent, mem_percent, net_percent, node_count, step=10):
    cpu_string = ":" * int(cpu_percent//step) + " " * int(step - (cpu_percent)//step)
    mem_string = ":" * int(mem_percent/step) + " " * int(step - (mem_percent)//step)
    net_string = ":" * int(net_percent//step) + " " * int(step - (net_percent)//step)

    print(f"{time} | CPU: {cpu_string} {cpu_percent}% | Memory: {mem_string} {mem_percent}% | Network: {net_string} {net_percent}% | Nodes: {node_count}")
