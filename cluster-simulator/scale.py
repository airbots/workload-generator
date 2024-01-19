import random
from datetime import datetime
import time
from utils import print_utilization
from cluster import Cluster

cluster = Cluster()


while True:
    now_string = datetime.now().strftime("%m-%d %H:%M:%S")
    cpu_percent, mem_percent, net_percent = cluster.get_avg_cluster_utilization()
    node_count = len(cluster.nodes)

    print_utilization(now_string, cpu_percent, mem_percent, net_percent, node_count)

    time.sleep(3)
