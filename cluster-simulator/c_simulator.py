# %%
import csv,time
print("*********************   Start  ************************")
print("****************   Cluster Simulator  *******************")

#ArrivalTime, No. of Tasks in job, CPU utilization for each task, Memory Utilization for each task, Network Utilization for each task, Deadline for the job
#ArrivalTime, Num_Task, Max_CPU_Utilization, Memory_Utilization, Network_Utilization, job_deadline
# 0,2,50,40,40,4
# 3,10,50,40,40,10   Reject?
# 17,45,20,40,20,7
# 26,52,20,40,20,11
# 36,157,20,40,20,11

#case 1
# ArrivalTime=0
# Num_Task=2
# Max_CPU_Utilization=50
# Memory_Utilization=40
# Network_Utilization=40
# job_deadline=4
# of node can split 

#case 2
# ArrivalTime=3
# Num_Task=10
# Max_CPU_Utilization=50
# Memory_Utilization=40
# Network_Utilization=40
# job_deadline=10

#metrics:
# Node1: 100,80,80,4 
# Node2: 0,0,0,4

#rows=[]
dict_rows={'ArrivalTime':'',
           'Num_Task':'', 
            'Max_CPU_Utilization':'', 
            'Memory_Utilization':'', 
            'Network_Utilization':'', 
            'job_deadline':''}

#run submitted job (mark the rectengles of node resources);
#call ML algorithm to get how many nodes needed;
def get_nodes_count():
    num_of_max_nodes=2 #to do using ML algo
    return num_of_max_nodes

#check status from output file
def is_task_running():
    is_running ='Y'
    return True

#emit current CPU/MEM/NETWORK utilization, acceptance rate;
mtrx_emit={'cpu_utilization' : '', 'memory_utilization':'' , 'nw_utilization':'', 'acceptance_rate':'' ,'job_status':''}
#decide newly submitted job can be finished before deadline or not, based on current running jobs and resources in cluster. 
#If not, reject the job and update the acceptance rate.
def c_smul(ArrivalTime, Num_Task, Max_CPU_Utilization, Memory_Utilization, Network_Utilization, job_deadline, acceptance_rate):
    mtrx_emit["cpu_utilization"]=    int(Num_Task) * int(Max_CPU_Utilization)
    mtrx_emit["memory_utilization"]=    int(Num_Task) * int(Memory_Utilization)
    mtrx_emit["nw_utilization"]=    int(Num_Task) * int(Network_Utilization)
    mtrx_emit["acceptance_rate"]= acceptance_rate

    remaining_time= 5 - int(ArrivalTime)
    if  int(Max_CPU_Utilization) <= 100 and (int(job_deadline) - (int(Max_CPU_Utilization)/100 ))/get_nodes_count()< remaining_time: 
        mtrx_emit['job_status']='Accepted'

    else:
        mtrx_emit['job_status']='Rejected'

    return mtrx_emit

# %%
import csv,time
def read_workload_data():
    workload_file="data/workload.txt"
    with open(workload_file,'r') as file:
        #csv_reader=csv.reader(file)
        csv_reader=csv.DictReader(file)
        #header=next(csv_reader)
        #data=[row for row in csv_reader]
        start_time=0
        counter=0
        counter_accepted=0
        for row in csv_reader:
            is_task_running()
            print(row)
            #print(next(int(row['ArrivalTime'])))

            #wait for next arrival time
            next_run=int(row['ArrivalTime'])-start_time
            time.sleep(next_run)
            remaining_time= 5 - int(row['ArrivalTime'])
            counter+=1
            if  int(row['Max_CPU_Utilization']) <= 100 and (int(row['job_deadline']) - (int(row['Max_CPU_Utilization'])/100 ))/get_nodes_count()< remaining_time: 
                job_status='Accepted'
                counter_accepted+=1
            else:
                job_status='Rejected'

            acceptance_rate=counter_accepted/counter * 100
            start_time=int(row['ArrivalTime'])
            
            print(c_smul(row['ArrivalTime'],row['Num_Task'], row['Max_CPU_Utilization'], row['Memory_Utilization'], row['Network_Utilization'], row['job_deadline'],acceptance_rate))

read_workload_data()




#print(c_smul(ArrivalTime,Num_Task, Max_CPU_Utilization, Memory_Utilization, Network_Utilization, job_deadline))


# %%
