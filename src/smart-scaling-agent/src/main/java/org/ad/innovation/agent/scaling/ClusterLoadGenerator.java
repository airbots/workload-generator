package org.ad.innovation.agent.scaling;

import org.ad.innovation.agent.ExponentialGenerator;
import org.ad.innovation.agent.NormalDistribution;
import org.ad.innovation.agent.UniformDistribution;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Example of running the load generator
 * \>java -jar load-generator-0.1.0-jar-with-dependencies.jar org.ad.innovation.agent.scaling.ClusterLoadGenerator -t 20 -lm 100 -lv 20 -dl 2 -r true -j 100 -u true -U 20,40,30
 * Output file will be written input workload_timestamp.txt file
 */
public class ClusterLoadGenerator extends AbstractLoadGenerator {

    public void generate_load() {
        this.timer=new ExponentialGenerator(waitInterval);
        this.loadSize=new ExponentialGenerator(averageSize);
        this.nd=new NormalDistribution(averageSize,variance);
        this.loadUD=new UniformDistribution(100);
        this.intervalUD=new UniformDistribution(10000);

        //Deadline will be propotional to the load reasonablly,
        if(!reasonableDeadline) {
            this.deadlineND = new NormalDistribution(averageSize/10,variance/10);
        }
        //write out the workload into a file
        try (FileWriter fws = new FileWriter(
                new File("workload_" + Long.toString(System.currentTimeMillis() / 1000) + ".txt"));
             BufferedWriter bws = new BufferedWriter(fws)) {
            long startTime=0;
            long sleepInterval=0;
            long load=0;
            long deadline=0;
            int cpu=this.cpu;
            int memory=this.mem;
            int network=this.net;
            bws.write("ArrivalTime, No. of Tasks in job, CPU utilization for each task, Memory Utilization for each task, Network Utilization for each task, Deadline for the job\n");
            while(this.numberOfJobs>0) {
                startTime+=sleepInterval;
                load=loadSize.next();
                while(load==0) {
                    load=loadSize.next();
                }
                if(reasonableDeadline) {
                    deadline = load*this.deadlineRatio;
                } else {
                    deadline = (long) deadlineND.next();
                    while(deadline==0) {
                        deadline = (long) deadlineND.next();
                    }
                }
                if(!this.uniformUtilization) {
                    cpu = resourceUtil.nextInt(100);
                    while(cpu==0)
                        cpu = resourceUtil.nextInt(100);
                    memory = resourceUtil.nextInt(100);
                    while(memory==0)
                        memory = resourceUtil.nextInt(100);
                    network = resourceUtil.nextInt(100);
                    while(network==0)
                        network = resourceUtil.nextInt(100);
                }
                String job = startTime+","+load+","+cpu+","+memory+","+network+","+deadline+"\n";
                bws.write(job);
                sleepInterval = timer.next();
                while(sleepInterval==0) {
                    sleepInterval = timer.next();
                }
                bws.flush();
                this.numberOfJobs--;
            }
        } catch(Exception e) {
            System.out.println("Exception during generating workload" + e.getMessage());
        }
    }
    public static void main(String[] args) {
        ClusterLoadGenerator clg=new ClusterLoadGenerator();
        System.out.println("**********************************************************************************");
        System.out.println("ClusterLoadGenerator parameter:");
        System.out.println("-t\tThe average time for the inter-arrival time");
        System.out.println("-lm\tAverage Load size");
        System.out.println("-lv\tVariance of Load size");
        System.out.println("-randomSize \t Specify whether your load size is random or the fixed load size,-1 is random");
        System.out.println("-strictUD\tUse fixed number of inter-arrival time and block number requested by each job");
        System.out.println("-strictSize\tFixed load for each job");
        System.out.println("-normalUD\tUse relaxed uniform distribution so that the inter-arrival time is exponential");
        System.out.println("-dl\t Job Deadline ratio comparing job load, if reasonable deadline, it should be this ratio multiply load");
        System.out.println("-r\t Is deadline reasonable");
        System.out.println("-j\t Total number of jobs in the load");
        System.out.println("-u\t Whether job utilization is uniform or not, if same means CPU/MEM/Net utilization will be the same, " +
                "otherwise, it will be random");
        System.out.println("-U\t If Resource Utilization is uniformed of a task, please specify them in form of 20,10,40, means CPU,mem,net");
        System.out.println("**********************************************************************************");
        for(int i=0; i < args.length; ++i) {
            try {
                if("-lm".equals(args[i])){
                    clg.averageSize=Long.parseLong(args[++i]);
                } else if("-lv".equals(args[i])){
                    clg.variance=Long.parseLong(args[++i]);
                } else if("-t".equals(args[i])){
                    clg.waitInterval=Integer.parseInt(args[++i]);
                } else if("-randomSize".equals(args[i])){
                    clg.randomSize=Integer.parseInt(args[++i]);
                } else if("-strictUD".equals(args[i])){
                    clg.strictUD=Integer.parseInt(args[++i]);
                } else if("-strictSize".equals(args[i])){
                    clg.strictSize=Integer.parseInt(args[++i]);
                } else if("-normalUD".equals(args[i])){
                    clg.normalUD=Integer.parseInt(args[++i]);
                } else if("-dl".equals(args[i])) {
                    clg.deadlineRatio = Integer.parseInt(args[++i]);
                } else if("-r".equals(args[i])) {
                    clg.reasonableDeadline=Boolean.parseBoolean(args[++i]);
                    //System.out.println("Is Deadline Random?\t"+reasonableDeadline);
                } else if("-j".equals(args[i])) {
                    clg.numberOfJobs=Integer.parseInt(args[++i]);
                } else if("-u".equals(args[i])) {
                    clg.uniformUtilization=Boolean.parseBoolean(args[++i]);
                } else if("-U".equals(args[i])) {
                    String utilization = args[++i];
                    if (utilization!=null && clg.uniformUtilization && utilization.split(",").length==3) {
                        String[] util=utilization.split(",");
                        clg.cpu=Integer.parseInt(util[0]);
                        clg.mem=Integer.parseInt(util[1]);
                        clg.net=Integer.parseInt(util[2]);
                    }
                }
            } catch(Exception e) {System.out.println("Input parameter parsing error:"+e.toString());}
        }
        clg.generate_load();
    }
}
