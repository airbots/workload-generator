package org.ad.innovation.agent.scaling;

import org.ad.innovation.agent.ExponentialGenerator;
import org.ad.innovation.agent.NormalDistribution;
import org.ad.innovation.agent.UniformDistribution;
import org.ad.innovation.agent.ZipfGenerator;

import java.util.Random;


public abstract class AbstractLoadGenerator {

    //static long averageTime=11 * 1000;
    static long averageSize=640;
    static long variance=1;
    static long waitInterval=1;
    static int randomSize=-1;
    static int strictUD;
    static int strictSize=10;
    static int normalUD;
    static long deadlineRatio=2;
    static boolean reasonableDeadline=true;
    static int numberOfJobs=100;
    static boolean uniformUtilization=true;
    static int cpu=0;
    static int mem=0;
    static int net=0;

    ExponentialGenerator timer;
    ExponentialGenerator loadSize;
    NormalDistribution nd;
    UniformDistribution loadUD;
    UniformDistribution intervalUD;
    NormalDistribution deadlineND;
    Random resourceUtil=new Random();
}
