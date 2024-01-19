package org.ad.innovation.agent;

import java.util.Random;


public class NormalDistribution {
	private Random rg;
	private long mean;
	private long sd;

	public NormalDistribution(long mean, long variance){
		this.mean=mean;
		this.sd=variance;
		this.rg = new Random();
	}
	
	public double next(){
		double ranFileLen=rg.nextGaussian()*sd +mean;
		return ranFileLen;
	}
}
