package org.ad.innovation.agent;
import java.util.*;
import java.lang.*;

public class ExponentialGenerator {
	private Random rnd=new Random(System.currentTimeMillis());
	long finish;
	long average;
	//long simPeriod;
	//long startTime;

	public ExponentialGenerator(long average){
		this.average=average;
		//this.simPeriod=simPeriod;
		//this.startTime=startTime;
		
	}
	public long next(){
		double prob=this.rnd.nextDouble();
		long interArrival;
			
		interArrival= (long) (average*(Math.log(1.0d/(1.0d-prob))));
		
		return interArrival; 
	}
		/*
		public double getProbability(){
			return ;
		}
		*/
}