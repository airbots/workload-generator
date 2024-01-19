package org.ad.innovation.agent;

import java.util.*;
import java.lang.*;
public class UniformDistribution {
	private Random rnd = new Random(System.currentTimeMillis());
	private int rangeInt;
	private float rangeFloat;
	private int nextIntegerValue;
	private float nextFloatValue;
	
	public UniformDistribution(){}
	public UniformDistribution(long blockNumber){
		rangeInt=(int)blockNumber;
		
	}
	
	int nextInt(int celling){
		nextIntegerValue=this.rnd.nextInt(celling);
		return nextIntegerValue;
	}
	int nextInt(){
		nextIntegerValue=this.rnd.nextInt(10);
		return nextIntegerValue;
	}
}
