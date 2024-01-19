package org.ad.innovation.agent;
import java.lang.*;
import java.util.*;

public class ZipfGenerator {

	 private Random rnd = new Random(System.currentTimeMillis());
	 private int size;
	 private double skew;
	 private double bottom = 0;

	 public ZipfGenerator(int size, double skew) {
	     this.size = size;
	     this.skew = skew;
	     for(int i=1;i<size+1; i++) {
	         this.bottom += (1/Math.pow(i, this.skew));
	     }
	 }
	 public int next() {
		 int rank;
		 double friquency = 0;
		 double dice;
		 rank = rnd.nextInt(size)+1;
		 friquency = (1.0d / Math.pow(rank, this.skew)) / this.bottom;
		 dice = rnd.nextDouble();
		 while(!(dice < friquency)) {
		     rank = rnd.nextInt(size)+1;
		     friquency = (1.0d / Math.pow(rank, this.skew)) / this.bottom;
		     dice = rnd.nextDouble();
		 }
		 return rank;
	 }
	 public double getProbability(int rank) {
		   return (1.0d / Math.pow(rank, this.skew)) / this.bottom;
	 }
}
