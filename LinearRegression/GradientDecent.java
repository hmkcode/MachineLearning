package com.hmkcode.ml;

import java.util.Arrays;

public class GradientDecent {
	
	static  Double x[][] = new Double[2][];
	static Double y[];
	
	static double stepsize = 0.001;
	static int m = 0;
	static double theta[] = new double[x.length];
	
	static Double hx[];
	static Double error[];
	
	static double e = Integer.MAX_VALUE;
	
	public static void main(String[] args) {
		//x[0] = new Double[] {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
		//x[1] = new Double[]{1.0,2.0,3.0,4.0,4.0,5.0,6.0,6.0,7.0,8.0,9.0,9.0,10.0,10.0};
		//y = new Double[] {23.0,29.0,49.0,64.0,74.0,87.0,96.0,97.0,109.0,119.0,149.0,145.0,154.0,166.0};
		
		x[0] = new Double[] {1.0,1.0,1.0,1.0,1.0};
		x[1] = new Double[] {0.0,1.0,2.0,3.0,4.0};
		y = new Double[]{1.0,3.0,7.0,13.0,21.0};
		
		m = x[0].length;
		hx = new Double[x[0].length];
		error = new Double[x[0].length];
		
		
		
		theta[0] = 0;
		theta[1] = 0;
		int counter = 0 ;
		
		while(e > 0.01){
			
			// 1. compute predictions
			for(int i = 0 ; i < m; i++)
				hx[i] = h(i);
			
			System.out.println("Prediction: "+ Arrays.asList(hx));
			
			// 2. compute error
			for(int i = 0 ; i < m; i++)
				error[i] =   hx[i] - y[i]  ; //change sign of update
			
			System.out.println("Errors: "+ Arrays.asList(error)+" "+sigma(0));
			
			// 3. update 
			
			//batchUpdate();
			stochasticUpdate();
			
			// 4. gradient magnitude
			e = Math.sqrt(Math.pow(sigma(0),2) + Math.pow(sigma(1),2));
			
			System.out.println("Gradient magnitude: "+e);
			
			System.out.println("theat0: "+theta[0]+" theta[1]: "+theta[1]);
			System.out.println("---------------------------");
			counter++;
		}
		System.out.println("Counter: "+counter);
	}
	
	
	
	
	public static void batchUpdate(){
		
		for(int j= 0; j < theta.length; j++)
			theta[j] = theta[j] - stepsize*sigma(j);
		
	}
	public static void stochasticUpdate(){
		
		for(int i=0; i < m; i++){
			for(int j= 0; j < theta.length; j++){
				theta[j] = theta[j] - stepsize*error[i]*x[j][i];
			}
		}
	}
	
	public static double h(int i){
		return theta[0]*x[0][i]+theta[1]*x[1][i];		
	}
	
	public static double sigma(int j){
		
		double sum=0;
		for(int i = 0; i < m; i++)
			sum += error[i]*x[j][i];
		
		return sum;
	}
}
