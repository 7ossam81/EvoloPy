# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:06:34 2016

@author: hossam
"""
import random
import numpy
import time
import math
import sklearn
from numpy import asarray
from sklearn.preprocessing import normalize
from solution import solution


  



def normr(Mat):
   """normalize the columns of the matrix
   B= normr(A) normalizes the row
   the dtype of A is float"""
   Mat=Mat.reshape(1, -1)
     # Enforce dtype float
   if Mat.dtype!='float':
      Mat = asarray(Mat,dtype=float)

   # if statement to enforce dtype float
   B = normalize(Mat,norm='l2',axis=1)
   B=numpy.reshape(B,-1)
   return B

def randk(t):
    if (t%2)==0:
        s=0.25
    else:
         s=0.75
    return s

def RouletteWheelSelection(weights):
  accumulation = numpy.cumsum(weights)
  p = random.random() * accumulation[-1]
  chosen_index = -1;
  for index in range (0, len(accumulation)):
    if (accumulation[index] > p):
      chosen_index = index;
      break;
  
  choice = chosen_index;

  return choice



def MVO(objf,lb,ub,dim,N,Max_time):

    "parameters"
    #dim=30
    #lb=-100
    #ub=100
    WEP_Max=1;
    WEP_Min=0.2
    #Max_time=1000
    #N=50
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    
    
    
    Universes = numpy.zeros((N, dim))
    for i in range(dim):
        Universes[:, i] = numpy.random.uniform(0,1, N) * (ub[i] - lb[i]) + lb[i]

    Sorted_universes=numpy.copy(Universes)
    
    convergence=numpy.zeros(Max_time)
     
    
    Best_universe=[0]*dim;
    Best_universe_Inflation_rate= float("inf")
    
    
    
    
    s=solution()

    
    Time=1;
    ############################################
    print("MVO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    while (Time<Max_time+1):
    
        "Eq. (3.3) in the paper"
        WEP=WEP_Min+Time*((WEP_Max-WEP_Min)/Max_time)
       
        TDR=1-(math.pow(Time,1/6)/math.pow(Max_time,1/6))
      
        Inflation_rates=[0]*len(Universes)
        
       
       
        for i in range(0,N):
            for j in range(dim):
                Universes[i,j]=numpy.clip(Universes[i,j], lb[j], ub[j])
    
            
    
            Inflation_rates[i]=objf(Universes[i,:]);
           
       
               
            if Inflation_rates[i]<Best_universe_Inflation_rate :
                        
                Best_universe_Inflation_rate=Inflation_rates[i]
                Best_universe=numpy.array(Universes[i,:])
             
        
        sorted_Inflation_rates = numpy.sort(Inflation_rates)
        sorted_indexes = numpy.argsort(Inflation_rates)
        
        for newindex in range(0,N):
            Sorted_universes[newindex,:]=numpy.array(Universes[sorted_indexes[newindex],:])   
            
        normalized_sorted_Inflation_rates=numpy.copy(normr(sorted_Inflation_rates))
    
        
        Universes[0,:]= numpy.array(Sorted_universes[0,:])
    
        for i in range(1,N):
            Back_hole_index=i
            for j in range(0,dim):
                r1=random.random()
                
                if r1<normalized_sorted_Inflation_rates[i]:
                    White_hole_index=RouletteWheelSelection(-sorted_Inflation_rates);
    
                    if White_hole_index==-1:
                        White_hole_index=0;
                    White_hole_index=0;
                    Universes[Back_hole_index,j]=Sorted_universes[White_hole_index,j];
            
                r2=random.random() 
                
                
                if r2<WEP:
                    r3=random.random() 
                    if r3<0.5:                    
                        Universes[i,j]=Best_universe[j]+TDR*((ub[j]-lb[j])*random.random()+lb[j]) #random.uniform(0,1)+lb);
                    if r3>0.5:          
                        Universes[i,j]=Best_universe[j]-TDR*((ub[j]-lb[j])*random.random()+lb[j]) #random.uniform(0,1)+lb);
                              
        
        
        
        
        convergence[Time-1]=Best_universe_Inflation_rate
        if (Time%1==0):
               print(['At iteration '+ str(Time)+ ' the best fitness is '+ str(Best_universe_Inflation_rate)]);

        
        
        
        Time=Time+1
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence
    s.optimizer="MVO"
    s.objfname=objf.__name__

    return s



