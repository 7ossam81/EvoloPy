# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import benchmarks
import csv
import numpy
import time


def selector(algo,func_details,popSize,Iter):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    dim=func_details[3]
    
    if(algo==0):
        x=pso.PSO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==1):
        x=mvo.MVO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==2):
        x=gwo.GWO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==3):
        x=mfo.MFO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    return x
    
    
# Select optimizers
PSO= False
MVO= False
GWO = False
MFO= True


# Select benchmark function
F1=True
F2=False
F3=False
F4=True
F5=False
F6=False


optimizer=[PSO, MVO, GWO, MFO]
benchmarkfunc=[F1,F2,F3,F4,F5,F6] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=2

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 50
Iterations= 500

#Export results ?
Export=True
#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
Flag=False

for i in range (0, len(optimizer)):
    for j in range (0, len(benchmarkfunc)):
        if((optimizer[i]==True) and (benchmarkfunc[j]==True)): 
            for k in range (0,NumOfRuns):
                Flag=True # at least one experiment
                func_details=benchmarks.getFunctionDetails(j)
                x=selector(i,func_details,PopulationSize,Iterations)
                if(Export==True):
                    with open(ExportToFile, 'a',newline='\n') as out:
                        writer = csv.writer(out,delimiter=',')
                        a=numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,"Cnvg"],x.convergence])
                        writer.writerow(a)
              
                    out.close()
if (Flag==False):
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        