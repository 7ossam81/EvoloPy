# -*- coding: utf-8 -*-
"""
Created on Thirsday March 21  2019

@author: 
% _____________________________________________________
% Main paper:
% Harris hawks optimization: Algorithm and applications
% Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
% Future Generation Computer Systems, 
% DOI: https://doi.org/10.1016/j.future.2019.02.028
% _____________________________________________________

"""
import random
import numpy
import math
from solution import solution
import time



def HHO(objf,lb,ub,dim,SearchAgents_no,Max_iter):

    #dim=30
    #SearchAgents_no=50
    #lb=-100
    #ub=100
    #Max_iter=500
        
    
    # initialize the location and Energy of the rabbit
    Rabbit_Location=numpy.zeros(dim)
    Rabbit_Energy=float("inf")  #change this to -inf for maximization problems
    
    
    #Initialize the locations of Harris' hawks
    X=numpy.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb
    
    #Initialize convergence
    convergence_curve=numpy.zeros(Max_iter)
    
    
    ############################
    s=solution()

    print("HHO is now tackling  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################
    
    t=0  # Loop counter
    
    # Main loop
    while t<Max_iter:
        for i in range(0,SearchAgents_no):
            
            # Check boundries
                      
            X[i,:]=numpy.clip(X[i,:], lb, ub)
            
            # fitness of locations
            fitness=objf(X[i,:])
            
            # Update the location of Rabbit
            if fitness<Rabbit_Energy: # Change this to > for maximization problem
                Rabbit_Energy=fitness 
                Rabbit_Location=X[i,:].copy() 
            
        E1=2*(1-(t/Max_iter)) # factor to show the decreaing energy of rabbit    
        
        # Update the location of Harris' hawks 
        for i in range(0,SearchAgents_no):

            E0=2*random.random()-1  # -1<E0<1
            Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy)>=1:
                #Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no*random.random())
                X_rand = X[rand_Hawk_index, :]
                if q<0.5:
                    # perch based on other family members
                    X[i,:]=X_rand-random.random()*abs(X_rand-2*random.random()*X[i,:])

                elif q>=0.5:
                    #perch on a random tall tree (random site inside group's home range)
                    X[i,:]=(Rabbit_Location - X.mean(0))-random.random()*((ub-lb)*random.random()+lb)

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy)<1:
                #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                #phase 1: ----- surprise pounce (seven kills) ----------
                #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r=random.random() # probablity of each event
                
                if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                    X[i,:]=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X[i,:])

                if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                    Jump_strength=2*(1- random.random()) # random jump strength of the rabbit
                    X[i,:]=(Rabbit_Location-X[i,:])-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])
                
                #phase 2: --------performing team rapid dives (leapfrog movements)----------

                if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                    #rabbit try to escape by many zigzag deceptive motions
                    Jump_strength=2*(1-random.random())
                    X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])

                    if objf(X1)< fitness: # improved move?
                        X[i,:] = X1.copy()
                    else: # hawks perform levy-based short rapid dives around the rabbit
                        X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])+numpy.multiply(numpy.random.randn(dim),Levy(dim))
                        if objf(X2)< fitness:
                            X[i,:] = X2.copy()
                if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                     Jump_strength=2*(1-random.random())
                     X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))
                     
                     if objf(X1)< fitness: # improved move?
                        X[i,:] = X1.copy()
                     else: # Perform levy-based short rapid dives around the rabbit
                         X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))+numpy.multiply(numpy.random.randn(dim),Levy(dim))
                         if objf(X2)< fitness:
                            X[i,:] = X2.copy()
                
        convergence_curve[t]=Rabbit_Energy
        if (t%1==0):
               print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Rabbit_Energy)])
        t=t+1
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="HHO"   
    s.objfname=objf.__name__
    s.best =Rabbit_Energy 
    s.bestIndividual = Rabbit_Location
    
    
    
    return s

def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
    u= 0.01*numpy.random.randn(dim)*sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v),(1/beta))
    step = numpy.divide(u,zz)
    return step
