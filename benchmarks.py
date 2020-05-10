# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris
"""

import numpy
import math

# define the function blocks
def prod( it ):
    p= 1
    for n in it:
        p *= n
    return p

def Ufun(x,a,k,m):
    y=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a));
    return y
    
def F1(x):
    s=numpy.sum(x**2);
    return s

def F2(x):
    o=sum(abs(x))+prod(abs(x));
    return o;     
           
def F3(x):
    dim=len(x)+1;
    o=0;
    for i in range(1,dim):
        o=o+(numpy.sum(x[0:i]))**2; 
    return o; 
    
def F4(x):
    o=max(abs(x));
    return o;     

def F5(x):
    dim=len(x);
    o=numpy.sum(100*(x[1:dim]-(x[0:dim-1]**2))**2+(x[0:dim-1]-1)**2);
    return o; 

def F6(x):
    o=numpy.sum(abs((x+.5))**2);
    return o;

def F7(x):
   dim=len(x);

   w=[i for i in range(len(x))]
   for i in range(0,dim):
        w[i]=i+1;
   o=numpy.sum(w*(x**4))+numpy.random.uniform(0,1);
   return o;

def F8(x):
    o=sum(-x*(numpy.sin(numpy.sqrt(abs(x)))));
    return o;

def F9(x):
    dim=len(x);
    o=numpy.sum(x**2-10*numpy.cos(2*math.pi*x))+10*dim;
    return o;


def F10(x):
    dim=len(x);
    o=-20*numpy.exp(-.2*numpy.sqrt(numpy.sum(x**2)/dim))-numpy.exp(numpy.sum(numpy.cos(2*math.pi*x))/dim)+20+numpy.exp(1);
    return o;

def F11(x):
    dim=len(x);
    w=[i for i in range(len(x))]
    w=[i+1 for i in w];
    o=numpy.sum(x**2)/4000-prod(numpy.cos(x/numpy.sqrt(w)))+1;   
    return o;
    
def F12(x):
    dim=len(x);
    o=(math.pi/dim)*(10*((numpy.sin(math.pi*(1+(x[0]+1)/4)))**2)+numpy.sum((((x[1:dim-1]+1)/4)**2)*(1+10*((numpy.sin(math.pi*(1+(x[1:dim-1]+1)/4))))**2))+((x[dim-1]+1)/4)**2)+numpy.sum(Ufun(x,10,100,4));   
    return o;
    
def F13(x): 
    dim=len(x);
    o=.1*((numpy.sin(3*math.pi*x[1]))**2+sum((x[0:dim-2]-1)**2*(1+(numpy.sin(3*math.pi*x[1:dim-1]))**2))+ 
    ((x[dim-1]-1)**2)*(1+(numpy.sin(2*math.pi*x[dim-1]))**2))+numpy.sum(Ufun(x,5,100,4));
    return o;
    
def F14(x): 
     aS=[[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],[-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]];     
     aS=numpy.asarray(aS);
     bS = numpy.zeros(25)
     v=numpy.matrix(x)
     for i in range(0,25):
         H=v-aS[:,i];
         bS[i]=numpy.sum((numpy.power(H,6)));   
     w=[i for i in range(25)]   
     for i in range(0,24):
        w[i]=i+1;
     o=((1./500)+numpy.sum(1./(w+bS)))**(-1);
     return o;  
     
def F15(L):  
    aK=[.1957,.1947,.1735,.16,.0844,.0627,.0456,.0342,.0323,.0235,.0246];
    bK=[.25,.5,1,2,4,6,8,10,12,14,16];
    aK=numpy.asarray(aK);
    bK=numpy.asarray(bK);
    bK = 1/bK;  
    fit=numpy.sum((aK-((L[0]*(bK**2+L[1]*bK))/(bK**2+L[2]*bK+L[3])))**2);
    return fit

def F16(L):  
     o=4*(L[0]**2)-2.1*(L[0]**4)+(L[0]**6)/3+L[0]*L[1]-4*(L[1]**2)+4*(L[1]**4);
     return o

def F17(L):  
    o=(L[1]-(L[0]**2)*5.1/(4*(numpy.pi**2))+5/numpy.pi*L[0]-6)**2+10*(1-1/(8*numpy.pi))*numpy.cos(L[0])+10;
    return o
    
def F18(L):  
    o=(1+(L[0]+L[1]+1)**2*(19-14*L[0]+3*(L[0]**2)-14*L[1]+6*L[0]*L[1]+3*L[1]**2))*(30+(2*L[0]-3*L[1])**2*(18-32*L[0]+12*(L[0]**2)+48*L[1]-36*L[0]*L[1]+27*(L[1]**2)));
    return o
# map the inputs to the function blocks
def F19(L):    
    aH=[[3,10,30],[.1,10,35],[3,10,30],[.1,10,35]];
    aH=numpy.asarray(aH);
    cH=[1,1.2,3,3.2];
    cH=numpy.asarray(cH);
    pH=[[.3689,.117,.2673],[.4699,.4387,.747],[.1091,.8732,.5547],[.03815,.5743,.8828]];
    pH=numpy.asarray(pH);
    o=0;
    for i in range(0,4):
     o=o-cH[i]*numpy.exp(-(numpy.sum(aH[i,:]*((L-pH[i,:])**2))));   
    return o
    

def F20(L):    
    aH=[[10,3,17,3.5,1.7,8],[.05,10,17,.1,8,14],[3,3.5,1.7,10,17,8],[17,8,.05,10,.1,14]];
    aH=numpy.asarray(aH);
    cH=[1,1.2,3,3.2];
    cH=numpy.asarray(cH);
    pH=[[.1312,.1696,.5569,.0124,.8283,.5886],[.2329,.4135,.8307,.3736,.1004,.9991],[.2348,.1415,.3522,.2883,.3047,.6650],[.4047,.8828,.8732,.5743,.1091,.0381]];
    pH=numpy.asarray(pH);
    o=0;
    for i in range(0,4):
     o=o-cH[i]*numpy.exp(-(numpy.sum(aH[i,:]*((L-pH[i,:])**2))));
    return o

def F21(L):
    aSH=[[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]];
    cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5];
    aSH=numpy.asarray(aSH);
    cSH=numpy.asarray(cSH);
    fit=0;
    for i in range(0,4):
      v=numpy.matrix(L-aSH[i,:])
      fit=fit-((v)*(v.T)+cSH[i])**(-1);
    o=fit.item(0);
    return o
  
def F22(L):
    aSH=[[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]];
    cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5];
    aSH=numpy.asarray(aSH);
    cSH=numpy.asarray(cSH);
    fit=0;
    for i in range(0,6):
      v=numpy.matrix(L-aSH[i,:])
      fit=fit-((v)*(v.T)+cSH[i])**(-1);
    o=fit.item(0);
    return o  

def F23(L):
    aSH=[[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]];
    cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5];
    aSH=numpy.asarray(aSH);
    cSH=numpy.asarray(cSH);
    fit=0;
    for i in range(0,9):
      v=numpy.matrix(L-aSH[i,:])
      fit=fit-((v)*(v.T)+cSH[i])**(-1);
    o=fit.item(0);
    return o  

def getFunctionDetails(a):
    
    # [name, lb, ub, dim]
    param = {  "F1": ["F1",-100,100,30],
               "F2" : ["F2",-10,10,30],
               "F3" : ["F3",-100,100,30],
               "F4" : ["F4",-100,100,30] ,
               "F5" : ["F5",-30,30,30],
               "F6" : ["F6",-100,100,30],
               "F7" : ["F7",-1.28,1.28,30],
               "F8" : ["F8",-500,500,30],
               "F9" : ["F9",-5.12,5.12,30],
               "F10" : ["F10",-32,32,30],
               "F11" : ["F11",-600,600,30] ,
               "F12" : ["F12",-50,50,30],
               "F13" : ["F13",-50,50,30],
               "F14" : ["F14",-65.536,65.536,2],
               "F15" : ["F15",-5,5,4],
               "F16" : ["F16",-5,5,2],
               "F17" : ["F17",-5,15,2],
               "F18" : ["F18",-2,2,2] ,
               "F19" : ["F19",0,1,3],
               "F20" : ["F20",0,1,6],
               "F21" : ["F21",0,10,4],
               "F22" : ["F22",0,10,4],
               "F23" : ["F23",0,10,4],
            }
    return param.get(a, "nothing")



