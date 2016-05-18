# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris
"""

import numpy
import math

# define the function blocks
def F1(x):
    s=numpy.sum(x**2);
    return s

def F6(x):    
    s=sum(abs((x+.5))**2);
    return s
# map the inputs to the function blocks

def getFunctionDetails(a):
    
    # [name, lb, ub, dim]
    param = {0: ["F1","-100","100","30"],
               1 : ["F2","-100","100","30"],
               2 : ["F3","-100","100","30"],
               3 : ["F4","-100","100","30"] ,
               4 : ["F5","-100","100","100"],
               5 : ["F6","-100","100","100"],
               6 : ["F7","-100","100","30"],
               7 : ["F8","-100","100","30"],
           }
    return param.get(a, "nothing")



