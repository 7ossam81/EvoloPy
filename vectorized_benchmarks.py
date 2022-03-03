# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris

Vectorized by AmirPouya Hemmasian, Dec 2021

Each function receives a NxD matrix (numpy array) as input.
Each row represents a point in the D-dimensional search space
The output is a numpy array of length N containing the fitness
value of the N points respectively.
"""

import numpy as np




# define the function blocks

# helper function:
def Ufun(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))

# Unimodal High-Dimensional Functions:

def F1(x):
    return np.sum(x**2, axis=1)

def F2(x):
    absx = np.abs(x)
    return np.sum(absx, axis=1) + np.prod(absx, axis=1)

def F3(x):
    return np.sum(np.cumsum(x, axis=1)**2, axis=1)

def F4(x):
    return np.max(np.abs(x), axis=1)

def F5(x):
    return np.sum( 100*(x[:, 1:] - x[:, :-1]**2)**2 + \
                   (x[:, :-1] - 1)**2, axis=1)

def F6(x):
    return np.sum((x + 0.5)**2, axis=1)

def F7(x):
    m, n = x.shape
    w = np.arange(1, n+1)
    return np.sum(w*x**4, axis=1) + np.random.rand(m)

# Multimodal High-Dimensional Functions:

def F8(x):
    return np.sum(-x*np.sin(np.abs(x)**0.5), axis=1)

def F9(x):
   return np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=1) + 10*x.shape[1]

def F10(x):
    return -20*np.exp(-0.2 * np.mean(x**2, axis=1)**0.5) \
        - np.exp(np.mean(np.cos(2*np.pi*x), axis=1)) + 20 + np.exp(1)

def F11(x):
    w = np.arange(1, x.shape[1]+1)
    return np.sum(x**2, axis=1)/4000 - \
        np.prod( np.cos(x*w**-0.5) , axis=1) + 1

def F12(x):
    y = 1 + (1 + x)/4
    return np.pi/x.shape[1]*(
        10*np.sin(np.pi*y[:, 0])**2
        + np.sum( (y[:, :-1] - 1)**2 *(1 + 10*np.sin(np.pi*y[:, 1:])**2), axis=1)
        + (y[:, -1] - 1)**2 ) + np.sum( Ufun(x, 10, 100, 4), axis=1)

def F13(x):
    return 0.1*( np.sin(3*np.pi*x[:,0])**2
        + np.sum( (x[:,:-1] - 1)**2 *(1 + np.sin(3*np.pi*x[:, 1:])**2 ), axis=1)
        + (x[:, -1] - 1)**2 * ( 1 + np.sin(2*np.pi*x[:, -1])**2 ) ) \
        + np.sum(Ufun(x, 5, 100, 4), axis=1)

# Multimodal Low-Dimensional Functions:

a14 = np.array([[-32, -16, 0, 16, 32,
                 -32, -16, 0, 16, 32,
                 -32, -16, 0, 16, 32,
                 -32, -16, 0, 16, 32,
                 -32, -16, 0, 16, 32],
                 [-32, -32, -32, -32, -32,
                  -16, -16, -16, -16, -16,
                  0, 0, 0, 0, 0,
                  16, 16, 16, 16, 16,
                  32, 32, 32, 32, 32]])[None, :, :]
                  
def F14(x):
    v = x[:, :, None]
    s = 1/(np.arange(1,26) + np.sum((v-a14)**6, axis=1))
    return 1/( 1/500 + np.sum(s, axis=-1) )
    
    
a15 = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
                0.0456, 0.0342, 0.0323, 0.0235, 0.0246,])
b15 = 1/np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])

def F15(L):
    L = L[:,:,None]
    return np.sum(
        (a15 -(L[:,0,:]*(b15 **2+L[:,1,:]*b15))/(b15**2+L[:,2,:]*b15+L[:,3,:]))**2,
    axis=-1)

def F16(L):
    return (
        4 * (L[:,0] ** 2)
        - 2.1 * (L[:,0] ** 4)
        + (L[:,0] ** 6) / 3
        + L[:,0] * L[:,1]
        - 4 * (L[:,1] ** 2)
        + 4 * (L[:,1] ** 4)
    )    

def F17(L):
    return (
        (L[:,1] - L[:,0]**2 * 5.1/(4*np.pi**2) + 5/np.pi*L[:,0] -6)**2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(L[:,0]) + 10
        )

def F18(L):
    return (1
        + (L[:,0] + L[:,1] + 1) ** 2
        * (
            19
            - 14 * L[:,0]
            + 3 * L[:,0] ** 2
            - 14 * L[:,1]
            + 6 * L[:,0] * L[:,1]
            + 3 * L[:,1] ** 2
        )
    ) * (30+ (2 * L[:,0] - 3 * L[:,1]) ** 2
        * (18
            - 32 * L[:,0]
            + 12 * L[:,0] ** 2
            + 48 * L[:,1]
            - 36 * L[:,0] * L[:,1]
            + 27 * L[:,1] ** 2
        )
    )

a19 = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]).T
c19 = np.array([1, 1.2, 3, 3.2])
p19 = np.array([[0.3689, 0.117, 0.2673],
                [0.4699, 0.4387, 0.747],
                [0.1091, 0.8732, 0.5547],
                [0.03815, 0.5743, 0.8828]]).T
                
def F19(L):
    L = L[:,:, None]
    return -np.sum(c19*np.exp(-np.sum(a19*(L-p19)**2,axis=1)), axis=1)
    
a20 = np.array([[10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14]]).T
c20 = np.array([1, 1.2, 3, 3.2])
p20 = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]).T    

def F20(L):
    L = L[:,:, None]
    return -np.sum(c20*np.exp(-np.sum(a20*(L-p20)**2,axis=1)), axis=1)
    
a20s = np.array([
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6]]).T
c20s = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

def F21(L):
    L = L[:,:,None]
    return -np.sum((np.sum((L-a20s[:,:5])**2, axis=1) + c20s[:5])**-1, axis=1)
    
def F22(L):
    L = L[:,:,None]
    return -np.sum((np.sum((L-a20s[:,:7])**2, axis=1) + c20s[:7])**-1, axis=1)
    
def F23(L):
    L = L[:,:,None]
    return -np.sum((np.sum((L-a20s)**2, axis=1) + c20s)**-1, axis=1)
    
    
    
param = {
        "F1": ["F1", -100, 100, 30],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -100, 100, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4]}

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    return param.get(a, "nothing")

'''
# ESAs space mission design benchmarks https://www.esa.int/gsp/ACT/projects/gtop/
from fcmaes.astro import (
    MessFull,
    Messenger,
    Gtoc1,
    Cassini1,
    Cassini2,
    Rosetta,
    Tandem,
    Sagas,
)
def Ca1(x):
    return Cassini1().fun(x)
def Ca2(x):
    return Cassini2().fun(x)
def Ros(x):
    return Rosetta().fun(x)
def Tan(x):
    return Tandem(5).fun(x)
def Sag(x):
    return Sagas().fun(x)
def Mef(x):
    return MessFull().fun(x)
def Mes(x):
    return Messenger().fun(x)
def Gt1(x):
    return Gtoc1().fun(x)

param = {
        "F1": ["F1", -100, 100, 30],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -100, 100, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],
        "Ca1": [
            "Ca1",
            Cassini1().bounds.lb,
            Cassini1().bounds.ub,
            len(Cassini1().bounds.lb),
        ],
        "Ca2": [
            "Ca2",
            Cassini2().bounds.lb,
            Cassini2().bounds.ub,
            len(Cassini2().bounds.lb),
        ],
        "Gt1": ["Gt1", Gtoc1().bounds.lb, Gtoc1().bounds.ub, len(Gtoc1().bounds.lb)],
        "Mes": [
            "Mes",
            Messenger().bounds.lb,
            Messenger().bounds.ub,
            len(Messenger().bounds.lb),
        ],
        "Mef": [
            "Mef",
            MessFull().bounds.lb,
            MessFull().bounds.ub,
            len(MessFull().bounds.lb),
        ],
        "Sag": ["Sag", Sagas().bounds.lb, Sagas().bounds.ub, len(Sagas().bounds.lb)],
        "Tan": [
            "Tan",
            Tandem(5).bounds.lb,
            Tandem(5).bounds.ub,
            len(Tandem(5).bounds.lb),
        ],
        "Ros": [
            "Ros",
            Rosetta().bounds.lb,
            Rosetta().bounds.ub,
            len(Rosetta().bounds.lb),
        ],
    }

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    return param.get(a, "nothing")
    
'''