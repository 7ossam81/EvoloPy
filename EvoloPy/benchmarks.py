# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris

updated on Sun Feb 9 06:05:50 2025
"""

import numpy as np
import math
from functools import reduce
from typing import Union, List, Tuple, Callable

# Define the function blocks - optimized versions
def prod(it):
    """Optimized product function using reduce"""
    return reduce(lambda x, y: x * y, it, 1)

def Ufun(x, a, k, m):
    """Penalty function"""
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))

def F1(x):
    """Sphere function"""
    return np.sum(x ** 2)

def F2(x):
    """Sum of absolute and product of absolute values"""
    return np.sum(np.abs(x)) + prod(np.abs(x))

def F3(x):
    """Sum of squared sums"""
    result = 0
    for i in range(1, len(x) + 1):
        result += (np.sum(x[0:i])) ** 2
    return result

def F4(x):
    """Maximum absolute value"""
    return np.max(np.abs(x))

def F5(x):
    """Rosenbrock function"""
    dim = len(x)
    return np.sum(100 * (x[1:dim] - (x[0:dim-1] ** 2)) ** 2 + (x[0:dim-1] - 1) ** 2)

def F6(x):
    """Shifted absolute square function"""
    return np.sum(np.abs(x + 0.5) ** 2)

def F7(x):
    """Sum of weighted fourth powers with noise"""
    dim = len(x)
    w = np.arange(1, dim + 1)
    return np.sum(w * (x ** 4)) + np.random.uniform(0, 1)

def F8(x):
    """Sum of negative products of sine of square root of absolute value"""
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F9(x):
    """Rastrigin function"""
    dim = len(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim

def F10(x):
    """Ackley function"""
    dim = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim)) - 
            np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 
            20 + np.exp(1))

def F11(x):
    """Griewank function"""
    dim = len(x)
    w = np.arange(1, dim + 1)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(w))) + 1

def F12(x):
    """Penalized function 1"""
    dim = len(x)
    y = 1 + (x + 1) / 4
    
    term1 = 10 * (np.sin(np.pi * y[0])) ** 2
    term2 = np.sum((y[0:dim-1] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * y[1:dim])) ** 2))
    term3 = (y[dim-1] - 1) ** 2
    
    pi_n = np.pi * dim
    return (pi_n / 10) * (term1 + term2 + term3) + np.sum(Ufun(x, 10, 100, 4))

def F13(x):
    """Penalized function 2"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    term1 = (np.sin(3 * np.pi * x[:, 0])) ** 2
    term2 = np.sum((x[:, :-1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[:, 1:])) ** 2), axis=1)
    term3 = ((x[:, -1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[:, -1])) ** 2)
    
    result = 0.1 * (term1 + term2 + term3) + np.sum(Ufun(x, 5, 100, 4))
    return result

def F14(x):
    aS = [
        [
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
        ],
        [
            -32, -32, -32, -32, -32,
            -16, -16, -16, -16, -16,
            0, 0, 0, 0, 0,
            16, 16, 16, 16, 16,
            32, 32, 32, 32, 32,
        ],
    ]
    aS = np.asarray(aS)
    bS = np.zeros(25)
    v = np.array(x)
    for i in range(0, 25):
        H = v - aS[:, i]
        bS[i] = np.sum((np.power(H, 6)))
    w = np.arange(1, 26)
    o = ((1.0 / 500) + np.sum(1.0 / (w + bS))) ** (-1)
    return o


def F15(L):
    aK = [
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ]
    bK = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    aK = np.asarray(aK)
    bK = np.asarray(bK)
    bK = 1 / bK
    fit = np.sum(
        (aK - ((L[0] * (bK ** 2 + L[1] * bK)) / (bK ** 2 + L[2] * bK + L[3]))) ** 2
    )
    return fit


def F16(L):
    o = (
        4 * (L[0] ** 2)
        - 2.1 * (L[0] ** 4)
        + (L[0] ** 6) / 3
        + L[0] * L[1]
        - 4 * (L[1] ** 2)
        + 4 * (L[1] ** 4)
    )
    return o


def F17(L):
    o = (
        (L[1] - (L[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * L[0] - 6)
        ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(L[0])
        + 10
    )
    return o


def F18(L):
    o = (
        1
        + (L[0] + L[1] + 1) ** 2
        * (
            19
            - 14 * L[0]
            + 3 * (L[0] ** 2)
            - 14 * L[1]
            + 6 * L[0] * L[1]
            + 3 * L[1] ** 2
        )
    ) * (
        30
        + (2 * L[0] - 3 * L[1]) ** 2
        * (
            18
            - 32 * L[0]
            + 12 * (L[0] ** 2)
            + 48 * L[1]
            - 36 * L[0] * L[1]
            + 27 * (L[1] ** 2)
        )
    )
    return o


# map the inumpyuts to the function blocks
def F19(L):
    aH = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
    aH = np.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = np.asarray(cH)
    pH = [
        [0.3689, 0.117, 0.2673],
        [0.4699, 0.4387, 0.747],
        [0.1091, 0.8732, 0.5547],
        [0.03815, 0.5743, 0.8828],
    ]
    pH = np.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F20(L):
    aH = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
    aH = np.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = np.asarray(cH)
    pH = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
    pH = np.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F21(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(5):
        v = np.array(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F22(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(7):
        v = np.array(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F23(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(10):
        v = np.array(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o

# Ackley function (commonly used in optimization)
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

# Rosenbrock function (tests convergence)
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Rastrigin function (tests local minima avoidance)
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Griewank function (tests exploration)
def griewank(x):
    part1 = np.sum(x**2) / 4000
    part2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return part1 - part2 + 1

def getFunctionDetails(a):
    # [name, lb, ub, dim]
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
        "ackley": ["ackley", -32.768, 32.768, 30],  # Ackley function
        "rosenbrock": ["rosenbrock", -5, 10, 30],  # Rosenbrock function
        "rastrigin": ["rastrigin", -5.12, 5.12, 30],  # Rastrigin function
        "griewank": ["griewank", -600, 600, 30],  # Griewank function
    }
    return param.get(a, "nothing")