"""
Parallel processing utilities for EvoloPy.

This module provides functions for parallel execution of optimization algorithms,
hardware detection, and optimal configuration for parallel processing.
"""

import os
import numpy as np
import multiprocessing
from typing import Callable, Dict, List, Union, Tuple, Any
import concurrent.futures
import platform
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def detect_hardware() -> Dict[str, Any]:
    """
    Detect available hardware resources for parallel processing.
    
    Returns:
        Dict[str, Any]: Dictionary containing hardware information:
            - cpu_count: Number of CPU cores
            - cpu_threads: Number of CPU threads
            - ram_gb: Available RAM in GB
            - gpu_available: Whether CUDA GPU is available
            - gpu_count: Number of CUDA GPUs
            - gpu_names: List of GPU names
            - gpu_memory: List of GPU memory in GB
    """
    # CPU detection
    cpu_count = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    
    # Memory detection
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # GPU detection
    gpu_available = False
    gpu_count = 0
    gpu_names = []
    gpu_memory = []
    
    # Check for CUDA GPUs via PyTorch
    if TORCH_AVAILABLE:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_names.append(torch.cuda.get_device_name(i))
                gpu_memory.append(torch.cuda.get_device_properties(i).total_memory / (1024**3))
    
    return {
        'cpu_count': cpu_count,
        'cpu_threads': cpu_threads,
        'ram_gb': ram_gb,
        'gpu_available': gpu_available,
        'gpu_count': gpu_count,
        'gpu_names': gpu_names,
        'gpu_memory': gpu_memory
    }


def get_optimal_process_count(parallel_backend: str = 'multiprocessing') -> int:
    """
    Determine the optimal number of processes for parallel execution.
    
    Parameters:
        parallel_backend (str): Parallel processing backend ('multiprocessing', 'cuda', 'auto')
    
    Returns:
        int: Optimal number of processes
    """
    # If GPU backend is requested but not available, fall back to CPU
    if parallel_backend == 'cuda' and not (TORCH_AVAILABLE and torch.cuda.is_available()):
        parallel_backend = 'multiprocessing'
    
    # If auto is specified, use GPU if available, otherwise CPU
    if parallel_backend == 'auto':
        if TORCH_AVAILABLE and torch.cuda.is_available():
            parallel_backend = 'cuda'
        else:
            parallel_backend = 'multiprocessing'
    
    # For CUDA, use the number of GPUs (or 1 if multiple streams not supported)
    if parallel_backend == 'cuda':
        return torch.cuda.device_count()
    
    # For CPU multiprocessing, leave one core free for system processes
    cpu_count = psutil.cpu_count(logical=False)
    if cpu_count is None:
        cpu_count = os.cpu_count() or 2
    
    # Use n-1 cores, but at least 1
    return max(1, cpu_count - 1)


def run_optimizer_parallel(
    optimizer_func: Callable, 
    objf: Callable, 
    lb: Union[float, List[float]], 
    ub: Union[float, List[float]], 
    dim: int, 
    PopSize: int, 
    iters: int, 
    num_runs: int,
    parallel_backend: str = 'multiprocessing',
    num_processes: int = None
) -> List[Any]:
    """
    Run an optimizer multiple times in parallel.
    
    Parameters:
        optimizer_func (Callable): The optimizer function to run
        objf (Callable): The objective function to optimize
        lb (float or List[float]): Lower bounds
        ub (float or List[float]): Upper bounds
        dim (int): Problem dimension
        PopSize (int): Population size
        iters (int): Number of iterations
        num_runs (int): Number of independent runs
        parallel_backend (str): Parallel processing backend ('multiprocessing', 'cuda', 'auto')
        num_processes (int, optional): Number of processes to use (None for auto)
    
    Returns:
        List[Any]: List of solution objects from each run
    """
    if num_processes is None:
        num_processes = get_optimal_process_count(parallel_backend)
    
    # For GPU backend
    if parallel_backend == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
        # Create a function to run on a specific GPU
        def run_on_gpu(run_id, gpu_id):
            # Set the CUDA device
            torch.cuda.set_device(gpu_id % torch.cuda.device_count())
            # Run the optimizer
            return optimizer_func(objf, lb, ub, dim, PopSize, iters)
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for i in range(num_runs):
                futures.append(executor.submit(run_on_gpu, i, i % num_processes))
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        return results
    
    # For CPU multiprocessing backend
    elif parallel_backend in ['multiprocessing', 'auto']:
        # Create a function that just executes the optimizer with fixed parameters
        def run_optimizer_once(run_id):
            return optimizer_func(objf, lb, ub, dim, PopSize, iters)
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for i in range(num_runs):
                futures.append(executor.submit(run_optimizer_once, i))
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        return results
    
    else:
        raise ValueError(f"Unknown parallel backend: {parallel_backend}")


def run_population_fitness_parallel(
    objf: Callable, 
    population: np.ndarray, 
    parallel_backend: str = 'multiprocessing',
    num_processes: int = None
) -> np.ndarray:
    """
    Evaluate fitness of a population in parallel.
    
    Parameters:
        objf (Callable): The objective function
        population (np.ndarray): Population matrix where each row is an individual
        parallel_backend (str): Parallel processing backend ('multiprocessing', 'cuda', 'auto')
        num_processes (int, optional): Number of processes to use (None for auto)
    
    Returns:
        np.ndarray: Array of fitness values
    """
    if num_processes is None:
        num_processes = get_optimal_process_count(parallel_backend)
    
    # For GPU backend with PyTorch
    if parallel_backend == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
        # Convert population to PyTorch tensor
        device = torch.device("cuda")
        population_tensor = torch.tensor(population, device=device, dtype=torch.float32)
        
        # Vectorized implementation if possible
        try:
            # Try vectorized evaluation (if objf supports tensor input)
            fitness = objf(population_tensor)
            return fitness.cpu().numpy()
        except:
            # Fall back to iterative evaluation
            fitness = torch.zeros(len(population), device=device)
            for i in range(len(population)):
                fitness[i] = objf(population_tensor[i])
            return fitness.cpu().numpy()
    
    # For CPU multiprocessing backend
    elif parallel_backend in ['multiprocessing', 'auto']:
        # Create a worker function to evaluate individual fitness
        def evaluate_individual(individual):
            return objf(individual)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            fitness = list(executor.map(evaluate_individual, population))
        
        return np.array(fitness)
    
    else:
        raise ValueError(f"Unknown parallel backend: {parallel_backend}") 