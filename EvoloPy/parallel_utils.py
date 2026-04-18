"""
Parallel processing utilities for EvoloPy.

This module provides functions for parallel execution of optimization algorithms,
hardware detection, and optimal configuration for parallel processing.
"""

import os
import numpy as np
from typing import Callable, Dict, List, Union, Any
import concurrent.futures
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


# =========================================================
# Top-level worker functions (must be module-level so they
# are picklable on Windows / macOS / Linux multiprocessing)
# =========================================================

def _run_optimizer_once_worker(args):
    """
    Worker for one optimizer run using CPU multiprocessing.
    Must be top-level to be picklable on Windows.
    """
    optimizer_func, objf, lb, ub, dim, PopSize, iters = args
    return optimizer_func(objf, lb, ub, dim, PopSize, iters)


def _run_optimizer_on_gpu_worker(args):
    """
    Worker for one optimizer run using GPU threads.
    Thread-based, so pickling is not the same issue, but we keep it top-level
    for consistency.
    """
    optimizer_func, objf, lb, ub, dim, PopSize, iters, gpu_id = args

    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but CUDA is not available.")

    torch.cuda.set_device(gpu_id % torch.cuda.device_count())
    return optimizer_func(objf, lb, ub, dim, PopSize, iters)


def _evaluate_individual_worker(args):
    """
    Worker for evaluating one individual's fitness using CPU multiprocessing.
    Must be top-level to be picklable on Windows.
    """
    objf, individual = args
    return objf(individual)


# =========================================================
# Hardware utilities
# =========================================================

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
        parallel_backend (str): Parallel processing backend
                                ('multiprocessing', 'cuda', 'auto')

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

    # For CUDA, use the number of GPUs
    if parallel_backend == 'cuda':
        return max(1, torch.cuda.device_count())

    # For CPU multiprocessing, leave one core free for system processes
    cpu_count = psutil.cpu_count(logical=False)
    if cpu_count is None:
        cpu_count = os.cpu_count() or 2

    return max(1, cpu_count - 1)


# =========================================================
# Parallel optimizer execution
# =========================================================

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
        parallel_backend (str): Parallel processing backend
                                ('multiprocessing', 'cuda', 'auto')
        num_processes (int, optional): Number of processes to use (None for auto)

    Returns:
        List[Any]: List of solution objects from each run
    """
    # Resolve auto backend
    if parallel_backend == 'auto':
        if TORCH_AVAILABLE and torch.cuda.is_available():
            parallel_backend = 'cuda'
        else:
            parallel_backend = 'multiprocessing'

    if num_processes is None:
        num_processes = get_optimal_process_count(parallel_backend)

    # GPU backend
    if parallel_backend == 'cuda':
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            raise ValueError("CUDA backend requested but CUDA is not available.")

        task_args = [
            (optimizer_func, objf, lb, ub, dim, PopSize, iters, i % max(1, torch.cuda.device_count()))
            for i in range(num_runs)
        ]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(_run_optimizer_on_gpu_worker, args) for args in task_args]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results

    # CPU multiprocessing backend
    elif parallel_backend == 'multiprocessing':
        task_args = [
            (optimizer_func, objf, lb, ub, dim, PopSize, iters)
            for _ in range(num_runs)
        ]

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(_run_optimizer_once_worker, args) for args in task_args]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results

    else:
        raise ValueError(f"Unknown parallel backend: {parallel_backend}")


# =========================================================
# Parallel fitness evaluation
# =========================================================

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
        parallel_backend (str): Parallel processing backend
                                ('multiprocessing', 'cuda', 'auto')
        num_processes (int, optional): Number of processes to use (None for auto)

    Returns:
        np.ndarray: Array of fitness values
    """
    # Resolve auto backend
    if parallel_backend == 'auto':
        if TORCH_AVAILABLE and torch.cuda.is_available():
            parallel_backend = 'cuda'
        else:
            parallel_backend = 'multiprocessing'

    if num_processes is None:
        num_processes = get_optimal_process_count(parallel_backend)

    # GPU backend with PyTorch
    if parallel_backend == 'cuda':
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            raise ValueError("CUDA backend requested but CUDA is not available.")

        device = torch.device("cuda")
        population_tensor = torch.tensor(population, device=device, dtype=torch.float32)

        try:
            # Try vectorized evaluation
            fitness = objf(population_tensor)
            return fitness.cpu().numpy()
        except Exception:
            # Fall back to iterative evaluation
            fitness = torch.zeros(len(population), device=device)
            for i in range(len(population)):
                fitness[i] = objf(population_tensor[i])
            return fitness.cpu().numpy()

    # CPU multiprocessing backend
    elif parallel_backend == 'multiprocessing':
        task_args = [(objf, individual) for individual in population]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            fitness = list(executor.map(_evaluate_individual_worker, task_args))

        return np.array(fitness)

    else:
        raise ValueError(f"Unknown parallel backend: {parallel_backend}")