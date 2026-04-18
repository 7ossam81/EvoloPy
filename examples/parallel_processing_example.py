#!/usr/bin/env python
"""
Example script demonstrating EvoloPy's parallel processing feature.

This script compares execution times of sequential and parallel processing
for running multiple optimization algorithms on benchmark functions.
"""

import sys
import os

# Get the absolute path to the EvoloPy directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the EvoloPy directory to the Python path
sys.path.append(base_dir)

import time
import numpy as np
import matplotlib.pyplot as plt
from EvoloPy.api import run_optimizer, run_multiple_optimizers, get_hardware_info

def run_benchmark(enable_parallel=False, backend="auto"):
    """Run a benchmark comparing different optimizers with and without parallel processing."""
    
    # Define parameters
    optimizers = ["PSO", "GWO", "MVO", "WOA"]
    functions = ["F1", "F5", "F10"]
    dim = 20
    population_size = 50
    iterations = 100
    num_runs = 10
    
    # Print configuration
    print("\n" + "="*50)
    print(f"Running benchmark with parallel={enable_parallel}, backend={backend if enable_parallel else 'N/A'}")
    print(f"Optimizers: {optimizers}")
    print(f"Functions: {functions}")
    print(f"Dimension: {dim}, Population size: {population_size}")
    print(f"Iterations: {iterations}, Number of runs: {num_runs}")
    print("="*50)
    
    # Store execution times
    execution_times = {}
    
    # Run each optimizer on each function
    for optimizer in optimizers:
        execution_times[optimizer] = {}
        
        for function in functions:
            print(f"\nRunning {optimizer} on {function}...")
            
            # Measure execution time
            start_time = time.time()
            
            # Run the optimizer
            result = run_optimizer(
                optimizer=optimizer,
                objective_func=function,
                dim=dim,
                population_size=population_size,
                iterations=iterations,
                num_runs=num_runs,
                enable_parallel=enable_parallel,
                parallel_backend=backend
            )
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            # Store times
            execution_times[optimizer][function] = {
                'optimizer_time': result['execution_time'],
                'total_time': total_time,
                'best_fitness': result['best_fitness']
            }
            
            # Print results
            print(f"  Best fitness: {result['best_fitness']}")
            print(f"  Optimizer execution time: {result['execution_time']:.2f} seconds")
            print(f"  Total wall time: {total_time:.2f} seconds")
    
    return execution_times

def plot_results(sequential_times, parallel_times):
    """Plot comparison of sequential vs parallel execution times."""
    
    optimizers = list(sequential_times.keys())
    functions = list(sequential_times[optimizers[0]].keys())
    
    # Create figure
    fig, axes = plt.subplots(len(functions), 1, figsize=(12, 5*len(functions)))
    if len(functions) == 1:
        axes = [axes]
    
    for i, function in enumerate(functions):
        ax = axes[i]
        
        # Prepare data
        seq_times = [sequential_times[opt][function]['total_time'] for opt in optimizers]
        par_times = [parallel_times[opt][function]['total_time'] for opt in optimizers]
        speedups = [seq_times[j]/par_times[j] for j in range(len(optimizers))]
        
        # Set width of bars
        bar_width = 0.35
        
        # Set position of bars
        r1 = np.arange(len(optimizers))
        r2 = [x + bar_width for x in r1]
        
        # Create bars
        ax.bar(r1, seq_times, width=bar_width, label='Sequential', color='blue')
        ax.bar(r2, par_times, width=bar_width, label='Parallel', color='green')
        
        # Add speedup text above bars
        for j, speedup in enumerate(speedups):
            ax.text(r2[j], par_times[j] + 0.1, f'{speedup:.1f}x', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Add labels and title
        ax.set_xlabel('Optimizer')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'Execution Time Comparison for {function}')
        ax.set_xticks([r + bar_width/2 for r in range(len(optimizers))])
        ax.set_xticklabels(optimizers)
        ax.legend()
        
        # Set y-axis limit to make room for text
        ax.set_ylim(0, max(max(seq_times), max(par_times)) * 1.2)
    
    plt.tight_layout()
    plt.savefig('parallel_benchmark_results.png')
    plt.show()

if __name__ == "__main__":
    # Display hardware information
    try:
        hw_info = get_hardware_info()
        print("Hardware Information:")
        print(f"CPU cores: {hw_info['cpu_count']}")
        print(f"CPU threads: {hw_info['cpu_threads']}")
        print(f"RAM: {hw_info['ram_gb']:.2f} GB")
        
        if hw_info['gpu_available']:
            print(f"CUDA GPUs available: {hw_info['gpu_count']}")
            for i, (name, mem) in enumerate(zip(hw_info['gpu_names'], hw_info['gpu_memory'])):
                print(f"  GPU {i}: {name} ({mem:.2f} GB)")
            
            # Use CUDA backend if GPU is available
            parallel_backend = "cuda"
        else:
            print("CUDA GPUs: None detected")
            # Fall back to multiprocessing if no GPU
            parallel_backend = "multiprocessing"
    except:
        print("Hardware detection not available. Using multiprocessing backend.")
        parallel_backend = "multiprocessing"
    
    # Run sequential benchmark
    sequential_times = run_benchmark(enable_parallel=False)
    
    # Run parallel benchmark
    parallel_times = run_benchmark(enable_parallel=True, backend=parallel_backend)
    
    # Plot results
    plot_results(sequential_times, parallel_times) 