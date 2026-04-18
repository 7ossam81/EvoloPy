# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:43:09 2017

@author: Raneem
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def run(convergence, optimizer_name, objective_func, directory=""):
    """
    Generate and save convergence plots for an optimizer.
    
    Parameters:
        convergence: List of convergence values for each run
        optimizer_name: Name of the optimizer
        objective_func: Name of the objective function
        directory: Directory to save the plot (with trailing slash)
    """
    # Ensure the directory exists
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create figure
    plt.figure()
    
    # If we have multiple runs, calculate mean and plot with confidence intervals
    if isinstance(convergence, list) and len(convergence) > 1:
        # Convert list of lists to numpy array for easier manipulation
        convergence_arr = np.array(convergence)
        
        # Calculate mean and std of convergence across runs
        mean_convergence = np.mean(convergence_arr, axis=0)
        std_convergence = np.std(convergence_arr, axis=0)
        
        # Plot mean convergence
        x = np.arange(1, len(mean_convergence) + 1)
        plt.plot(x, mean_convergence, 'b-', label=f'{optimizer_name} mean')
        
        # Add confidence interval (mean ± std)
        plt.fill_between(x, mean_convergence - std_convergence, 
                        mean_convergence + std_convergence, 
                        color='b', alpha=0.2, label='Standard Deviation')
    else:
        # If single run, plot the convergence directly
        if isinstance(convergence, list) and len(convergence) == 1:
            convergence = convergence[0]
        
        x = np.arange(1, len(convergence) + 1)
        plt.plot(x, convergence, 'b-', label=optimizer_name)
    
    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'Convergence - {optimizer_name} on {objective_func}')
    plt.legend()
    
    # Set log scale for y-axis if values are all positive and vary by orders of magnitude
    if np.all(np.array(convergence) > 0) and np.max(convergence) / np.min(convergence) > 100:
        plt.yscale('log')
    
    # Save the figure
    filename = f"{directory}convergence.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    return filename

def run_comparison_avg(optimizers, objective_func, all_convergence_data, directory=""):
    """
    Generate and save comparative average convergence plot for multiple optimizers on one function.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    plt.figure()

    for idx, optimizer_name in enumerate(optimizers):
        convergence = all_convergence_data[idx]

        if isinstance(convergence, list) and len(convergence) > 1:
            convergence_arr = np.array(convergence)
            mean_convergence = np.mean(convergence_arr, axis=0)
            x = np.arange(1, len(mean_convergence) + 1)
            plt.plot(x, mean_convergence, label=optimizer_name)
        else:
            if isinstance(convergence, list) and len(convergence) == 1:
                convergence = convergence[0]

            x = np.arange(1, len(convergence) + 1)
            plt.plot(x, convergence, label=optimizer_name)

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'Average Convergence Comparison on {objective_func}')
    plt.legend()

    filename = f"{directory}convergence_avg.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    return filename

def run_comparison_best(optimizers, objective_func, all_convergence_data, directory=""):
    """
    Generate and save comparative best-run convergence plot for multiple optimizers on one function.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    plt.figure()

    for idx, optimizer_name in enumerate(optimizers):
        convergence_runs = all_convergence_data[idx]

        if isinstance(convergence_runs, list) and len(convergence_runs) > 1:
            best_run = min(convergence_runs, key=lambda conv: conv[-1])
        else:
            best_run = convergence_runs[0] if isinstance(convergence_runs, list) else convergence_runs

        x = np.arange(1, len(best_run) + 1)
        plt.plot(x, best_run, label=optimizer_name)

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'Best Convergence Comparison on {objective_func}')
    plt.legend()

    filename = f"{directory}convergence_best.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    return filename