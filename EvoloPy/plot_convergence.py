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
    filename = f"{directory}{optimizer_name}_{objective_func}_convergence.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    return filename
