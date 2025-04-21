# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:43:09 2017

@author: Raneem
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def run(optimizer_name, objective_func, convergence, directory=""):
    """
    Generate and save boxplot for final results of multiple runs.
    
    Parameters:
        optimizer_name: Name of the optimizer
        objective_func: Name of the objective function
        convergence: List of convergence data for each run
        directory: Directory to save the plot (with trailing slash)
    """
    # Ensure the directory exists
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create figure
    plt.figure()
    
    # Extract final fitness values from each run
    final_fitness = [conv[-1] for conv in convergence]
    
    # Create boxplot
    bp = plt.boxplot(final_fitness, patch_artist=True)
    
    # Set colors
    plt.setp(bp['boxes'], color='blue', facecolor='lightblue')
    plt.setp(bp['medians'], color='red')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], marker='o', markerfacecolor='red', markersize=6)
    
    # Add labels and title
    plt.ylabel('Final Fitness Value')
    plt.title(f'Boxplot - {optimizer_name} on {objective_func}')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    # Add statistics as text
    stats = {
        'Min': np.min(final_fitness),
        'Max': np.max(final_fitness),
        'Mean': np.mean(final_fitness),
        'Median': np.median(final_fitness),
        'Std': np.std(final_fitness)
    }
    
    stat_text = '\n'.join([f'{k}: {v:.6g}' for k, v in stats.items()])
    plt.figtext(0.65, 0.7, stat_text, fontsize=9, 
                bbox=dict(facecolor='lightgray', alpha=0.5))
    
    # Save the figure
    filename = f"{directory}{optimizer_name}_{objective_func}_boxplot.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    return filename

def generateBoxPlot(optimizers, objectives, fitness_values, directory=""):
    """
    Generate and save boxplot for comparing multiple optimizers on multiple objectives.
    
    Parameters:
        optimizers: List of optimizer names
        objectives: List of objective function names
        fitness_values: List of lists of fitness values for each optimizer-objective pair
        directory: Directory to save the plot (with trailing slash)
    """
    # Ensure the directory exists
    Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create a boxplot for each objective function
    for i, obj in enumerate(objectives):
        plt.figure(figsize=(10, 6))
        
        # Extract data for this objective
        data = []
        labels = []
        
        for j, opt in enumerate(optimizers):
            if isinstance(fitness_values[j], list):
                data.append(fitness_values[j])
                labels.append(opt)
        
        # Create boxplot if we have data
        if data:
            bp = plt.boxplot(data, patch_artist=True, labels=labels)
            
            # Set colors - use different colors for each optimizer
            colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow', 
                     'lightcyan', 'lightsalmon', 'lightcoral', 'lightgray']
            
            for box, color in zip(bp['boxes'], colors[:len(data)]):
                box.set(color='blue', facecolor=color)
            
            plt.setp(bp['medians'], color='red')
            plt.setp(bp['whiskers'], color='black')
            plt.setp(bp['fliers'], marker='o', markerfacecolor='red', markersize=4)
            
            # Add labels and title
            plt.ylabel('Final Fitness Value')
            plt.title(f'Algorithm Comparison on {obj}')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save the figure
            filename = f"{directory}comparison_{obj}_boxplot.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
    
    return directory
