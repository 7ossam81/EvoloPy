"""
Nature-inspired optimization algorithms implemented in EvoloPy.

This module provides direct access to all optimization algorithms:

Examples:
    >>> from EvoloPy.optimizers import PSO, GWO
    >>> result_pso = PSO.PSO(objective_function, lb=-10, ub=10, dim=5, PopSize=30, iters=50)
    >>> result_gwo = GWO.GWO(objective_function, lb=-10, ub=10, dim=5, PopSize=30, iters=50)
    
    # Or if you want to dynamically select an optimizer:
    >>> from EvoloPy.optimizers import optimizer_map
    >>> optimizer_func = optimizer_map["PSO"]
    >>> result = optimizer_func(objective_function, lb=-10, ub=10, dim=5, PopSize=30, iters=50)
"""

import importlib
from typing import Dict, Callable, Any

# Import each optimizer module dynamically
optimizer_modules = [
    "PSO", "GWO", "MVO", "MFO", "CS", "BAT", 
    "WOA", "FFA", "SSA", "GA", "HHO", "SCA", 
    "JAYA", "DE"
]

# Dictionary mapping algorithm names to their functions
optimizer_map = {}

# Dynamically import each optimizer module
for name in optimizer_modules:
    try:
        module = importlib.import_module(f"EvoloPy.optimizers.{name}")
        # Make the module available for direct import
        globals()[name] = module
        # Get the optimizer function from the module
        optimizer_function = getattr(module, name)
        # Add it to the optimizer map
        optimizer_map[name] = optimizer_function
    except (ImportError, AttributeError):
        # Skip optimizers that aren't available
        pass

# List of all available optimizers for import
__all__ = list(optimizer_map.keys()) + ["optimizer_map"] 