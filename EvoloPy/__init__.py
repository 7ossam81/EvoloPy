"""
EvoloPy: An open source nature-inspired optimization library
"""

__version__ = "1.1.1"

# Import main API functions for easy access
from EvoloPy.api import (
    available_optimizers,
    available_benchmarks,
    run_optimizer,
    run_multiple_optimizers,
    get_optimizer_class
)

# Provide direct imports for convenience
import EvoloPy.optimizers as optimizers
import EvoloPy.benchmarks as benchmarks

# Export main functions at the top level
__all__ = [
    'available_optimizers',
    'available_benchmarks',
    'run_optimizer', 
    'run_multiple_optimizers',
    'get_optimizer_class',
    'optimizers',
    'benchmarks',
]
