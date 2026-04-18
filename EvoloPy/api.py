"""
High-level API functions for EvoloPy.

This module provides simplified access to EvoloPy's core functionality
with a more Pythonic and user-friendly interface.
"""

import numpy as np
from typing import Union, List, Dict, Any, Callable, Optional
from EvoloPy.optimizer import run as optimizer_run
from EvoloPy.solution import solution
import importlib
import time

# Import parallel processing utilities
try:
    from EvoloPy.parallel_utils import detect_hardware, get_optimal_process_count
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

def get_optimizer_map():
    """Get a dictionary mapping optimizer names to their functions."""
    optimizer_map = {}
    optimizer_modules = [
        "PSO", "GWO", "MVO", "MFO", "CS", "BAT", 
        "WOA", "FFA", "SSA", "GA", "HHO", "SCA", 
        "JAYA", "DE"
    ]
    
    for name in optimizer_modules:
        try:
            module = importlib.import_module(f"EvoloPy.optimizers.{name}")
            optimizer_function = getattr(module, name)
            optimizer_map[name] = optimizer_function
        except (ImportError, AttributeError):
            # Skip optimizers that aren't available
            pass
    
    return optimizer_map

def get_optimizer_class(optimizer_name: str) -> Callable:
    """
    Get the optimizer class/function by name.
    
    Parameters:
        optimizer_name (str): Name of the optimizer algorithm
    
    Returns:
        Callable: The optimizer function
        
    Raises:
        ValueError: If the optimizer does not exist
        
    Example:
        >>> from EvoloPy.api import get_optimizer_class
        >>> PSO = get_optimizer_class("PSO")
        >>> result = PSO(objective_function, lb=-10, ub=10, dim=5, PopSize=30, iters=50)
    """
    optimizer_map = get_optimizer_map()
    
    if optimizer_name not in optimizer_map:
        raise ValueError(f"Optimizer '{optimizer_name}' not found. Available optimizers: {list(optimizer_map.keys())}")
        
    return optimizer_map[optimizer_name]

def available_optimizers() -> List[str]:
    """
    Get a list of all available optimization algorithms.
    
    Returns:
        List[str]: List of optimizer names
        
    Example:
        >>> from EvoloPy.api import available_optimizers
        >>> print(available_optimizers())
        ['PSO', 'GWO', 'MVO', ...]
    """
    return list(get_optimizer_map().keys())

def available_benchmarks() -> List[str]:
    """
    Get a list of all available benchmark functions.
    
    Returns:
        List[str]: List of benchmark function names
        
    Example:
        >>> from EvoloPy.api import available_benchmarks
        >>> print(available_benchmarks())
        ['F1', 'F2', 'F3', ...]
    """
    # List of all benchmark functions
    return [
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
        "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19",
        "F20", "F21", "F22", "F23", "F24", "ackley", "rosenbrock", 
        "rastrigin", "griewank"
    ]

def run_optimizer(
    optimizer: str,
    objective_func: Union[str, Callable],
    lb: Union[float, List[float]] = -100,
    ub: Union[float, List[float]] = 100,
    dim: int = 30,
    population_size: int = 30,
    iterations: int = 50,
    num_runs: int = 1,
    export_results: bool = True,
    export_details: bool = True,
    export_convergence: bool = True,
    export_boxplot: bool = True,
    display_plots: bool = True,
    results_directory: Optional[str] = None,
    enable_parallel: bool = False,
    parallel_backend: str = 'auto',
    num_processes: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a single optimizer on a specified objective function.

    Parameters:
        optimizer (str): Name of the optimizer algorithm
        objective_func (str or callable): Either a benchmark name (e.g., "F1")
                                         or a custom objective function
        lb (float or list): Lower bound for variables
        ub (float or list): Upper bound for variables
        dim (int): Problem dimension
        population_size (int): Size of the population
        iterations (int): Maximum number of iterations
        num_runs (int): Number of independent runs
        export_results (bool): Whether to export average results
        export_details (bool): Whether to export detailed results
        export_convergence (bool): Whether to export convergence plots
        export_boxplot (bool): Whether to export boxplots
        display_plots (bool): Whether to keep plots in the returned dictionary
        results_directory (str, optional): Root directory to save results
        enable_parallel (bool): Whether to enable parallel processing
        parallel_backend (str): Parallel processing backend ('multiprocessing', 'cuda', 'auto')
        num_processes (int, optional): Number of processes to use (None for auto-detection)

    Returns:
        Dict[str, Any]: Results dictionary containing:
            - 'best_solution': Best solution found
            - 'best_fitness': Best fitness value
            - 'convergence': Convergence history of the best run
            - 'execution_time': Average execution time across runs
            - 'plots': Optional dictionary (currently empty placeholder)
    """
    import os
    import csv
    from pathlib import Path
    import numpy as np

    optimizer_map = get_optimizer_map()

    # Check if optimizer exists
    if optimizer not in optimizer_map:
        raise ValueError(
            f"Optimizer '{optimizer}' not found. Available optimizers: {available_optimizers()}"
        )

    # Check parallel configuration
    if enable_parallel and not PARALLEL_AVAILABLE:
        print("Warning: Parallel processing requested but not available. Installing psutil package is required.")
        enable_parallel = False

    # Boxplot only makes sense for multiple runs
    if num_runs <= 1:
        export_boxplot = False
    else:
        export_boxplot = True

    # Create organized results root if not provided
    if results_directory is None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        results_directory = f"{timestamp}/"

    # Ensure directory has trailing slash
    if not results_directory.endswith('/'):
        results_directory += '/'

    plots = {}

    # =========================================================
    # Case 1: Built-in benchmark function (string)
    # =========================================================
    if isinstance(objective_func, str):
        if objective_func not in available_benchmarks():
            raise ValueError(
                f"Benchmark '{objective_func}' not found. Available benchmarks: {available_benchmarks()}"
            )

        params = {"PopulationSize": population_size, "Iterations": iterations}
        export_flags = {
            "Export_avg": export_results,
            "Export_details": export_details,
            "Export_convergence": export_convergence,
            "Export_boxplot": export_boxplot
        }

        # Call the main optimizer engine using the ROOT directory
        results = optimizer_run(
            [optimizer],
            [objective_func],
            num_runs,
            params,
            export_flags,
            results_directory,
            enable_parallel,
            parallel_backend,
            num_processes
        )

        # Save config inside <function>/results/
        func_dir = os.path.join(results_directory, objective_func)
        results_subdir = os.path.join(func_dir, "results")
        Path(results_subdir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(results_subdir, f"{optimizer}_config.txt"), "w") as f:
            f.write(f"Optimizer: {optimizer}\n")
            f.write(f"Function: {objective_func}\n")
            f.write(f"Dimension: {dim}\n")
            f.write(f"Population Size: {population_size}\n")
            f.write(f"Iterations: {iterations}\n")
            f.write(f"Number of Runs: {num_runs}\n")
            f.write(f"Lower Bound: {lb}\n")
            f.write(f"Upper Bound: {ub}\n")
            f.write(f"Parallel: {enable_parallel}\n")
            if enable_parallel:
                f.write(f"Parallel Backend: {parallel_backend}\n")
                f.write(f"Processes: {num_processes or 'auto'}\n")

        # Process and return results
        if hasattr(results, 'bestIndividual'):
            return {
                'best_solution': results.bestIndividual,
                'best_fitness': results.best_score if hasattr(results, 'best_score') else results.best,
                'convergence': results.convergence,
                'execution_time': results.executionTime,
                'plots': plots if display_plots else None
            }

        elif isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'bestIndividual'):
            best_result = min(results, key=lambda x: getattr(x, 'best_score', float('inf')))
            return {
                'best_solution': best_result.bestIndividual,
                'best_fitness': best_result.best_score if hasattr(best_result, 'best_score') else best_result.best,
                'convergence': best_result.convergence,
                'execution_time': best_result.executionTime,
                'plots': plots if display_plots else None
            }

        else:
            return {
                'best_solution': None,
                'best_fitness': None,
                'convergence': None,
                'execution_time': None,
                'plots': plots if display_plots else None
            }

    # =========================================================
    # Case 2: Custom objective function (callable)
    # =========================================================
    elif callable(objective_func):
        optimizer_func = optimizer_map[optimizer]
        func_name = getattr(objective_func, '__name__', 'custom_function')

        from EvoloPy import plot_convergence, plot_boxplot

        func_dir = os.path.join(results_directory, func_name)
        results_subdir = os.path.join(func_dir, "results")
        optimizer_plot_dir = os.path.join(func_dir, "plots", optimizer)

        Path(results_subdir).mkdir(parents=True, exist_ok=True)
        Path(optimizer_plot_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        with open(os.path.join(results_subdir, f"{optimizer}_config.txt"), "w") as f:
            f.write(f"Optimizer: {optimizer}\n")
            f.write(f"Function: {func_name}\n")
            f.write(f"Dimension: {dim}\n")
            f.write(f"Population Size: {population_size}\n")
            f.write(f"Iterations: {iterations}\n")
            f.write(f"Number of Runs: {num_runs}\n")
            f.write(f"Lower Bound: {lb}\n")
            f.write(f"Upper Bound: {ub}\n")
            f.write(f"Parallel: {enable_parallel}\n")
            if enable_parallel:
                f.write(f"Parallel Backend: {parallel_backend}\n")
                f.write(f"Processes: {num_processes or 'auto'}\n")

        # Execute runs
        if enable_parallel and num_runs > 1:
            from EvoloPy.parallel_utils import run_optimizer_parallel

            run_results = run_optimizer_parallel(
                optimizer_func=optimizer_func,
                objf=objective_func,
                lb=lb,
                ub=ub,
                dim=dim,
                PopSize=population_size,
                iters=iterations,
                num_runs=num_runs,
                parallel_backend=parallel_backend,
                num_processes=num_processes
            )
        else:
            run_results = [
                optimizer_func(objective_func, lb, ub, dim, population_size, iterations)
                for _ in range(num_runs)
            ]

        convergence = [r.convergence for r in run_results]
        execution_time = [r.executionTime for r in run_results]

        # Best run
        best_result = min(run_results, key=lambda r: r.best_score)

        # -------------------------
        # Export details.csv
        # -------------------------
        if export_details:
            details_file = os.path.join(results_subdir, "details.csv")
            with open(details_file, "w", newline="\n") as out:
                writer = csv.writer(out, delimiter=",")
                header = ["Optimizer", "Run", "ExecutionTime", "BestScore", "BestIndividual"]
                header.extend([f"Iter{i+1}" for i in range(iterations)])
                writer.writerow(header)

                for k, r in enumerate(run_results):
                    row = [
                        optimizer,
                        k + 1,
                        r.executionTime,
                        r.best_score,
                        str(r.bestIndividual)
                    ]
                    row.extend(r.convergence.tolist())
                    writer.writerow(row)

        # -------------------------
        # Export avg.csv and best.csv
        # -------------------------
        if export_results:
            avg_file = os.path.join(results_subdir, "avg.csv")
            avg_convergence = np.mean(convergence, axis=0)
            std_convergence = np.std(convergence, axis=0)
            avg_execution_time = np.mean(execution_time)
            std_execution_time = np.std(execution_time)

            with open(avg_file, "w", newline="\n") as out:
                writer = csv.writer(out, delimiter=",")
                header = ["Optimizer", "AvgExecutionTime", "StdExecutionTime", "AvgBestScore", "StdBestScore"]
                header.extend([f"AvgIter{i+1}" for i in range(iterations)])
                header.extend([f"StdIter{i+1}" for i in range(iterations)])
                writer.writerow(header)

                row = [
                    optimizer,
                    avg_execution_time,
                    std_execution_time,
                    avg_convergence[-1],
                    std_convergence[-1]
                ]
                row.extend(avg_convergence.tolist())
                row.extend(std_convergence.tolist())
                writer.writerow(row)

            best_file = os.path.join(results_subdir, "best.csv")
            with open(best_file, "w", newline="\n") as out:
                writer = csv.writer(out, delimiter=",")
                header = ["Optimizer", "ExecutionTime", "BestScore", "BestIndividual"]
                header.extend([f"Iter{i+1}" for i in range(iterations)])
                writer.writerow(header)

                row = [
                    optimizer,
                    best_result.executionTime,
                    best_result.best_score,
                    str(best_result.bestIndividual)
                ]
                row.extend(best_result.convergence.tolist())
                writer.writerow(row)

        # -------------------------
        # Export plots
        # -------------------------
        if export_convergence:
            plot_convergence.run(convergence, optimizer, func_name, optimizer_plot_dir + "/")

        if export_boxplot and num_runs > 1:
            plot_boxplot.run(optimizer, func_name, convergence, optimizer_plot_dir + "/")

        return {
            'best_solution': best_result.bestIndividual,
            'best_fitness': best_result.best_score,
            'convergence': best_result.convergence,
            'execution_time': np.mean(execution_time),
            'plots': plots if display_plots else None
        }

    else:
        raise TypeError("objective_func must be either a string (benchmark name) or a callable function")
def run_multiple_optimizers(
    optimizers: List[str],
    objective_funcs: Union[List[Union[str, Callable]], Union[str, Callable]],
    lb: Union[float, List[float]] = -100,
    ub: Union[float, List[float]] = 100,
    dim: int = 30,
    population_size: int = 30,
    iterations: int = 50,
    num_runs: int = 1,
    export_results: bool = True,
    export_details: bool = True,
    export_convergence: bool = True,
    export_boxplot: bool = True,
    display_plots: bool = True,
    results_directory: Optional[str] = None,
    enable_parallel: bool = False,
    parallel_backend: str = 'auto',
    num_processes: Optional[int] = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run multiple optimizers on multiple objective functions.
    Supports benchmark names (strings) and custom callable functions.
    """

    import os
    import csv
    import time
    import numpy as np
    from pathlib import Path

    optimizer_map = get_optimizer_map()

    # Validate optimizers
    for opt in optimizers:
        if opt not in optimizer_map:
            raise ValueError(
                f"Optimizer '{opt}' not found. Available optimizers: {available_optimizers()}"
            )

    # Normalize objective_funcs to a list
    if not isinstance(objective_funcs, list):
        objective_funcs = [objective_funcs]

    # Check parallel configuration
    if enable_parallel and not PARALLEL_AVAILABLE:
        print("Warning: Parallel processing requested but not available. Installing psutil package is required.")
        enable_parallel = False

    # Boxplot only makes sense for multiple runs
    if num_runs <= 1:
        export_boxplot = False
    else:
        export_boxplot = True

    # Create shared root directory
    if results_directory is None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        results_directory = f"{timestamp}/"

    if not results_directory.endswith('/'):
        results_directory += '/'

    # =========================================================
    # Case 1: all objective functions are benchmark names
    # =========================================================
    if all(isinstance(func, str) for func in objective_funcs):
        for func in objective_funcs:
            if func not in available_benchmarks():
                raise ValueError(
                    f"Benchmark '{func}' not found. Available benchmarks: {available_benchmarks()}"
                )

        params = {
            "PopulationSize": population_size,
            "Iterations": iterations
        }

        export_flags = {
            "Export_avg": export_results,
            "Export_details": export_details,
            "Export_convergence": export_convergence,
            "Export_boxplot": export_boxplot
        }

        raw_results = optimizer_run(
            optimizers,
            objective_funcs,
            num_runs,
            params,
            export_flags,
            results_directory,
            enable_parallel,
            parallel_backend,
            num_processes
        )

        if not isinstance(raw_results, list):
            raw_results = [raw_results]

        all_results = {opt: {} for opt in optimizers}

        for sol in raw_results:
            all_results[sol.optimizer][sol.objfname] = {
                'best_solution': sol.bestIndividual,
                'best_fitness': sol.best_score if hasattr(sol, 'best_score') else sol.best,
                'convergence': sol.convergence,
                'execution_time': sol.executionTime
            }

        all_results['raw_results'] = raw_results
        all_results['results_directory'] = results_directory
        return all_results

    # =========================================================
    # Case 2: all objective functions are callable
    # =========================================================
    elif all(callable(func) for func in objective_funcs):
        from EvoloPy import plot_convergence, plot_boxplot, plot_bar

        all_results = {opt: {} for opt in optimizers}

        for func in objective_funcs:
            func_name = getattr(func, '__name__', 'custom_function')

            func_dir = os.path.join(results_directory, func_name)
            results_subdir = os.path.join(func_dir, "results")
            plots_dir = os.path.join(func_dir, "plots")
            comparison_dir = os.path.join(plots_dir, "comparison")

            Path(results_subdir).mkdir(parents=True, exist_ok=True)
            Path(plots_dir).mkdir(parents=True, exist_ok=True)

            function_convergence_data = []
            function_best_scores = []

            for opt in optimizers:
                optimizer_func = optimizer_map[opt]
                optimizer_plot_dir = os.path.join(plots_dir, opt)
                Path(optimizer_plot_dir).mkdir(parents=True, exist_ok=True)

                # Save config
                with open(os.path.join(results_subdir, f"{opt}_config.txt"), "w") as f:
                    f.write(f"Optimizer: {opt}\n")
                    f.write(f"Function: {func_name}\n")
                    f.write(f"Dimension: {dim}\n")
                    f.write(f"Population Size: {population_size}\n")
                    f.write(f"Iterations: {iterations}\n")
                    f.write(f"Number of Runs: {num_runs}\n")
                    f.write(f"Lower Bound: {lb}\n")
                    f.write(f"Upper Bound: {ub}\n")
                    f.write(f"Parallel: {enable_parallel}\n")
                    if enable_parallel:
                        f.write(f"Parallel Backend: {parallel_backend}\n")
                        f.write(f"Processes: {num_processes or 'auto'}\n")

                # Run optimizer
                if enable_parallel and num_runs > 1:
                    from EvoloPy.parallel_utils import run_optimizer_parallel

                    run_results = run_optimizer_parallel(
                        optimizer_func=optimizer_func,
                        objf=func,
                        lb=lb,
                        ub=ub,
                        dim=dim,
                        PopSize=population_size,
                        iters=iterations,
                        num_runs=num_runs,
                        parallel_backend=parallel_backend,
                        num_processes=num_processes
                    )
                else:
                    run_results = [
                        optimizer_func(func, lb, ub, dim, population_size, iterations)
                        for _ in range(num_runs)
                    ]

                convergence = [r.convergence for r in run_results]
                execution_time = [r.executionTime for r in run_results]

                # Best run for this optimizer
                best_result = min(run_results, key=lambda r: r.best_score)

                function_convergence_data.append(convergence)
                function_best_scores.append(best_result.best_score)

                # -------------------------
                # Append details.csv
                # -------------------------
                if export_details:
                    details_file = os.path.join(results_subdir, "details.csv")
                    with open(details_file, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if out.tell() == 0:
                            header = ["Optimizer", "Run", "ExecutionTime", "BestScore", "BestIndividual"]
                            header.extend([f"Iter{i+1}" for i in range(iterations)])
                            writer.writerow(header)

                        for k, r in enumerate(run_results):
                            row = [
                                opt,
                                k + 1,
                                r.executionTime,
                                r.best_score,
                                str(r.bestIndividual)
                            ]
                            row.extend(r.convergence.tolist())
                            writer.writerow(row)

                # -------------------------
                # Append avg.csv
                # -------------------------
                if export_results:
                    avg_file = os.path.join(results_subdir, "avg.csv")
                    avg_convergence = np.mean(convergence, axis=0)
                    std_convergence = np.std(convergence, axis=0)
                    avg_execution_time = np.mean(execution_time)
                    std_execution_time = np.std(execution_time)

                    with open(avg_file, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if out.tell() == 0:
                            header = ["Optimizer", "AvgExecutionTime", "StdExecutionTime", "AvgBestScore", "StdBestScore"]
                            header.extend([f"AvgIter{i+1}" for i in range(iterations)])
                            header.extend([f"StdIter{i+1}" for i in range(iterations)])
                            writer.writerow(header)

                        row = [
                            opt,
                            avg_execution_time,
                            std_execution_time,
                            avg_convergence[-1],
                            std_convergence[-1]
                        ]
                        row.extend(avg_convergence.tolist())
                        row.extend(std_convergence.tolist())
                        writer.writerow(row)

                    # -------------------------
                    # Append best.csv
                    # -------------------------
                    best_file = os.path.join(results_subdir, "best.csv")
                    with open(best_file, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if out.tell() == 0:
                            header = ["Optimizer", "ExecutionTime", "BestScore", "BestIndividual"]
                            header.extend([f"Iter{i+1}" for i in range(iterations)])
                            writer.writerow(header)

                        row = [
                            opt,
                            best_result.executionTime,
                            best_result.best_score,
                            str(best_result.bestIndividual)
                        ]
                        row.extend(best_result.convergence.tolist())
                        writer.writerow(row)

                # -------------------------
                # Individual plots
                # -------------------------
                if export_convergence:
                    plot_convergence.run(convergence, opt, func_name, optimizer_plot_dir + "/")

                if export_boxplot and num_runs > 1:
                    plot_boxplot.run(opt, func_name, convergence, optimizer_plot_dir + "/")

                # Store API result
                all_results[opt][func_name] = {
                    'best_solution': best_result.bestIndividual,
                    'best_fitness': best_result.best_score,
                    'convergence': best_result.convergence,
                    'execution_time': float(np.mean(execution_time)),
                    'plots': {} if display_plots else None
                }

            # -------------------------
            # Comparison plots
            # -------------------------
            if len(function_convergence_data) > 1:
                Path(comparison_dir).mkdir(parents=True, exist_ok=True)

                if export_convergence:
                    plot_convergence.run_comparison_avg(
                        optimizers, func_name, function_convergence_data, comparison_dir + "/"
                    )
                    plot_convergence.run_comparison_best(
                        optimizers, func_name, function_convergence_data, comparison_dir + "/"
                    )

                if export_boxplot and num_runs > 1:
                    function_boxplot_data = [
                        [conv[-1] for conv in optimizer_runs]
                        for optimizer_runs in function_convergence_data
                    ]
                    plot_boxplot.run_comparison(
                        optimizers, func_name, function_boxplot_data, comparison_dir + "/"
                    )

                if len(function_best_scores) > 1:
                    plot_bar.run(
                        optimizers, func_name, function_best_scores, comparison_dir + "/"
                    )

        all_results['results_directory'] = results_directory
        return all_results

    else:
        raise TypeError(
            "objective_funcs must be all benchmark names (strings) or all callable functions"
        )

def get_hardware_info() -> Dict[str, Any]:
    """
    Get information about available hardware for parallel processing.
    
    Returns:
        Dict[str, Any]: Dictionary containing hardware information:
            - cpu_count: Number of CPU cores
            - cpu_threads: Number of CPU threads
            - ram_gb: Available RAM in GB
            - gpu_available: Whether CUDA GPU is available
            - gpu_count: Number of CUDA GPUs
            - gpu_names: List of GPU names
            - gpu_memory: List of GPU memory in GB
            
    Example:
        >>> from EvoloPy.api import get_hardware_info
        >>> hw_info = get_hardware_info()
        >>> print(f"CPU cores: {hw_info['cpu_count']}")
        >>> if hw_info['gpu_available']:
        ...     print(f"GPU: {hw_info['gpu_names'][0]}")
    """
    if not PARALLEL_AVAILABLE:
        raise ImportError("Hardware detection requires the psutil package. Install with: pip install psutil")
    return detect_hardware() 