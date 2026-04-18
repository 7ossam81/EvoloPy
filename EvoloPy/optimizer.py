# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
from pathlib import Path
import EvoloPy.optimizers.PSO as pso
import EvoloPy.optimizers.MVO as mvo
import EvoloPy.optimizers.GWO as gwo
import EvoloPy.optimizers.MFO as mfo
import EvoloPy.optimizers.CS as cs
import EvoloPy.optimizers.BAT as bat
import EvoloPy.optimizers.WOA as woa
import EvoloPy.optimizers.FFA as ffa
import EvoloPy.optimizers.SSA as ssa
import EvoloPy.optimizers.GA as ga
import EvoloPy.optimizers.HHO as hho
import EvoloPy.optimizers.SCA as sca
import EvoloPy.optimizers.JAYA as jaya
import EvoloPy.optimizers.DE as de
from EvoloPy import benchmarks
from EvoloPy.solution import solution
from EvoloPy.parallel_utils import run_optimizer_parallel, detect_hardware
import csv
import numpy
import time
import warnings
# import os
from EvoloPy import plot_convergence
from EvoloPy import plot_boxplot
from EvoloPy import plot_bar

warnings.simplefilter(action="ignore")


def selector(algo, func_details, popSize, Iter):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]

    if algo == "SSA":
        sol = ssa.SSA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "PSO":
        sol = pso.PSO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GA":
        sol = ga.GA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "BAT":
        sol = bat.BAT(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "FFA":
        sol = ffa.FFA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GWO":
        sol = gwo.GWO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "WOA":
        sol = woa.WOA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MVO":
        sol = mvo.MVO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MFO":
        sol = mfo.MFO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "CS":
        sol = cs.CS(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "HHO":
        sol = hho.HHO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "SCA":
        sol = sca.SCA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "JAYA":
        sol = jaya.JAYA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "DE":
        sol = de.DE(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    else:
        return None
    return sol


def get_optimizer_function(algo):
    """
    Return the optimizer function for a given algorithm name.
    
    Parameters:
        algo (str): Name of the algorithm
        
    Returns:
        function: The optimizer function
    """
    if algo == "SSA":
        return ssa.SSA
    elif algo == "PSO":
        return pso.PSO
    elif algo == "GA":
        return ga.GA
    elif algo == "BAT":
        return bat.BAT
    elif algo == "FFA":
        return ffa.FFA
    elif algo == "GWO":
        return gwo.GWO
    elif algo == "WOA":
        return woa.WOA
    elif algo == "MVO":
        return mvo.MVO
    elif algo == "MFO":
        return mfo.MFO
    elif algo == "CS":
        return cs.CS
    elif algo == "HHO":
        return hho.HHO
    elif algo == "SCA":
        return sca.SCA
    elif algo == "JAYA":
        return jaya.JAYA
    elif algo == "DE":
        return de.DE
    else:
        return None


def run(optimizer, objectivefunc, NumOfRuns, params, export_flags, results_directory=None, enable_parallel=False, parallel_backend='auto', num_processes=None):

    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    optimizer : list
        The list of optimizers names
    objectivefunc : list
        The list of benchmark functions
    NumOfRuns : int
        The number of independent runs
    params  : set
        The set of parameters which are:
        1. Size of population (PopulationSize)
        2. The number of iterations (Iterations)
    export_flags : set
        The set of Boolean flags which are:
        1. Export (Exporting the results in a file)
        2. Export_details (Exporting the detailed results in files)
        3. Export_convergence (Exporting the covergence plots)
        4. Export_boxplot (Exporting the box plots)
    results_directory : str, optional
        Directory to save results (default: timestamp-based directory)
    enable_parallel : bool, optional
        Whether to enable parallel processing (default: False)
    parallel_backend : str, optional
        Parallel processing backend ('multiprocessing', 'cuda', 'auto')
    num_processes : int, optional
        Number of processes to use (None for auto-detection)

    Returns
    -----------
    List of solution objects or a single solution object
    """

    # Select general parameters for all optimizers (population size, number of iterations) ....
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]

    # Export results ?
    Export = export_flags["Export_avg"]
    Export_details = export_flags["Export_details"]
    Export_convergence = export_flags["Export_convergence"]
    Export_boxplot = export_flags["Export_boxplot"]

    # Create directory for results
    if results_directory is None:
        results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    
    # Ensure directory has trailing slash
    if not results_directory.endswith('/'):
        results_directory += '/'
    
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    # Print parallel processing information if enabled
    if enable_parallel:
        hardware_info = detect_hardware()
        print("Parallel processing enabled")
        print(f"Backend: {parallel_backend}")
        if parallel_backend == 'auto':
            if hardware_info['gpu_available']:
                print("Auto-selected backend: CUDA GPU")
            else:
                print("Auto-selected backend: CPU multiprocessing")
        
        print(f"CPU cores: {hardware_info['cpu_count']}")
        if hardware_info['gpu_available']:
            print(f"GPUs available: {hardware_info['gpu_count']}")
            for i, (name, mem) in enumerate(zip(hardware_info['gpu_names'], hardware_info['gpu_memory'])):
                print(f"  GPU {i}: {name} ({mem:.2f} GB)")
        
        if num_processes is None:
            from EvoloPy.parallel_utils import get_optimal_process_count
            num_processes = get_optimal_process_count(parallel_backend)
        print(f"Using {num_processes} parallel processes")

    # Create all_results list to contain the solutions
    all_results = []

    # CSV Header for for the cinvergence
    CnvgHeader = []

    for l in range(0, Iterations):
        CnvgHeader.append("Iter" + str(l + 1))

    for j in range(0, len(objectivefunc)):
        # Create a function-specific results directory
        func_dir = results_directory + objectivefunc[j] + "/"
        results_dir = func_dir + "results/"
        plots_dir = func_dir + "plots/"
        
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        
        function_convergence_data = []
        function_best_scores = []

        for i in range(0, len(optimizer)):
            optimizer_plot_dir = plots_dir + optimizer[i] + "/"
            Path(optimizer_plot_dir).mkdir(parents=True, exist_ok=True)

            convergence = [0] * NumOfRuns
            executionTime = [0] * NumOfRuns
            
            # Run optimization process
            if enable_parallel and NumOfRuns > 1:
                # Get optimizer function
                optimizer_func = get_optimizer_function(optimizer[i])
                if optimizer_func is None:
                    print(f"Unknown optimizer: {optimizer[i]}")
                    continue
                
                # Get benchmark function
                func_details = benchmarks.getFunctionDetails(objectivefunc[j])
                objf = getattr(benchmarks, func_details[0])
                lb = func_details[1]
                ub = func_details[2]
                dim = func_details[3]
                
                # Execute multiple runs in parallel
                start_time = time.time()
                parallel_results = run_optimizer_parallel(
                    optimizer_func=optimizer_func,
                    objf=objf,
                    lb=lb,
                    ub=ub,
                    dim=dim,
                    PopSize=PopulationSize,
                    iters=Iterations,
                    num_runs=NumOfRuns,
                    parallel_backend=parallel_backend,
                    num_processes=num_processes
                )
                
                # Extract results
                for k in range(NumOfRuns):
                    sol = parallel_results[k]
                    convergence[k] = sol.convergence
                    executionTime[k] = sol.executionTime
                    optimizerName = sol.optimizer
                    objfname = sol.objfname
                    
                    # Export detailed results if needed
                    if Export_details:
                        ExportToFile = results_dir + "details.csv"
                        with open(ExportToFile, "a", newline="\n") as out:
                            writer = csv.writer(out, delimiter=",")
                            if out.tell() == 0:  # Write header only if file is empty
                                header = numpy.concatenate(
                                    [["Optimizer", "Run", "ExecutionTime", "BestFitness", "BestIndividual"], CnvgHeader]
                                )
                                writer.writerow(header)
                            
                            a = [
                                optimizerName,
                                k + 1,
                                sol.executionTime,
                                sol.best_score,
                                str(sol.bestIndividual)   # stays in ONE CSV cell
                            ] + sol.convergence.tolist()
                            
                            writer.writerow(a)
                
                # Create a solution object to store the best result
                best_result_idx = numpy.argmin([objf(sol.bestIndividual) for sol in parallel_results])
                best_result = parallel_results[best_result_idx]
                sol = solution()
                sol.optimizer = optimizer[i]
                sol.objfname = objectivefunc[j]
                sol.best = getattr(benchmarks, func_details[0])(best_result.bestIndividual)
                sol.bestIndividual = best_result.bestIndividual
                sol.best_score = sol.best
                sol.convergence = best_result.convergence
                sol.executionTime = sum(executionTime) / NumOfRuns
                sol.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
                sol.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
                sol.lb = lb
                sol.ub = ub
                sol.dim = dim
                sol.popnum = PopulationSize
                sol.maxiers = Iterations
                
                all_results.append(sol)
            else:
                # Sequential execution for each run
                all_run_results = []  # To store all run results
                
                for k in range(0, NumOfRuns):
                    func_details = benchmarks.getFunctionDetails(objectivefunc[j])
                    sol = selector(optimizer[i], func_details, PopulationSize, Iterations)
                    
                    if sol is None:  # If the optimizer isn't found
                        print(f"Error: {optimizer[i]} optimizer is not defined!")
                        continue

                    # Store the result for later use
                    all_run_results.append(sol)
                    
                    convergence[k] = sol.convergence
                    executionTime[k] = sol.executionTime
                    
                    # Export detailed results if needed
                    if Export_details:
                        ExportToFile = results_dir + "details.csv"
                        with open(ExportToFile, "a", newline="\n") as out:
                            writer = csv.writer(out, delimiter=",")
                            if out.tell() == 0:  # Write header only if file is empty
                                header = numpy.concatenate(
                                    [["Optimizer", "Run", "ExecutionTime", "BestFitness", "BestIndividual"], CnvgHeader]
                                )
                                writer.writerow(header)

                            a = [
                                sol.optimizer,
                                k + 1,
                                sol.executionTime,
                                sol.best_score,
                                str(sol.bestIndividual)   # stays in ONE Excel cell
                            ] + sol.convergence.tolist()  # expands one value per column
                            writer.writerow(a)

                # Calculate mean execution time
                mean_execution_time = sum(executionTime) / NumOfRuns

                # Define objf from func_details
                objf = getattr(benchmarks, func_details[0])

                # Choose the best result from all runs
                if all_run_results:
                    best_result = min(all_run_results, key=lambda result: result.best_score)
                else:
                    # Fallback if no valid results
                    best_result = selector(optimizer[i], func_details, PopulationSize, Iterations)

                # Store results in a solution object
                sol = solution()
                sol.optimizer = optimizer[i]
                sol.objfname = objectivefunc[j]
                sol.best = getattr(benchmarks, func_details[0])(best_result.bestIndividual)
                sol.bestIndividual = best_result.bestIndividual
                sol.best_score = sol.best
                sol.convergence = best_result.convergence
                sol.executionTime = mean_execution_time
                sol.startTime = best_result.startTime
                sol.endTime = best_result.endTime
                sol.lb = func_details[1]
                sol.ub = func_details[2]
                sol.dim = func_details[3]
                sol.popnum = PopulationSize
                sol.maxiers = Iterations

                all_results.append(sol)

            function_best_scores.append(best_result.best_score)

            if Export:
                ExportToFile = results_dir + "best.csv"
                with open(ExportToFile, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")

                    if out.tell() == 0:  # Write header only if file is empty
                        header = ["Optimizer", "ExecutionTime", "BestFitness", "BestIndividual"]
                        header.extend([f"Iter{i+1}" for i in range(Iterations)])
                        writer.writerow(header)

                    row_data = [
                        optimizer[i],
                        best_result.executionTime,
                        best_result.convergence[-1],
                        str(best_result.bestIndividual)
                    ]
                    row_data.extend(best_result.convergence.tolist())
                    writer.writerow(row_data)

            # Export average convergence for all runs
            if Export:
                ExportToFile = results_dir + "avg.csv"
                with open(ExportToFile, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")
                    avg_convergence = numpy.mean(convergence, axis=0)
                    std_convergence = numpy.std(convergence, axis=0)
                    avg_execution_time = numpy.mean(executionTime)
                    std_execution_time = numpy.std(executionTime)

                    if out.tell() == 0:  # Write header only if file is empty
                        header = ["Optimizer", "AvgExecutionTime", "StdExecutionTime", "AvgBestFitness", "StdBestFitness"]
                        header.extend([f"AvgIter{i+1}" for i in range(Iterations)])
                        header.extend([f"StdIter{i+1}" for i in range(Iterations)])
                        writer.writerow(header)

                    row_data = [optimizer[i], avg_execution_time, std_execution_time, avg_convergence[-1], std_convergence[-1]]
                    row_data.extend(avg_convergence.tolist())
                    row_data.extend(std_convergence.tolist())
                    writer.writerow(row_data)

            # Generate convergence plots
            if Export_convergence:
                plot_convergence.run(convergence, optimizer[i], objectivefunc[j], optimizer_plot_dir)

            # Generate box plots
            if Export_boxplot and NumOfRuns > 1:
                plot_boxplot.run(optimizer[i], objectivefunc[j], convergence, optimizer_plot_dir)

            # Store data for comparative plots
            function_convergence_data.append(convergence)

            # Store data for comparative plots across all optimizers of the same function
            # function_convergence_data.append(convergence)
            # function_boxplot_data.append([conv[-1] for conv in convergence])

        if len(function_convergence_data) > 1:
            comparison_dir = plots_dir + "comparison/"
            Path(comparison_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate comparative plots for all optimizers on the same function
            if Export_convergence and len(function_convergence_data) > 1:
                plot_convergence.run_comparison_avg(optimizer, objectivefunc[j], function_convergence_data, comparison_dir)
                plot_convergence.run_comparison_best(optimizer, objectivefunc[j], function_convergence_data, comparison_dir)

            if Export_boxplot and NumOfRuns > 1 and len(function_convergence_data) > 1:
                function_boxplot_data = [[conv[-1] for conv in optimizer_runs] for optimizer_runs in function_convergence_data]
                plot_boxplot.run_comparison(optimizer, objectivefunc[j], function_boxplot_data, comparison_dir)

            if len(function_best_scores) > 1:
                plot_bar.run(optimizer, objectivefunc[j], function_best_scores, comparison_dir)

    # Print the results
    if Export or Export_details:
        print("Results saved to:", results_directory)

    # Return all results
    if len(all_results) == 1:
        return all_results[0]  # Return a single solution object if only one optimizer and function
    else:
        return all_results  # Return a list of solution objects
