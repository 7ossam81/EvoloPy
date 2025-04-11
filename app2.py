import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from optimizer import selector
from benchmarks import getFunctionDetails

def run_optimizer(optimizer, function, pop_size, iterations):
    """Runs the selected optimizer and returns the result."""
    func_details = getFunctionDetails(function)
    if func_details == "nothing":
        return None, f"Invalid function: {function}"
    
    result = selector(optimizer, func_details, pop_size, iterations)
    return result, None

def calculate_convergence_rate(convergence):
    """Calculates convergence rate (percentage improvement per iteration)."""
    rates = [0]  # First iteration has no previous comparison
    for i in range(1, len(convergence)):
        if convergence[i - 1] != 0:
            rate = ((convergence[i - 1] - convergence[i]) / abs(convergence[i - 1])) * 100
        else:
            rate = 0
        rates.append(rate)
    return rates

def extract_best_values(convergence):
    """Extracts Best (Alpha), Second Best (Beta), and Third Best (Gamma) values."""
    sorted_vals = sorted(set(convergence))  # Remove duplicates and sort
    alpha = sorted_vals[0] if len(sorted_vals) > 0 else None
    beta = sorted_vals[1] if len(sorted_vals) > 1 else None
    gamma = sorted_vals[2] if len(sorted_vals) > 2 else None
    return alpha, beta, gamma

def main():
    st.title("EvoloPy Optimization GUI ")
    st.sidebar.header("Settings")
    
    # Available optimizers and benchmark functions
    optimizers = ["BAT", "CS", "DE", "FFA", "GA", "GWO", "HHO", "JAYA", "MFO", "MVO"]
    benchmark_functions = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
    
    # Allow selection of multiple optimizers & benchmark functions
    selected_optimizers = st.sidebar.multiselect("Choose Optimizers", optimizers, default=["GA", "DE"])
    selected_functions = st.sidebar.multiselect("Select Benchmark Functions", benchmark_functions, default=["F1", "F2"])
    
    # Population size and iterations input
    pop_size = st.sidebar.number_input("Population Size", min_value=5, max_value=100, value=30)
    iterations = st.sidebar.number_input("Iterations", min_value=10, max_value=1000, value=100)
    
    if st.sidebar.button("Run Optimization"):
        with st.spinner("Running optimization..."):
            results = {}  # Dictionary to store results

            # Run each optimizer on each function
            for function in selected_functions:
                for optimizer in selected_optimizers:
                    result, error = run_optimizer(optimizer, function, pop_size, iterations)
                    if error:
                        st.error(f"Error with {optimizer} on {function}: {error}")
                    else:
                        results[(optimizer, function)] = result
            
            if results:
                st.success("Optimization completed!")
                
                # Display results for each optimizer-function pair
                for (optimizer, function), result in results.items():
                    st.write(f"## Results for {optimizer} on {function}")
                    st.write("Best Individual:", result.bestIndividual)

                    # Compute best (alpha), second best (beta), and third best (gamma)
                    alpha, beta, gamma = extract_best_values(result.convergence)
                    
                    # Get best fitness
                    best_fitness = np.min(result.convergence) if isinstance(result.convergence, np.ndarray) else min(result.convergence)
                    st.write("Best Fitness (Alpha):", alpha)
                    st.write("Second Best (Beta):", beta)
                    st.write("Third Best (Gamma):", gamma)
                    st.write("Execution Time:", result.executionTime)

                    # Compute convergence rate
                    convergence_rates = calculate_convergence_rate(result.convergence)
                    
                    # Create dataframe for plotting
                    df = pd.DataFrame({
                        "Iteration": range(1, len(result.convergence) + 1),
                        "Fitness Value": result.convergence,
                        "Convergence Rate (%)": convergence_rates
                    })

                    # Display fitness values as table
                    st.write("### Fitness Values Over Iterations")
                    st.dataframe(df)

                    # Convergence plot (Plotly)
                    st.write("### Convergence Plot")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["Iteration"], y=df["Fitness Value"],
                        mode='lines+markers', name=f"{optimizer} on {function}",
                        line=dict(width=2)
                    ))
                    fig.update_layout(
                        xaxis_title="Iterations", yaxis_title="Fitness",
                        legend_title="Optimizer", template="plotly_white"
                    )
                    st.plotly_chart(fig)

                    # Convergence rate plot (Plotly)
                    st.write("### Convergence Rate Plot")
                    fig_rate = px.bar(df, x="Iteration", y="Convergence Rate (%)",
                                      title=f"Convergence Rate for {optimizer} on {function}")
                    st.plotly_chart(fig_rate)

                # Compare optimizers across functions
                st.write("## Optimizer Comparison Across Functions")
                fig = go.Figure()
                for (optimizer, function), result in results.items():
                    df = pd.DataFrame({"Iteration": range(1, len(result.convergence) + 1), "Fitness Value": result.convergence})
                    fig.add_trace(go.Scatter(
                        x=df["Iteration"], y=df["Fitness Value"],
                        mode='lines', name=f"{optimizer} on {function}",
                        line=dict(width=2)
                    ))

                fig.update_layout(
                    xaxis_title="Iterations", yaxis_title="Fitness",
                    legend_title="Optimizer", template="plotly_white"
                )
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
