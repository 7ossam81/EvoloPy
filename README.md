<div align="center">
<img width="200" alt="EvoloPy-logo" src="https://github.com/user-attachments/assets/496f9a76-1fcc-4e4f-9586-8f327a434134">
</div>

# EvoloPy: Nature-Inspired Optimization in Python

[![PyPI version](https://badge.fury.io/py/EvoloPy.svg)](https://badge.fury.io/py/EvoloPy)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

EvoloPy is a powerful, easy-to-use optimization library featuring 14 nature-inspired algorithms, performance visualizations, and parallel processing capabilities. Version 4.0 introduces enhanced output organization, better bounds handling, and improved result reporting.

## ✨ Features

- 🔍 **14 Optimizers**: PSO, GWO, MVO, and more
- 🚀 **Parallel Processing**: Multi-core CPU and GPU acceleration (v3.0+) 
- 📊 **Visualization Tools**: Convergence curves and performance comparisons
- 🔧 **Simple API**: Consistent interface across all algorithms
- 📋 **24 Benchmark Functions**: Extensive testing suite
- 📁 **Organized Results**: Structured output folders for easy analysis (v4.0+)

## 📦 Installation

### Basic Installation
```bash
pip install EvoloPy
```

### With GPU Acceleration
```bash
pip install EvoloPy[gpu]
```

## 🚀 Quick Start Guide

### 1. Simple Optimization

```python
from EvoloPy.api import run_optimizer

# Run PSO on benchmark function F1
result = run_optimizer(
    optimizer="PSO",
    objective_func="F1",
    dim=30,
    lb=-100,                 # Lower bounds (scalar or list)
    ub=100,                  # Upper bounds (scalar or list)
    population_size=50,
    iterations=100,
    num_runs=5,              # Number of independent runs
    results_directory=None   # Auto-create timestamped directory
)

print(f"Best fitness: {result['best_fitness']}")
print(f"Best solution: {result['best_solution']}")
print(f"Execution time: {result['execution_time']} seconds")
```

### 2. Jupyter Notebook Visualization (v4.0.5+)

```python
from EvoloPy.api import run_optimizer, run_multiple_optimizers
import matplotlib.pyplot as plt

# Compare multiple optimizers with visualization
results = run_multiple_optimizers(
    optimizers=["PSO", "GWO", "MVO"],
    objective_funcs=["F1", "F5"],
    dim=10,
    lb=-100,
    ub=100,
    population_size=30,
    iterations=50,
    num_runs=3,
    display_plots=True  # Enable interactive plots in notebook
)

# Display convergence comparison plot
plt.figure(results['plots']['convergence_F1'])
plt.show()

# Display performance summary across functions
plt.figure(results['plots']['performance_summary'])
plt.show()

# Single optimizer with visualization
result = run_optimizer(
    optimizer="PSO",
    objective_func="F1",
    dim=10,
    population_size=30,
    iterations=50,
    num_runs=5,
    display_plots=True  # Enable interactive plots
)

# Display convergence plot
plt.figure(result['plots']['avg_convergence'])
plt.show()

# Display boxplot of multiple runs
plt.figure(result['plots']['boxplot'])
plt.show()
```

### 3. Optimize Your Custom Function

```python
import numpy as np
from EvoloPy.optimizers import PSO

# Define your custom objective function
def my_equation(x):
    # Example: Minimize f(x) = sum(x^2) + sum(sin(x))
    return np.sum(x**2) + np.sum(np.sin(x))

# Run optimization on your function
result = PSO.PSO(
    objf=my_equation,        # Your custom function
    lb=[-10] * 5,            # Lower bounds as list (v4.0+ supports lists)
    ub=[10] * 5,             # Upper bounds as list
    dim=5,                   # Dimension
    PopSize=30,              # Population size
    iters=100                # Max iterations
)

# Get results
best_solution = result.bestIndividual
best_fitness = result.best_score  # v4.0+ exposes best_score directly

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

### 4. Parallel Processing (v3.0+)

```python
from EvoloPy.api import run_optimizer, get_hardware_info

# Check available hardware
hw_info = get_hardware_info()
print(f"CPU cores: {hw_info['cpu_count']}")
if hw_info['gpu_available']:
    print(f"GPU: {hw_info['gpu_names'][0]}")

# Run with parallel processing
result = run_optimizer(
    optimizer="PSO",
    objective_func="F1",
    dim=30,
    lb=-100,
    ub=100,
    population_size=50,
    iterations=100,
    num_runs=10,                # Number of independent runs
    enable_parallel=True,       # Enable parallel processing
    parallel_backend="auto"     # Auto-select CPU or GPU
)
```

### 5. Compare Multiple Optimizers

```python
from EvoloPy.api import run_multiple_optimizers

# Compare PSO, GWO and MVO on F1 and F5
results = run_multiple_optimizers(
    optimizers=["PSO", "GWO", "MVO"],
    objective_funcs=["F1", "F5"],
    dim=30,
    lb=-100,                   # v4.0+ supports list bounds
    ub=100,
    population_size=50,
    iterations=100,
    num_runs=5,                # Multiple runs for statistical significance
    export_convergence=True,   # Generate convergence plots
    export_boxplot=True        # Generate boxplots for multiple runs
)
```

### 6. Command Line Usage

```bash
# List available optimizers and benchmarks
evolopy --list

# Run PSO on F1 (v4.0+ syntax)
evolopy --optimizer PSO --function F1 --dim 30 --iterations 100 --runs 5

# Run with parallel processing and list bounds
evolopy --optimizer PSO --function F1 --dim 30 --iterations 100 --lb "-10,-10,-10" --ub "10,10,10" --parallel

# Disable certain exports
evolopy --optimizer PSO --function F1 --no-export-boxplot --no-export-details
```

## 📋 Available Optimizers

| Abbreviation | Algorithm Name                   |
|--------------|----------------------------------|
| PSO          | Particle Swarm Optimization      |
| GWO          | Grey Wolf Optimizer              |
| MVO          | Multi-Verse Optimizer            |
| MFO          | Moth Flame Optimization          |
| CS           | Cuckoo Search                    |
| BAT          | Bat Algorithm                    |
| WOA          | Whale Optimization Algorithm     |
| FFA          | Firefly Algorithm                |
| SSA          | Salp Swarm Algorithm             |
| GA           | Genetic Algorithm                |
| HHO          | Harris Hawks Optimization        |
| SCA          | Sine Cosine Algorithm            |
| JAYA         | JAYA Algorithm                   |
| DE           | Differential Evolution           |

## 🧠 Optimization Algorithms in Detail

### PSO (Particle Swarm Optimization)
- **Inspiration**: Social behavior of bird flocking or fish schooling
- **Year**: 1995
- **Developers**: Kennedy and Eberhart
- **Key Parameters**: 
  - Inertia weight (w): Controls momentum of particles
  - Cognitive coefficient (c1): Personal influence factor 
  - Social coefficient (c2): Social influence factor
  - Maximum velocity (Vmax): Limits velocity to prevent explosion
- **Process**: Particles move through search space, remembering their own best positions and global best position. Velocity is updated using personal and social components.
- **Strengths**: Easy implementation, few parameters, fast convergence on unimodal functions
- **Ideal For**: Continuous optimization problems, especially when gradient information is unavailable

### GWO (Grey Wolf Optimizer)
- **Inspiration**: Social hierarchy and hunting behavior of grey wolves
- **Year**: 2014
- **Developer**: Mirjalili
- **Key Parameters**:
  - Alpha, beta, and delta wolves: Three best solutions
  - Parameter a: Decreases linearly from 2 to 0
- **Process**: Solutions are ranked as alpha, beta, delta, and omega wolves. Omegas update their positions based on alpha, beta, and delta positions.
- **Strengths**: Balanced exploration and exploitation, avoids local optima well
- **Ideal For**: Complex multimodal optimization problems

### MVO (Multi-Verse Optimizer)
- **Inspiration**: Theory of multi-verse in physics
- **Year**: 2016
- **Developers**: Mirjalili, Mirjalili, and Hatamlou
- **Key Parameters**:
  - Wormhole Existence Probability (WEP): Controls exploration/exploitation
  - Traveling Distance Rate (TDR): Controls teleportation distance
- **Process**: Uses concepts of white holes, black holes, and wormholes to create "universes" that can exchange information
- **Strengths**: Strong global exploration, good at escaping local optima
- **Ideal For**: Complex multimodal problems with many local optima

### MFO (Moth Flame Optimization)
- **Inspiration**: Navigation behavior of moths in nature (flying toward light sources)
- **Year**: 2015
- **Developer**: Mirjalili
- **Key Parameters**:
  - Number of flames: Controls exploration/exploitation balance
  - Parameter b: Defines the shape of the spiral
- **Process**: Moths fly around flames in spiral pattern, with flames representing best solutions
- **Strengths**: Good balance between exploration and exploitation
- **Ideal For**: Problems requiring precision in local search

### CS (Cuckoo Search)
- **Inspiration**: Brood parasitism behavior of cuckoo birds
- **Year**: 2009
- **Developers**: Yang and Deb
- **Key Parameters**:
  - Discovery probability (pa): Controls fraction of worse solutions to be abandoned
  - Lévy flight parameters: Controls the random walk characteristics
- **Process**: Uses Lévy flights for exploration and host nest switching for exploitation
- **Strengths**: Effective at exploring large spaces and converging to global optimum
- **Ideal For**: Continuous nonlinear optimization problems

### BAT (Bat Algorithm)
- **Inspiration**: Echolocation behavior of microbats
- **Year**: 2010
- **Developer**: Yang
- **Key Parameters**:
  - Pulse rate: Controls exploitation
  - Loudness: Controls exploration
  - Frequency range: Controls step size
- **Process**: Bats use echolocation to sense distance and prey, adjusting flight patterns based on loudness and pulse rate
- **Strengths**: Balance between exploration and exploitation, adaptive parameters
- **Ideal For**: Continuous optimization problems with constraints

### WOA (Whale Optimization Algorithm)
- **Inspiration**: Hunting behavior of humpback whales, particularly bubble-net feeding
- **Year**: 2016
- **Developer**: Mirjalili and Lewis
- **Key Parameters**:
  - Parameter a: Decreases linearly from 2 to 0
  - Parameter b: Defines spiral shape
- **Process**: Uses encircling prey, bubble-net attack (exploitation), and search for prey (exploration)
- **Strengths**: Good balance between exploration and exploitation phases
- **Ideal For**: Highly nonlinear optimization problems

### FFA (Firefly Algorithm)
- **Inspiration**: Flashing behavior of fireflies
- **Year**: 2008
- **Developer**: Yang
- **Key Parameters**:
  - Light absorption coefficient: Controls visibility
  - Attractiveness: Defines the strength of attraction
  - Randomization parameter: Controls random movement
- **Process**: Fireflies are attracted to each other based on brightness, which relates to objective function
- **Strengths**: Good at dealing with multimodal problems, automatic subdivision of population
- **Ideal For**: Multimodal problems and problems requiring multi-swarm approach

### SSA (Salp Swarm Algorithm)
- **Inspiration**: Swarming behavior of salps in deep oceans
- **Year**: 2017
- **Developer**: Mirjalili
- **Key Parameters**:
  - Parameter c1: Balances exploration and exploitation
- **Process**: Salps form chains with leader salp guiding the movement, and followers using chain formation rules
- **Strengths**: Simple implementation, good exploration capability
- **Ideal For**: Problems requiring both exploration and quick convergence

### GA (Genetic Algorithm)
- **Inspiration**: Natural selection and genetics
- **Year**: 1975
- **Developer**: Holland
- **Key Parameters**:
  - Crossover rate: Controls rate of information exchange
  - Mutation rate: Controls rate of random changes
  - Selection pressure: Controls selective pressure toward better solutions
- **Process**: Uses selection, crossover, and mutation to evolve population of solutions
- **Strengths**: Robust performance, well-suited for combinatorial problems
- **Ideal For**: Discrete optimization, combinatorial problems, complex landscapes

### HHO (Harris Hawks Optimization)
- **Inspiration**: Cooperative hunting behavior of Harris hawks
- **Year**: 2019
- **Developers**: Heidari et al.
- **Key Parameters**:
  - Energy of the rabbit (E): Controls transition between exploration/exploitation
  - Jump strength (J): Controls escaping behavior of prey
- **Process**: Hawks perform surprise pounce, varied dives, and team encircling to catch prey
- **Strengths**: Strong exploration capabilities, adaptive behavior
- **Ideal For**: Global optimization problems requiring high exploration

### SCA (Sine Cosine Algorithm)
- **Inspiration**: Mathematical sine and cosine functions
- **Year**: 2016
- **Developer**: Mirjalili
- **Key Parameters**:
  - Parameter r1: Controls search direction
  - Parameter r2: Controls search distance
  - Parameter r3: Adds randomness
  - Parameter r4: Controls switching between sine/cosine
- **Process**: Fluctuates solutions using sine and cosine functions to converge toward best solution
- **Strengths**: Mathematical foundation, balanced exploration/exploitation
- **Ideal For**: Problems with smooth objective functions

### JAYA (JAYA Algorithm)
- **Inspiration**: Sanskrit word meaning "victory"
- **Year**: 2016
- **Developer**: Rao
- **Key Parameters**: 
  - No algorithm-specific parameters (parameter-less)
- **Process**: Solutions move toward best solution and away from worst solution
- **Strengths**: Simple implementation, no control parameters, fast convergence
- **Ideal For**: Constraint optimization problems, problems requiring minimal tuning

### DE (Differential Evolution)
- **Inspiration**: Evolutionary process with vector differences
- **Year**: 1997
- **Developers**: Storn and Price
- **Key Parameters**:
  - Crossover rate (CR): Controls rate of crossover
  - Differential weight (F): Controls amplitude of difference vectors
  - Population size: Controls diversity
- **Process**: Creates new candidate solutions by combining existing ones with weighted differences
- **Strengths**: Effective for continuous function optimization, robust to noise
- **Ideal For**: Continuous nonlinear, non-differentiable, multimodal optimization problems

## 🛠️ How to Optimize Your Own Function

### Simple Custom Functions

```python
import numpy as np
from EvoloPy.optimizers import PSO

# Step 1: Define your objective function (minimize this)
def my_equation(x):
    # Example: Rosenbrock function
    sum_value = 0
    for i in range(len(x) - 1):
        sum_value += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2
    return sum_value

# Step 2: Set optimization parameters
lb = [-5] * 10       # v4.0+ supports list bounds
ub = [5] * 10
dim = 10
population = 40
iterations = 200

# Step 3: Run the optimizer
result = PSO.PSO(my_equation, lb, ub, dim, population, iterations)

# Step 4: Get and use the results
best_solution = result.bestIndividual
best_fitness = result.best_score  # v4.0+ provides best_score directly
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

### Complex Functions with Additional Data

```python
import numpy as np
from EvoloPy.optimizers import PSO

# For functions that need additional data, use a class-based approach
class MyOptimizationProblem:
    def __init__(self, data, weights):
        self.data = data
        self.weights = weights
    
    def objective_function(self, x):
        # Example: Weighted sum of squared error
        error = np.sum(self.weights * (self.data - x)**2)
        return error

# Create your optimization problem with data
my_data = np.random.rand(10)
my_weights = np.random.rand(10)
problem = MyOptimizationProblem(my_data, my_weights)

# Run optimization
result = PSO.PSO(
    problem.objective_function,
    lb=[-10] * 10,    # v4.0+ supports list bounds
    ub=[10] * 10, 
    dim=10, 
    PopSize=30, 
    iters=100
)

print(f"Best solution: {result.bestIndividual}")
print(f"Best fitness: {result.best_score}")  # v4.0+ exposes best_score directly
```

## 📁 Results Organization (v4.0+)

Version 4.0 introduces a more organized output structure:

```
results_[timestamp]/
├── [optimizer_name]/
│   └── [function_name]/
│       ├── config.txt                        # Configuration summary
│       ├── avg_results.csv                   # Average results across runs
│       ├── detailed_results.csv              # Detailed results for each run
│       ├── [optimizer]_[function]_boxplot.png    # Boxplot visualization
│       └── [optimizer]_[function]_convergence.png # Convergence plot
└── multiple_optimizers/
    └── ... (similar structure for multiple optimizers)
```

## 🚄 Parallel Processing (v3.0+)

Significantly speed up optimization by using multiple CPU cores or GPUs.

```python
from EvoloPy.api import run_optimizer

# Enable parallel processing
result = run_optimizer(
    optimizer="PSO",
    objective_func="F1",
    dim=30,
    lb=-100,
    ub=100,
    population_size=50,
    iterations=100,
    num_runs=10,
    enable_parallel=True,       # Enable parallel processing
    parallel_backend="auto",    # "auto", "multiprocessing", or "cuda"
    num_processes=None          # None = auto-detect optimal count
)
```

## 👨‍💻 Leadership & Credits

### EvoloPy v4.0 Refinements

Version 4.0 introduced several improvements:
- Consistent bounds handling (lists instead of scalars)
- Organized output folders with better structure
- Direct access to best fitness scores
- Enhanced visualization and statistics
- Improved handling of multiple runs

For details on v4.0 improvements, see [changes_v4.md](changes_v4.md).

### Parallel Processing System (v3.0)

The parallel processing system in v3.0 was implemented by **Jaber Jaber** ([@jaberjaber23](https://github.com/jaberjaber23)), providing:

- **Dramatic Performance Improvements**: Up to 20x speedup on large tasks
- **Multi-platform Support**: Utilizes both CPU cores and CUDA GPUs
- **Smart Hardware Detection**: Automatically configures optimal settings
- **Improved Scalability**: Handles much larger optimization problems
- **Excellent Developer Experience**: Simple API with automatic backend selection

For details on Jaber's parallel processing implementation, see [changes_v3.md](changes_v3.md).

### Original EvoloPy Concept

Original library concept and initial algorithms by Faris, Aljarah, Mirjalili, Castillo, and Guervós.

## 📄 Citation

If you use EvoloPy in your research, please cite:

```bibtex
@inproceedings{faris2016evolopy,
  title={EvoloPy: An Open-source Nature-inspired Optimization Framework in Python},
  author={Faris, Hossam and Aljarah, Ibrahim and Mirjalili, Seyedali and Castillo, Pedro A and Guervós, Juan Julián Merelo},
  booktitle={IJCCI (ECTA)},
  pages={171--177},
  year={2016}
}
```

## 📚 Benchmark Functions

The following table provides details about the benchmark functions available in EvoloPy:

| Function ID | Name                 | Formula                                     | Range              | Dimension | Properties                          |
|-------------|----------------------|---------------------------------------------|--------------------|-----------|------------------------------------|
| F1          | Sphere               | $f(x) = \sum_{i=1}^{n} x_i^2$              | [-100, 100]        | 30        | Unimodal, Separable                |
| F2          | Sum of Abs & Product | $f(x) = \sum\|x_i\| + \prod\|x_i\|$        | [-10, 10]          | 30        | Unimodal, Non-separable            |
| F3          | Sum of Squares       | $f(x) = \sum_{i=1}^{n} (\sum_{j=1}^{i} x_j)^2$ | [-100, 100]    | 30        | Unimodal, Non-separable            |
| F4          | Maximum              | $f(x) = \max_i\{\|x_i\|\}$               | [-100, 100]        | 30        | Unimodal, Non-separable            |
| F5          | Rosenbrock           | $f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]$ | [-30, 30] | 30 | Multimodal, Non-separable     |
| F6          | Shifted Absolute     | $f(x) = \sum_{i=1}^{n} \|x_i + 0.5\|^2$    | [-100, 100]        | 30        | Unimodal, Separable                |
| F7          | Sum of Powers        | $f(x) = \sum_{i=1}^{n} i \cdot x_i^4 + \text{random}[0,1)$ | [-1.28, 1.28] | 30 | Unimodal, Non-separable, Noisy |
| F8          | Sine Problem         | $f(x) = \sum_{i=1}^{n} -x_i \sin(\sqrt{\|x_i\|})$ | [-500, 500]  | 30        | Multimodal, Separable              |
| F9          | Rastrigin            | $f(x) = \sum_{i=1}^{n} [x_i^2 - 10\cos(2\pi x_i) + 10]$ | [-5.12, 5.12] | 30 | Multimodal, Separable           |
| F10         | Ackley               | $f(x) = -20e^{-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}} - e^{\frac{1}{n}\sum_{i=1}^{n}\cos(2\pi x_i)} + 20 + e$ | [-32, 32] | 30 | Multimodal, Non-separable |
| F11         | Griewank             | $f(x) = \frac{1}{4000}\sum_{i=1}^{n}x_i^2 - \prod_{i=1}^{n}\cos(\frac{x_i}{\sqrt{i}}) + 1$ | [-600, 600] | 30 | Multimodal, Non-separable |
| F12         | Penalized 1          | Complex formula with penalties              | [-50, 50]          | 30        | Multimodal, Non-separable          |
| F13         | Penalized 2          | Complex formula with penalties              | [-50, 50]          | 30        | Multimodal, Non-separable          |
| F14         | Shekel's Foxholes    | Complex formula                             | [-65.536, 65.536]  | 2         | Multimodal with many local minima  |
| F15         | Kowalik              | Complex formula                             | [-5, 5]            | 4         | Multimodal, Non-separable          |
| F16         | Six-Hump Camel       | Complex formula                             | [-5, 5]            | 2         | Multimodal, Non-separable          |
| F17         | Branin               | Complex formula                             | [-5, 15]           | 2         | Multimodal, Non-separable          |
| F18         | Goldstein-Price      | Complex formula                             | [-2, 2]            | 2         | Multimodal, Non-separable          |
| F19         | Hartman 3            | Complex formula                             | [0, 1]             | 3         | Multimodal, Non-separable          |
| F20         | Hartman 6            | Complex formula                             | [0, 1]             | 6         | Multimodal, Non-separable          |
| F21         | Shekel 5             | Complex formula                             | [0, 10]            | 4         | Multimodal, Non-separable          |
| F22         | Shekel 7             | Complex formula                             | [0, 10]            | 4         | Multimodal, Non-separable          |
| F23         | Shekel 10            | Complex formula                             | [0, 10]            | 4         | Multimodal, Non-separable          |
| ackley      | Ackley (Standalone)  | Same as F10                                | [-32.768, 32.768]  | 30        | Multimodal, Non-separable          |
| rosenbrock  | Rosenbrock (Standalone) | Same as F5                              | [-5, 10]           | 30        | Multimodal, Non-separable          |
| rastrigin   | Rastrigin (Standalone) | Same as F9                               | [-5.12, 5.12]      | 30        | Multimodal, Separable              |
| griewank    | Griewank (Standalone) | Same as F11                               | [-600, 600]        | 30        | Multimodal, Non-separable          |

## 📜 License

EvoloPy is licensed under the MIT License.


