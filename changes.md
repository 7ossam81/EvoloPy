# EvoloPy – SA Addition

This is a clone of the [EvoloPy](https://github.com/7ossam81/EvoloPy) optimization framework, originally developed by Faris, Aljarah, Mirjalili, Castillo, and Guervós. All credit for the original framework and its algorithms belongs to the original authors.

## What was added

A new optimizer, **Simulated Annealing Optimization for Continuous problems (SA)**, has been implemented and added to the `optimizers` folder as `SA.py`. The algorithm is based on the work of Scott Kirkpatrick, C. Daniel Gelatt Jr., and Mario P. Vecchi, published in the journal Science in May 1983. and follows the same interface conventions as the existing optimizers in the package (PSO, GWO, etc.).

### Implementation notes


- **Initial temperature `(T0)`**: When `T0` is not provided, it is estimated automatically from preliminary uphill moves. The mean positive change in objective value is used together with the target acceptance probability to calculate an initial temperature. This avoids requiring a problem-specific temperature to be manually selected.

- **Gaussian neighborhood**: New candidate solutions are generated using Gaussian perturbations around the current solution. The standard deviation is defined relative to the search range as `step_size * (ub - lb)`, allowing the neighborhood size to automatically adapt to the scale of each decision variable.

- **Move acceptance**: Improving or equal solutions `(ΔE ≤ 0)` are always accepted. For worsening solutions `(ΔE > 0)`, the Metropolis probability `exp(-ΔE / T)` determines whether the candidate is accepted. This allows SA to escape local optima, particularly at higher temperatures.

- **Boundary handling**: Candidate solutions are clipped to the predefined lower and upper bounds after every perturbation. This ensures that all evaluated solutions remain within the feasible search space.
- **Equilibrium iterations**: Rather than cooling after every individual move, equilibrium neighboring solutions are evaluated at each temperature. This allows the algorithm to explore the current temperature level before reducing the temperature.

- **Geometric cooling**: The temperature is updated according to `T_new = alpha * T`. With `alpha = 0.9`, the temperature gradually decreases, reducing the probability of accepting worsening solutions as the search progresses.

- **Stopping conditions**: The optimization terminates when either the temperature falls below Tmin or the maximum number of temperature levels `(max_temperatures)` is reached. This prevents excessively long runs if the cooling schedule requires many iterations.

- **Convergence history**: The best fitness found after each temperature level is stored in history. This provides a convergence curve showing how the best solution improves as the temperature decreases.

- **Single-solution population**: Unlike population-based metaheuristics, SA maintains only one active solution (current) at a time. The best solution is maintained separately so that a previously discovered optimum is not lost when the algorithm accepts a worse solution.

One important point in this implementation is that step_size and alpha control different aspects of the search: step_size determines how far the algorithm explores around the current solution, while alpha determines how quickly the algorithm transitions from exploration to exploitation.


## Usage

```python
from EvoloPy.optimizers.SA import SA

result = ACO(
    objf=my_function,
    lb=-100,
    ub=100,
    dim=30,
    equilibrium=100,
    x0=None,
    T0=None,
    Tmin=1e-3,
    alpha=0.9,
    step_size=0.1,
    max_temperatures=10000,
    target_acceptance=0.8,
    seed=None

)

print(result.best_score)
print(result.bestIndividual)
```
## References

- Kirkpatrick, Scott, C. Daniel Gelatt Jr, and Mario P. Vecchi. "Optimization by simulated annealing." *science 220.4598 (1983): 671-680*.
- El-Ghazali, Talbi. "Metaheuristics: from design to implementation." Jonh Wiley and Sons Inc., Chichester 9 (2009): 10-11.
