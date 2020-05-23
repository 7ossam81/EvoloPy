# EvoloPy: An open source nature-inspired optimization toolbox for global optimization in Python

The EvoloPy toolbox provides classical and recent nature-inspired metaheuristic for the global optimization. The list of optimizers that have been implemented includes Particle Swarm Optimization (PSO), Multi-Verse Optimizer (MVO), Grey Wolf Optimizer (GWO), and Moth Flame Optimization (MFO). The full list of implemented optimizers is available here https://github.com/7ossam81/EvoloPy/wiki/List-of-optimizers


## Features
- Six nature-inspired metaheuristic optimizers were implemented.
- The implimentation uses the fast array manipulation using `NumPy`.
- Matrix support using `SciPy`'s package.
- More optimizers is comming soon.

## Installation
- Python 3.xx is required.

Run

    pip3 install -r requirements.txt

(possibly with `sudo`)

That command above will install  `sklearn`, `NumPy`, and `SciPy` for
you.

- If you are installing EvoloPy Toolbox onto Windows, please Install Anaconda from here https://www.continuum.io/downloads, which is the leading open data science platform powered by Python.
- If you are installing onto Ubuntu or Debian and using Python 3 then
  this will pull in all the dependencies from the repositories:
  
      sudo apt-get install python3-numpy python3-scipy liblapack-dev libatlas-base-dev libgsl0-dev fftw-dev libglpk-dev libdsdp-dev

## Get the source

Clone the Git repository from GitHub

    git clone https://github.com/7ossam81/EvoloPy.git


## Quick User Guide

EvoloPy toolbox contains twenty three benchamrks (F1-F23). The main file is the optimizer.py, which considered the interface of the toolbox. In the optimizer.py you can setup your experiment by selecting the optmizers, the benchmarks, number of runs, number of iterations, and population size. 
The following is a sample example to use the EvoloPy toolbox.  
Select optimizers from the list of available ones: "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS","HHO","SCA","JAYA","DE". For example:
```
optimizer=["SSA","PSO","GA"]  
```

After that, Select benchmark function from the list of available ones: "F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19". For example:
```
objectivefunc=["F3","F4"]  
```

Select number of repetitions for each experiment. To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.  For example:
```
NumOfRuns=10  
```
Select general parameters for all optimizers (population size, number of iterations). For example:
```
params = {'PopulationSize' : 30, 'Iterations' : 50}
```
Choose whether to Export the results in different formats. For example:
```
export_flags = {'Export_avg':True, 'Export_details':True, 'Export_convergence':True, 'Export_boxplot':True}
```

Now your experiment is ready to run. Enjoy!

## Contribute
- Issue Tracker: https://github.com/7ossam81/EvoloPy/issues  
- Source Code: https://github.com/7ossam81/EvoloPy

## Useful Links
- Video Demo:https://www.youtube.com/watch?v=8t10SyrhDjQ
- Paper source: https://github.com/7ossam81/EvoloPy
- Paper: https://www.scitepress.org/Papers/2016/60482/60482.pdf
- Poster source: https://github.com/7ossam81/EvoloPy-poster
- Live Demo: http://evo-ml.com/evolopy-live-demo/

## List of contributors

## Reference

For more information about EvoloPy, please refer to our paper: 

Faris, Hossam, Ibrahim Aljarah, Seyedali Mirjalili, Pedro A. Castillo, and Juan Julián Merelo Guervós. "EvoloPy: An Open-source Nature-inspired Optimization Framework in Python." In IJCCI (ECTA), pp. 171-177. 2016.
https://www.scitepress.org/Papers/2016/60482/60482.pdf

Please include the following related citations:

- Qaddoura, Raneem, Hossam Faris, Ibrahim Aljarah, and Pedro A. Castillo. "EvoCluster: An Open-Source Nature-Inspired Optimization Clustering Framework in Python." In International Conference on the Applications of Evolutionary Computation (Part of EvoStar), pp. 20-36. Springer, Cham, 2020.
- Ruba Abu Khurma, Ibrahim Aljarah, Ahmad Sharieh, and Seyedali Mirjalili. Evolopy-fs: An open-source nature-inspired optimization framework in python for feature selection. In Evolutionary Machine Learning Techniques, pages 131–173. Springer, 2020



## Support

Use the [issue tracker](https://github.com/7ossam81/EvoloPy/issues). 


