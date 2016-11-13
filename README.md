###EvoloPy: An open source nature-inspired optimization toolbox for global optimization in Python

The EvoloPy toolbox provides classical and recent nature-inspired metaheuristic for the global optimization. The list of optimizers that have been implemented includes Particle Swarm Optimization (PSO), Multi-Verse Optimizer (MVO), Grey Wolf Optimizer (GWO), and Moth Flame Optimization (MFO). The full list of implemented optimizers is available here https://github.com/7ossam81/EvoloPy/wiki/List-of-optimizers


##Features
- Six nature-inspired metaheuristic optimizers were implemented.
- The implimentation uses the fast array manipulation using `NumPy`.
- Matrix support using `SciPy`'s package.
- More optimizers is comming soon.

##Installation
- Python 3.xx is required.

Run

    pip3 install -r requirements.txt

(possibly with `sudo`)

That command above will install  `sklearn`, `NumPy` and `SciPy` for
you.

- If you are installing EvoloPy Toolbox onto Windows, please Install Anaconda from here https://www.continuum.io/downloads, which is the leading open data science platform powered by Python.
- If you are installing onto Ubuntu or Debian and using Python 3 then
  this will pull in all the dependencies from the repositories:
  
      sudo apt-get install python3-numpy python3-scipy liblapack-dev libatlas-base-dev libgsl0-dev fftw-dev libglpk-dev libdsdp-dev

##Get the source

Clone the Git repository from GitHub

    git clone https://github.com/7ossam81/EvoloPy.git


##Quick User Guide

EvoloPy toolbox contains twenty three benchamrks (F1-F23). The main file is the optimizer.py, which considered the interface of the toolbox. In the optimizer.py you can setup your experiment by selecting the optmizers, the benchmarks, number of runs, number of iterations, and population size. 
The following is a sample example to use the EvoloPy toolbox.  
To choose PSO optimizer for your experiment, change the PSO flag to true and others to false.  
```
Select optimizers:    
PSO= True  
MVO= False  
GWO = False  
MFO= False  
CS= False    
...
```
After that, Select benchmark function:
```
F1=True  
F2=False  
F3=False  
F4=False  
F5=False  
F6=False  
....  
```

Change NumOfRuns, PopulationSize, and Iterations variables as you want:  
```
NumOfRuns=10  
PopulationSize = 50  
Iterations= 1000
```

Now your experiment is ready to run. Enjoy!

##Contribute
- Issue Tracker: https://github.com/7ossam81/EvoloPy/issues  
- Source Code: https://github.com/7ossam81/EvoloPy

##Support

Use the [issue tracker](https://github.com/7ossam81/EvoloPy/issues). 


