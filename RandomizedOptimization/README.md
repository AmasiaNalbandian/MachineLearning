# Randomized Optimization
This is a directory containing multiple files to allow for experimenting with 4 different randomized optimization algorithms. They are: 
1. RHC
2. Simulated Annealing
3. Genetic Algorithm
4. MIMIC

Trial runs are done by using 3 Fitness problems: 
1. nQueens
2. Knapsack
3. kColors


## Running Code
### Environment setup
To run the following code, you will need to have python installed. The specific version used for development was python 3.11.5. 

### Installing Packages
All the required packages are placed in the folder called `req.txt`. 

To install all packages run the following:
```
pip install -r req.txt
```

### Entry points: Notebooks(.ipynb)

There are three files to use as entry points. 

1. Running Hyperparameter Experiments

```python3 RO_fitness```

Make sure to add the values to test in the algorithms input parameters `run_<algorithm>()`

2. Running Problem Size Experiments

```python3 fitness_problems.py all```

You can replace all with nQueens, kColors or knapsacking to run only one.

3. NNOptimization.ipynb
Holds all the code to run the neural network with the optimizers. This uses a variety of different files to run different tuning, and algorithms.