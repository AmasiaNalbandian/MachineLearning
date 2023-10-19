import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

root_folder = "figures"

#Saves figures to their respective directories
def save_plot(plt, fitness_problem, size, plot):
    # Define the subfolder name based on fitness_problem and p_size
    sub_folder = os.path.join(root_folder, fitness_problem, plot)
    
    # Create the save folder (and its parents) if they don't exist
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    
    # Save the figure to the specified sub-folder
    filename = f"{fitness_problem}_{size}_{plot}.png"
    file_path = os.path.join(sub_folder, filename)
    plt.savefig(file_path)
    plt.close() 

# Send a list of curves to plot fitness/iteration
def fit_v_iteration(fitness_problem, size, curves, subtitle=""):
    title="Fitness vs. Iteration"
    plt.figure(figsize=(8, 6))
    
    for curve_data in curves:
        curve, label = curve_data['curve'], curve_data['label']
        plt.plot(curve[:, 0], label=label, linestyle='-')
        print(f"Final Fitness for {label} {subtitle}: {curve[-1, 0]}")


    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'{fitness_problem}\n{title}\n{subtitle}')
    plt.legend()
    plt.grid()
    # plt.show()
    save_plot(plt, fitness_problem, size, f"Fitness_vs_Iteration{subtitle}")

# Function Evaluations vs. Iteration
def fevals_v_iteration(fitness_problem, size, curves, subtitle=""):
    title="FEvals vs. Iteration"
    plt.figure(figsize=(8, 6))

    for curve_data in curves:
        curve, label = curve_data['curve'], curve_data['label']
        plt.plot(curve[:, 1], label=label, linestyle='-')
        print(f"Final FEvals for {fitness_problem} {label} : {curve[-1, 1]}")

    plt.xlabel('Iterations')
    plt.ylabel('FEvals')
    plt.title(f'{fitness_problem}\n{title}\n{subtitle}')
    plt.legend()
    plt.grid()
    # plt.show()
    save_plot(plt, fitness_problem, size, f"FEvals_vs_Iteration{subtitle}")

def timings_v_problem_size(fitness_problem, sizes, timings, subtitle=""):
    title = "Time vs. Problem Size"
    plt.figure(figsize=(8, 6))

    # Loop through timings and plot each algorithm's time against sizes
    for algo, time_log in timings.items():
        plt.plot(sizes, time_log, label=algo, linestyle='-')
        print(f"Final Time for {algo} on size {sizes[-1]}: {time_log[-1]}")

    plt.xlabel('Problem Size')
    plt.ylabel('Time (s)')
    plt.title(f'{fitness_problem}\n{title}')
    plt.legend()
    plt.grid()
    # Save the plot
    save_plot(plt, fitness_problem, sizes[-1], "Time_vs_ProblemSize{subtitle}")

