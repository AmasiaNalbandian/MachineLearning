import matplotlib.pyplot as plt
import os

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


def fit_v_iteration(fitness_problem, size, rhc_curve, sa_curve, ga_curve, mimic_curve):
    plt.figure(figsize=(8, 6))
    plt.plot(rhc_curve[:, 0], label='RHC', linestyle='-')
    plt.plot(sa_curve[:, 0], label='SA', linestyle='-')
    plt.plot(ga_curve[:, 0], label='GA', linestyle='-')
    plt.plot(mimic_curve[:, 0], label='MIMIC', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'{fitness_problem}\nFitness vs. Iteration')
    plt.legend()
    plt.grid()
    # plt.show()
    save_plot(plt, fitness_problem, size, "Fitness_vs_Iteration")



# Function Evaluations vs. Iteration
def fevals_v_iteration(fitness_problem, size, rhc_curve, sa_curve, ga_curve, mimic_curve):
    plt.figure(figsize=(8, 6))
    plt.plot(rhc_curve[:, 1], label='RHC', linestyle='-')
    plt.plot(sa_curve[:, 1], label='SA', linestyle='-')
    plt.plot(ga_curve[:, 1], label='GA', linestyle='-')
    plt.plot(mimic_curve[:, 1], label='MIMIC', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'{fitness_problem}\nFEvals vs. Iteration')
    plt.legend()
    plt.grid()
    # plt.show()
    save_plot(plt, fitness_problem, size, "FEVals_vs_Iteration")


# Function Evaluations vs. Time
def fevals_vs_time(fitness_problem, p_size, rhc_curve, sa_curve, ga_curve, mimic_curve, rhc_time, sa_time, ga_time, mimic_time):
    plt.figure(figsize=(8, 6))
    plt.plot(rhc_time, rhc_curve[:, 1], label='RHC', linestyle='-')
    plt.plot(sa_time, sa_curve[:, 1], label='SA', linestyle='-')
    plt.plot(ga_time, ga_curve[:, 1], label='GA', linestyle='-')
    plt.plot(mimic_time, mimic_curve[:, 1], label='MIMIC', linestyle='-')
    plt.xlabel('Wall Clock Time (seconds)')
    plt.ylabel('FEvals')
    plt.title(f'{fitness_problem}\nFEvals vs. Wall Clock Time')
    plt.legend()
    plt.grid()
    # plt.show()

    save_plot(plt, fitness_problem, p_size, "FEvals_vs_Time")

