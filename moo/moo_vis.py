import pickle 

from nsga import History, Population, Individual, non_dom_sort
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


history = History()

with open("individual_hist.pkl", "rb") as f:
    history.individual_hist = pickle.load(f)

map_copy = deepcopy(history.individual_hist)
for key, val in map_copy.items():
    genotype, fitnesses, generation = val
    if fitnesses[0] > 0.04:
        history.individual_hist.pop(key)
    # print(val)

history.plot_data()

def create_individual(key, val):
    genotype, fitnesses, _ = val
    ind = Individual(genotype)
    ind.fitnesses = fitnesses
    ind.id = key
    return ind

individuals = [create_individual(key, val) for key, val in history.individual_hist.items()] # Reconstruct individuals from individual_hist
pop = Population(individuals) # Reconstruct population
fronts = non_dom_sort(pop) # Look at the Pareto front and use this to create a cool plot. 
pareto_front = fronts[0]

single_obj_ind = Individual(genotype=(3.0, 9.357549135260617, 1.0, 1.3683146573615916))
# single_obj_ind.get_fitnesses()
single_obj_ind.fitnesses = [np.float64(0.005000676291607206), np.float64(77.71939731352629)]

def visualize_moo(individual_hist, pareto_front, single_obj_ind):
    individ_data = individual_hist.values()
    # generation_vals = [data[2] for data in individ_data]
    fitness_data = np.array([data[1] for data in individ_data] + [single_obj_ind.fitnesses])

    colors = []
    for key in individual_hist.keys():
        added = False
        for individual in pareto_front:
            if key == individual.id:
                colors.append('red')
                added = True
                break
        if not added:
                colors.append('blue')

    colors.append('green')
    plt.figure(figsize=(8, 6))
    plt.scatter(fitness_data[:, 0], fitness_data[:, 1], c=colors)
    plt.xlabel('Variance Objective')
    plt.ylabel('Sum Deposition Objective')
    plt.yscale('log')
    plt.title('NSGA-2 Population Development')
    plt.grid(True)


    plt.scatter([], [], color='red', label='Pareto Front')
    plt.scatter([], [], color='green', label='Single Objective')
    plt.scatter([], [], color='blue', label='Other')
    plt.legend(title='Point Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("./moo.png")

visualize_moo(history.individual_hist, pareto_front, single_obj_ind)