import pickle 

from nsga import History, Population, Individual, non_dom_sort
from copy import deepcopy

from new_fds import temporal_converge, spatial_converge

history = History()

with open("individual_hist.pkl", "rb") as f:
    history.individual_hist = pickle.load(f)

map_copy = deepcopy(history.individual_hist)
for key, val in map_copy.items():
    genotype, fitnesses, generation = val
    if fitnesses[0] > 0.04:
        history.individual_hist.pop(key)

history.plot_data()

def create_individual(key, val):
    genotype, fitnesses, _ = val
    ind = Individual(genotype)
    ind.fitnesses = fitnesses
    ind.id = key
    return ind

individuals = [create_individual(key, val) for key, val in history.individual_hist.items()] 
pop = Population(individuals) 
fronts = non_dom_sort(pop) 
pareto_front = fronts[0]

R = 0.1
L = R * 10

def find_convergent_params_in_pareto_front(pareto_front):
    vals = []
    text = ""
    for individual in pareto_front:
        E, U, D, kA, _, _, _, _ = individual.genotype
        temp_conv = temporal_converge((R, L, E, U, D), [25, 50, 100, 200], 70)
        spat_conv = spatial_converge((R, L, E, U, D), [8, 16, 32, 64], 5000)
        vals.append((individual.genotype, temp_conv, spat_conv))
        text += f"E: {E}, U: {U}, D: {D}, kA: {kA}\nTemp converge output: {str(temp_conv)}\nSpat converge output: {str(spat_conv)}\n"
    with open("./test.txt", 'w') as f:
        f.write(text)
    return vals

print(find_convergent_params_in_pareto_front(pareto_front))