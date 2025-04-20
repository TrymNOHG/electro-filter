import random
import numpy as np
import math 
import uuid
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from fds import simulate_concentration

# Problem-specific constraints
BOUNDS = [
    (1.0, 3.0),     # E
    (1.0, 10.0),    # U
    (1.0, 2.0),     # D
    (1.0, 5.0)      # kA
]

R = 0.1
L = 10*R

# Representation constants
INDIVIDUAL_LENGTH = 2*len(BOUNDS)

# Self-adaptive mutation constants
NUM_STRATEGY_PARAMS = len(BOUNDS)
TAU_MARK = 1 / math.sqrt(2*NUM_STRATEGY_PARAMS)
TAU = 1 / math.sqrt(2*math.sqrt(NUM_STRATEGY_PARAMS)) # This parameter may be thought of as a sort of learning rate...
EPSILON_0 = 0.001

# Population constants
MU = 10
LAMBDA = 4

# Recombination constants
ALPHA = 0.5 # For intermediate crossover

# Example objective function
def variance_objective(deposition_c):
    """
        deposition_c - The concentration along the deposition wall.
    """
    return np.var(deposition_c) 

def deposition_objective(deposition_c):
    """
        deposition_c - The concentration along the deposition wall.
    """
    return sum(deposition_c) 

OBJECTIVES = [
    variance_objective, 
    deposition_objective
]

FITNESS_EVAL = [
    lambda x, y: x < y, # Objective 1 is a minimization problem
    lambda x, y: x > y  # 2 is max.
]



class Individual:
    def __init__(self, genotype):
        self.id = uuid.uuid1().int
        self.genotype = genotype
        self.fitnesses = None
        self.domination_count = 0
        self.dominates: list[int] = []
        self.crowd_distance = 0
        self.rank = math.inf
    
    def get_fitnesses(self):
        if self.fitnesses is None:
            self.run_sim() # Genotype to phenotype logic.
        return self.fitnesses
    
    def run_sim(self):
        E, U, D, kA = self.genotype[:4]
        Cwall = simulate_concentration(R, L, E, U, D, kA)
        final_profile = Cwall[-1, :]
        self.fitnesses = [
            obj(final_profile) for obj in OBJECTIVES
        ]
    
    def __repr__(self):
        return f"""
            ID: {self.id}
            Genotype: {self.genotype}
            Fitnesses: {self.fitnesses}
            Dominates: {self.dominates}
            Number dominated by: {self.domination_count}
            Rank: {self.rank}
        """
    
class Population:
    def __init__(self, population: list = []):
        self.population: list[Individual] = population
        self.individual_map: map[int, Individual] = {}
        if len(self.population) > 0:
            self.update_map()

    def add_individuals(self, individuals: list[Individual]):
        for individual in individuals:
            self.individual_map[individual.id] = individual
            self.population.append(individual)
    
    def update_map(self):
        for individual in self.population:
            if self.individual_map.get(individual.id, None) is None:
                self.individual_map[individual.id] = individual
        return self.individual_map
    
    def get_stats(self):
        average_fitnesses = [0 for _ in range(len(OBJECTIVES))]
        best_fitnesses = [math.inf, -math.inf] # Min-max problem assumed
        for individual in self.population:
            for i, fitness in enumerate(individual.get_fitnesses()):
                average_fitnesses[i] += fitness
                if FITNESS_EVAL[i](fitness, best_fitnesses[i]):
                    best_fitnesses[i] = fitness
                
        return [avg_fit / len(self.population) for avg_fit in average_fitnesses], best_fitnesses
    
    def reset_front_data(self):
        for individual in self.population:
            individual.rank = math.inf
            individual.crowd_distance = 0
    
    def __repr__(self):
        output = ""
        for individual in self.population:
            output += f"{individual}\n"
        return output
    
class History:
    def __init__(self):
        self.individual_hist = {} # id: (genotype, fitnesses, generation)
        self.gen_hist = []
    
    def add_gen_data(self, population: Population):
        for individual in population.population:
            self.individual_hist[individual.id] = (individual.genotype, individual.get_fitnesses(), len(self.gen_hist))
        self.gen_hist.append(pop.get_stats())
    
    def plot_data(self):
        individ_data = self.individual_hist.values()
        generation_vals = [data[2] for data in individ_data]
        fitness_data = np.array([data[1] for data in individ_data]) 
        plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % cmap.N) for i in generation_vals] 
        plt.figure(figsize=(8, 6))
        plt.scatter(fitness_data[:, 0], fitness_data[:, 1], c=colors)
        plt.xlabel('Variance Objective')
        plt.ylabel('Sum Deposition Objective')
        plt.title('NSGA-2 Population Development')
        plt.grid(True)

        for gen_val in np.unique(generation_vals):
            plt.scatter([], [], c=[cmap(gen_val % cmap.N)], label=f"Gen. {gen_val}")
        plt.legend(title='Generation Number', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig("./history.png")


def init_individual():
    """
        This function generates an individual of the form: [x_1, ..., x_n, σ_1, ... σ_n]
    """
    individual = []
    for i in range(len(BOUNDS)):
        lower, upper = BOUNDS[i]
        individual.insert(i, random.uniform(lower, upper)) # x_i
        individual.append(np.random.normal(0, 0.1)) # σ_i
    
    return Individual(np.array(individual))

def init_population(num_individuals):
    return [init_individual() for _ in range(num_individuals)]

def real_mutate(individual: Individual):
    new_genotype = deepcopy(individual.genotype)
    normal_tau_mark = TAU_MARK * np.random.normal(0, 1)
    for i in range(NUM_STRATEGY_PARAMS):
        new_sigma = new_genotype[-NUM_STRATEGY_PARAMS+i] * np.exp(normal_tau_mark + TAU * np.random.normal(0, 1))
        new_genotype[-NUM_STRATEGY_PARAMS+i] = new_sigma if new_sigma > EPSILON_0 else EPSILON_0
        new_genotype[i] += new_genotype[-NUM_STRATEGY_PARAMS+i] * np.random.normal(0, 1) # Should I ensure it stays within the bounds here?
    return Individual(new_genotype)

def tournament_selection(population, num_parents=LAMBDA, k=2):
    parents = []
    num_parents = 0
    while num_parents < 2*LAMBDA:
        sample = [population[random.randint(0, len(population) - 1)] for _ in range(k)]
        if sample[0].rank == sample[1].rank: # If ranking same, use crowding distance
            parents.append(sample[0] if sample[0].crowd_distance > sample[1].crowd_distance else sample[1]) # Higher crowding distance is better?
        else:
            parents.append(sample[0] if sample[0].rank < sample[1].rank else sample[1])
        num_parents += 1
    return parents


def recombination(parents):
    offspring = []
    for i in range(0, len(parents)-1, 2):
        parent_1 = parents[i]
        parent_2 = parents[i+1]
        offspring.append(uni_inter_xover(parent_1, parent_2))
    return offspring

def uni_inter_xover(individual_1: Individual, individual_2: Individual):
    # recomb_point = random.randint(0, len(individual_1) - 1) # This could be used if I want to test with simple uniform intermediate recombination operator.
    new_genotype = individual_1.genotype * ALPHA + individual_2.genotype * (1-ALPHA)
    return Individual(deepcopy(new_genotype))

def swap_xover(individual_1: Individual, individual_2: Individual):
    new_genotype = deepcopy(individual_1.genotype)
    for i in len(range(individual_1.genotype)):
        if random.random() < ALPHA:
            new_genotype[i] = individual_2.genotype[i]
    return Individual(new_genotype)

def does_dominate(individual_1: Individual, individual_2: Individual):
    # Check if individual_1 is in individual_2 or vice versa to avoid additional comp
    if individual_2.id in individual_1.dominates or individual_1.id in individual_2.dominates:
        return
    
    val_1 = True
    val_2 = False
    val_1_eval = [lambda x, y: x  <= y, lambda x, y: x  >= y, ]
    for i in range(len(individual_1.get_fitnesses())):
        val_1 &= val_1_eval[i](individual_1.get_fitnesses()[i], individual_2.get_fitnesses()[i]) # This assumes min-min problem. In reality, we have a min-max problem, so will need to change this.
        val_2 |= FITNESS_EVAL[i](individual_1.get_fitnesses()[i], individual_2.get_fitnesses()[i])

    if not val_1 and not val_2:
        individual_2.dominates.append(individual_1.id)
        individual_1.domination_count += 1
    elif val_1 and val_2:
        individual_1.dominates.append(individual_2.id)
        individual_2.domination_count += 1
    else:
        pass
        
    return

def build_front(current_front: list[Individual], id_to_obj: map, rank: int):
    potential_next_front = set()
    for individual in current_front:
        for other_individual_id in individual.dominates:
            if other_individual_id in id_to_obj.keys():
                id_to_obj[other_individual_id].domination_count -= 1
                potential_next_front.add(other_individual_id)

    actual_next_front = []
    for idx in potential_next_front:
        if id_to_obj[idx].domination_count == 0:
            # print(f"id: {idx}")
            id_to_obj[idx].rank = rank
            actual_next_front.append(id_to_obj[idx])
   
    if len(actual_next_front) == 0:
        return [current_front]
    else: 
        new_front = [current_front] + build_front(actual_next_front, id_to_obj, rank+1)
        # new_front.extend(build_front(actual_next_front, id_to_obj, rank+1))
        return new_front


def non_dom_sort(pop: Population):
    """
        This function sorts the population based on non-dominating frontiers.
        An individual dominates another individual if it is better than that individual in at least one objective and never worse than.
    """

    for i, individual in enumerate(pop.population):
        for j, other_individual in enumerate(pop.population):
            if i == j:
                continue
            else:
                does_dominate(individual, other_individual) # Use this to sort.

    initial_front = []
    for individual in pop.population:
        individual.rank = math.inf
        if individual.domination_count == 0:
            individual.rank = 0
            initial_front.append(individual)

    fronts = build_front(initial_front, pop.individual_map, 1)    
    return fronts


def sort_by_crowding_distance(front: list[Individual]):
    for i in range(len(OBJECTIVES)):
        front = sorted(front, key=lambda x: x.get_fitnesses()[i], reverse=True)
        front[0].crowd_distance = math.inf
        front[-1].crowd_distance = math.inf
        for j, individual in enumerate(front[1:-1]):
            individual.crowd_distance += (front[j+1].get_fitnesses()[i] - front[j-1].get_fitnesses()[i]) / (front[0].get_fitnesses()[i] - front[-1].get_fitnesses()[i])
    
    return sorted(front, key=lambda x: x.crowd_distance, reverse=True) # Make sure this is ordered correctly

def select_survivors(population: Population):
    fronts = non_dom_sort(population)
    new_pop_list = []
    for front in fronts:
        if len(front) < MU - len(new_pop_list):
            new_pop_list.extend(front)
        else:
            new_pop_list.extend(sort_by_crowding_distance(front)[:MU - len(new_pop_list)])
    return Population(new_pop_list)

if __name__ == "__main__":
    hist = History()
    pop = Population(init_population(15)) # Initialization
    hist.add_gen_data(pop)
    non_dom_sort(pop)
    NUM_GENERATIONS = 2
    for i in range(NUM_GENERATIONS):
        parents = tournament_selection(pop.population) # Parent Selection
        offspring = recombination(parents) # Offspring
        pop.add_individuals(offspring)
        
        mutated_individuals = []
        for i, individual in enumerate(pop.population):
            mutated_individuals.append(real_mutate(individual))
        pop.add_individuals(mutated_individuals)

        pop.reset_front_data()
        pop = select_survivors(pop)

        # Calculate statistics around the population...
        print("Average Fitnesses | Best Fitnesses")
        print(pop.get_stats())
        hist.add_gen_data(pop)

    hist.plot_data()

# TODO: Discourage variable bound violations through objective penalty.