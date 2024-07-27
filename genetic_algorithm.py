import random
import numpy as np
import csv
import os
from visualization import draw_frame, update_graph

param_ranges = {
    'confidence_threshold': (0.5, 0.9),
    'scale_factor': (0.5, 2.0),
    'brightness': (-50, 50),
    'contrast': (0.5, 2.0),
    'max_cosine_distance': (0.1, 0.4),
    'nn_budget': (50, 150)
}


def initialize_population(pop_size, initial_params):
    population = [initial_params]
    for _ in range(pop_size - 1):  # Reduzir em um o tamanho da população, já que adicionamos manualmente um indivíduo
        individual = {
            'confidence_threshold': random.uniform(param_ranges['confidence_threshold'][0],
                                                   param_ranges['confidence_threshold'][1]),
            'max_cosine_distance': random.uniform(param_ranges['max_cosine_distance'][0],
                                                  param_ranges['max_cosine_distance'][1]),
            'scale_factor': random.uniform(param_ranges['scale_factor'][0],
                                           param_ranges['scale_factor'][1]),
            'brightness': random.uniform(param_ranges['brightness'][0], param_ranges['brightness'][1]),
            'contrast': random.uniform(param_ranges['contrast'][0], param_ranges['contrast'][1]),
            'nn_budget': 100
        }
        population.append(individual)
    return population


def crossover(parent1, parent2):
    child = {}
    keys = list(parent1.keys())
    crossover_point1 = random.randint(0, len(keys) - 1)
    crossover_point2 = random.randint(crossover_point1, len(keys) - 1)

    for i, key in enumerate(keys):
        if crossover_point1 <= i <= crossover_point2:
            child[key] = parent2[key]
        else:
            child[key] = parent1[key]

    return child


def mutate(individual, mutation_rate, generation, max_generations):
    adaptive_mutation_rate = mutation_rate * (1 - (generation / max_generations))
    for key in individual.keys():
        if random.random() < adaptive_mutation_rate:
            if key == 'scale_factor':
                individual[key] = random.uniform(param_ranges[key][0], param_ranges[key][1])
            else:
                if isinstance(param_ranges[key], tuple):
                    value_range = param_ranges[key][1] - param_ranges[key][0]
                    mutation = np.random.normal(0, value_range * 0.1)
                    individual[key] = np.clip(individual[key] + mutation, param_ranges[key][0], param_ranges[key][1])
                elif isinstance(param_ranges[key], list):
                    individual[key] = random.choice(param_ranges[key])
    return individual


def genetic_algorithm(pop_size, generations, num_select, evaluate_models, initial_params, screen, font, progress_file,
                      accuracies, best_fitnesses, mutation_rate=0.1, elitism_rate=0.1):
    global best_individual, best_fitness
    population = initialize_population(pop_size, initial_params)

    with open(progress_file, 'w') as f:
        f.write("generation,best_fitness,confidence_threshold,max_cosine_distance,scale_factor,brightness,contrast\n")

    num_elites = int(pop_size * elitism_rate)

    for generation in range(generations):
        print(f"Processing Generation {generation + 1}/{generations}")
        fitnesses = np.array([evaluate_models(ind) for ind in population])
        best_fitness = np.max(fitnesses)
        best_individual = population[np.argmax(fitnesses)]
        print(f"Best fitness: {best_fitness}, Best params: {best_individual}")

        accuracies.append(fitnesses)
        best_fitnesses.append(best_fitness)

        # Seleção com elitismo
        elite_indices = np.argsort(fitnesses)[-num_elites:]
        elites = [population[i] for i in elite_indices]

        selected_indices = np.argsort(fitnesses)[-num_select:]
        selected_individuals = [population[i] for i in selected_indices]

        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, generation, generations)
            new_population.append(child)

        population = new_population

        draw_frame(screen, font, None, generation + 1, best_individual, best_fitness)
        # update_graph(screen, font, accuracies, generations, best_fitnesses)

        with open(progress_file, 'a') as f:
            f.write(
                f"{generation},{best_fitness},{best_individual['confidence_threshold']},{best_individual['max_cosine_distance']},{best_individual['scale_factor']},{best_individual['brightness']},{best_individual['contrast']}\n")

    return best_individual, best_fitness