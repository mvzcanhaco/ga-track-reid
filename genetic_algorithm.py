import random
import numpy as np
import csv
import os
from visualization import draw_frame

param_ranges = {
    'confidence_threshold': (0.5, 0.9),
    'scale_factor': (0.5, 2.0),
    'brightness': (-50, 50),
    'contrast': (0.5, 2.0),
    'max_cosine_distance': (0.1, 0.4)
}

def initialize_population(pop_size, initial_params):
    population = []
    # Adicionar um indivíduo com os parâmetros iniciais fornecidos
    population.append(initial_params)

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
        }
        population.append(individual)
    return population

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(individual, mutation_rate, generation, max_generations):
    adaptive_mutation_rate = mutation_rate * (1 - (generation / max_generations))
    for key in individual.keys():
        if random.random() < adaptive_mutation_rate:  # Taxa de mutação adaptativa
            if key == 'scale_factor':
                individual[key] = random.uniform(param_ranges[key][0], param_ranges[key][1])
            else:
                individual[key] = random.uniform(param_ranges[key][0], param_ranges[key][1])
    return individual

def genetic_algorithm(pop_size, generations, num_select, evaluate_models, initial_params, screen, font, progress_file,
                      mutation_rate=0.1):
    global best_individual, best_fitness
    population = initialize_population(pop_size, initial_params)

    # Escrever cabeçalho no arquivo CSV
    with open(progress_file, 'w') as f:
        f.write(
            "generation,best_fitness,confidence_threshold,max_cosine_distance,scale_factor,brightness,contrast\n")

    for generation in range(generations):
        print(f"Processing Generation {generation + 1}/{generations}")
        fitnesses = np.array([evaluate_models(ind) for ind in population])
        best_fitness = np.max(fitnesses)
        best_individual = population[np.argmax(fitnesses)]
        print(f"Best fitness: {best_fitness}, Best params: {best_individual}")

        # Seleção
        selected_indices = np.argsort(fitnesses)[-num_select:]
        selected_individuals = [population[i] for i in selected_indices]

        # Reprodução
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, generation, generations)
            new_population.append(child)

        population = new_population

        # Visualização do progresso
        draw_frame(screen, font, None, generation + 1, best_individual, best_fitness)

        # Salvar progresso no arquivo CSV
        with open(progress_file, 'a') as f:
            f.write(
                f"{generation},{best_fitness},{best_individual['confidence_threshold']},{best_individual['max_cosine_distance']},{best_individual['scale_factor']},{best_individual['brightness']},{best_individual['contrast']}\n")

    return best_individual, best_fitness
