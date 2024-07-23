import random
import numpy as np
import csv
import os
from visualization import draw_frame

param_ranges = {
    'confidence_threshold': (0.5, 0.9),
    'image_resize_dim': [224, 256, 512],  # Updated to use a list
    'brightness': (-50, 50),
    'contrast': (0.5, 2.0),
    'max_cosine_distance': (0.1, 0.4)
}

def create_population(pop_size, initial_params):
    population = []
    population.append(initial_params)
    for _ in range(pop_size-1):
        individual = {
            'confidence_threshold': initial_params['confidence_threshold'] + random.uniform(-0.5, 0.5),
            'max_cosine_distance': initial_params['max_cosine_distance'] + random.uniform(-0.1, 0.1),
            'image_resize_dim': initial_params['image_resize_dim'] + random.choice([-32, 32]),
            'brightness': initial_params['brightness'] + random.uniform(-50, 50),
            'contrast': initial_params['contrast'] + random.uniform(-0.5, 0.5),
        }
        population.append(individual)
    return population

def fitness_function(individual, evaluate_models):
    return evaluate_models(individual)

def select(population, fitnesses, num_select):
    selected = []
    selected_indices = np.argsort(fitnesses)[-num_select:]
    for i in selected_indices:
        selected.append(population[i])
    return selected

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child

def mutate(chromosome, mutation_rate=0.01):
    for key in chromosome:
        if random.random() < mutation_rate:
            if isinstance(param_ranges[key], tuple):
                chromosome[key] = random.uniform(*param_ranges[key])
            else:
                chromosome[key] = random.choice(param_ranges[key])
    return chromosome

def save_progress(generation, best_individual, best_fitness, file_path='genetic_progress.csv'):
    fieldnames = ['generation', 'best_fitness'] + list(best_individual.keys())
    write_header = not os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {'generation': generation, 'best_fitness': best_fitness}
        row.update(best_individual)
        writer.writerow(row)

def genetic_algorithm(pop_size, generations, num_select, evaluate_models, initial_params, screen, font, progress_file='genetic_progress.csv'):
    population = create_population(pop_size, initial_params)
    for generation in range(generations):
        print(f"Processing Generation {generation + 1}/{generations}")
        fitnesses = np.array([fitness_function(ind, evaluate_models) for ind in population])
        best_fitness = np.max(fitnesses)
        best_individual = population[np.argmax(fitnesses)]
        save_progress(generation, best_individual, best_fitness, progress_file)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # Call draw_frame to update Pygame window with the best frame
        draw_frame(screen, font, None, generation, best_individual, best_fitness)

        selected = select(population, fitnesses, num_select)
        offspring = []
        for i in range(len(selected) // 2):
            parent1, parent2 = selected[i * 2], selected[i * 2 + 1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent1, parent2)
            offspring.extend([mutate(child1), mutate(child2)])
        population = offspring
    best_individual = population[np.argmax(fitnesses)]
    best_fitness = np.max(fitnesses)
    return best_individual, best_fitness
