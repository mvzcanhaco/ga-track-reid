import pygame
import matplotlib.pyplot as plt
import numpy as np
import io
import cv2


# Inicializa o Pygame
def initialize_pygame():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Genetic Algorithm Progress")
    font = pygame.font.Font(None, 36)
    return screen, font


def draw_frame(screen, font, frame, generation=0, best_params=None, best_fitness=0, fitness_history=None):
    if best_params is None:
        best_params = {}
    screen.fill((255, 255, 255))

    # Display the current generation and best fitness
    gen_text = font.render(f'Generation: {generation}', True, (0, 0, 0))
    screen.blit(gen_text, (10, 10))

    fitness_text = font.render(f'Best Fitness: {best_fitness:.4f}', True, (0, 0, 0))
    screen.blit(fitness_text, (10, 40))

    params_text = font.render(f'Params: {best_params}', True, (0, 0, 0))
    screen.blit(params_text, (10, 70))

    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (200, 300))  # Resize for display
        frame_surface = pygame.surfarray.make_surface(frame)
        screen.blit(frame_surface, (400, 50))

    # if fitness_history is not None:
    #     plt.figure(figsize=(6, 4))
    #     x_labels = [f'G{gen + 1}_P{pop + 1}' for gen in range(len(fitness_history)) for pop in
    #                 range(len(fitness_history[gen]))]
    #     y_values = [fitness for gen in fitness_history for fitness in gen]
    #     plt.plot(x_labels, y_values, marker='o')
    #     plt.xlabel('Population')
    #     plt.ylabel('Fitness')
    #     plt.tight_layout()
    #     plt.savefig('fitness_graph.png')
    #     plt.close()
    #
    #     fitness_img = pygame.image.load('fitness_graph.png')
    #     fitness_img = pygame.transform.scale(fitness_img, (600, 400))
    #     screen.blit(fitness_img, (350, 100))
    pygame.display.flip()


# Atualiza o gráfico de progresso dinâmico
def update_graph(screen, font, accuracies, generations, best_fitnesses):
    plt.clf()
    x_labels = [f"G{gen + 1}_P{pop + 1}" for gen in range(generations) for pop in range(len(accuracies[0]))]
    y_values = [acc for gen in accuracies for acc in gen]

    plt.plot(x_labels, y_values, marker='o')
    plt.xticks(rotation=90)

    for i, bf in enumerate(best_fitnesses):
        plt.axvline(x=i * len(accuracies[0]), color='r', linestyle='--', label=f'Best Fitness G{i + 1}: {bf:.4f}')
    plt.legend()
    plt.tight_layout()

    graph_surface = plot_to_pygame_surface()
    screen.blit(graph_surface, (400, 0))
    pygame.display.update()


# Converte o gráfico Matplotlib para uma superfície Pygame
def plot_to_pygame_surface():
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image = pygame.image.load(buffer)
    buffer.close()
    return image


def draw_bounding_boxes(frame, detections):
    for detection in detections:
        bbox = detection.to_tlbr()
        track_id = detection.track_id
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)


def visualize_progress(generation, best_params, best_fitness, frame=None):
    screen, font = initialize_pygame()
    draw_frame(screen, font, frame, generation, best_params, best_fitness)


# Update Pygame event loop to prevent window from closing
def update_pygame_event_loop():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
