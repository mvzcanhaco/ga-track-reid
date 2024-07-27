import pygame
import cv2

def initialize_pygame():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Genetic Algorithm Progress")
    font = pygame.font.Font(None, 36)
    return screen, font

def draw_frame(screen, font, frame, generation=0, best_params={}, best_fitness=0):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    screen.fill((255, 255, 255))

    text = font.render(f"Generation: {generation}", True, (0, 0, 0))
    screen.blit(text, (50, 50))

    text = font.render(f"Best Fitness: {best_fitness}", True, (0, 0, 0))
    screen.blit(text, (50, 100))

    y = 150
    for param, value in best_params.items():
        param_text = font.render(f"{param}: {value}", True, (0, 0, 0))
        screen.blit(param_text, (50, y))
        y += 50

    if frame is not None:
        frame_surface = pygame.surfarray.make_surface(cv2.transpose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        screen.blit(frame_surface, (400, 50))

    pygame.display.flip()

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
