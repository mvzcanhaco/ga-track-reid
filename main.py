import os
import cv2
import numpy as np
import torch
from tensorflow.keras.applications import MobileNetV2
from deep_sort_realtime.deepsort_tracker import DeepSort
from genetic_algorithm import genetic_algorithm, individual_to_string
from processing_functions import simplified_evaluate_metrics, process_image, process_video
from visualization import initialize_pygame, draw_frame, update_pygame_event_loop
from ultralytics import YOLO

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO('yolov8s.pt')  # Certifique-se de ter o modelo YOLOv8 baixado e no caminho correto
print("YOLOv8 model loaded successfully.")

# Load MobileNetV2 model for Re-id
print("Loading MobileNetV2 model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
print("MobileNetV2 model loaded successfully.")


def initialize_deepsort(max_cosine_distance):
    return DeepSort(
        max_age=30,
        n_init=3,
        nn_budget=100,
        max_cosine_distance=max_cosine_distance,
    )


def evaluate_models(params, detection_model, feature_extraction_model, dataset_path, screen, font, result_file,
                    generation, individual_id_map, next_id):
    deepsort = initialize_deepsort(params['max_cosine_distance'])
    detection_proportions = []
    reid_precisions = []
    inter_folder_precisions = []
    fitnesses = []
    processed_folders = set()

    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        print(f"Processing images and videos for {person_folder}...")
        if not os.path.isdir(person_path):
            continue
        if person_folder in processed_folders:
            continue
        processed_folders.add(person_folder)

        person_detections = []
        for filename in os.listdir(person_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(person_path, filename)
                _, detections = process_image(
                    image_path, deepsort, detection_model, feature_extraction_model,
                    params['confidence_threshold'], params['scale_factor'],
                    params['brightness'], params['contrast'], screen, font
                )
                person_detections.extend(detections)
            elif filename.endswith(".mp4") or filename.endswith(".avi"):
                video_path = os.path.join(person_path, filename)
                _, detections = process_video(
                    video_path, deepsort, detection_model, feature_extraction_model,
                    params['confidence_threshold'], params['scale_factor'],
                    params['brightness'], params['contrast'], screen, font
                )
                person_detections.extend(detections)

        detection_proportion, reid_precision, inter_folder_precision = simplified_evaluate_metrics(person_detections,
                                                                                                   processed_folders)
        detection_proportions.append(detection_proportion)
        reid_precisions.append(reid_precision)
        inter_folder_precisions.append(inter_folder_precision)
        fitness = 0.6 * detection_proportion + 0.3 * reid_precision + 0.1 * inter_folder_precision
        fitnesses.append(fitness)

        # Save results to a single file
        with open(result_file, 'a') as rf:
            individual_str = individual_to_string(params)
            if individual_str not in individual_id_map:
                individual_id_map[individual_str] = next_id
                next_id += 1
            individual_id = individual_id_map[individual_str]
            is_best_population = 'BP' if fitness == max(fitnesses) else ''
            is_best_global = 'BG' if fitness == max(fitnesses) else ''
            best_marker = is_best_population if is_best_population else is_best_global
            rf.write(
                f"{person_folder},{individual_id},{generation},{fitness},{detection_proportion},{reid_precision},{params['confidence_threshold']},{params['max_cosine_distance']},{params['scale_factor']},{params['brightness']},{params['contrast']}\n")

    average_detection_proportion = np.mean(detection_proportions)
    average_reid_precision = np.mean(reid_precisions)
    average_inter_folder_precision = np.mean(inter_folder_precisions)

    return {
        "detection_proportion": average_detection_proportion,
        "reid_precision": average_reid_precision,
        "inter_folder_precision": average_inter_folder_precision
    }


if __name__ == "__main__":
    dataset_path = "dataset"

    initial_params = {
        'confidence_threshold': 0.5,
        'max_cosine_distance': 0.2,
        'scale_factor': 1.0,
        'brightness': 1.0,
        'contrast': 1.0,
    }

    screen, font = initialize_pygame()

    result_file = "detailed_results.csv"
    individual_id_map = {}
    next_id = 1
    with open(result_file, 'w') as rf:
        rf.write(
            "folder,id,generation,fitness,detection_proportion,reid_precision,confidence_threshold,max_cosine_distance,scale_factor,brightness,contrast\n")


    def evaluate_models_lambda(params, generation, individual_id_map, next_id):
        return evaluate_models(params, model, base_model, dataset_path, screen, font, result_file, generation,
                               individual_id_map, next_id)


    generations = 2
    best_params, best_fitness = genetic_algorithm(
        pop_size=5,
        generations=generations,
        num_select=2,
        evaluate_models=evaluate_models_lambda,
        initial_params=initial_params,
        screen=screen,
        font=font,
        progress_file='genetic_progress.csv'
    )

    print(f"Best parameters found: {best_params}", f"Best fitness: {best_fitness}")
    draw_frame(screen, font, None, generations, best_params, best_fitness)

    while True:
        update_pygame_event_loop()
