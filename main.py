import os
import cv2
import numpy as np
import torch
from tensorflow.keras.applications import MobileNetV2
from deep_sort_realtime.deepsort_tracker import DeepSort
from genetic_algorithm import genetic_algorithm
from processing_functions import simplified_evaluate_metrics, process_image, process_video
from visualization import initialize_pygame, draw_frame

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("YOLOv5 model loaded successfully.")

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


def evaluate_models(params, detection_model, feature_extraction_model, dataset_path, screen, font):
    deepsort = initialize_deepsort(params['max_cosine_distance'])
    detection_proportions = []
    reid_precisions = []
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

        detection_proportion, reid_precision = simplified_evaluate_metrics(person_detections)
        detection_proportions.append(detection_proportion)
        reid_precisions.append(reid_precision)
        print(
            f"Metrics for {person_folder}: Detection Proportion: {detection_proportion}, Re-ID Precision: {reid_precision}")

    average_detection_proportion = np.mean(detection_proportions)
    average_reid_precision = np.mean(reid_precisions)

    return {
        "detection_proportion": average_detection_proportion,
        "reid_precision": average_reid_precision
    }


# Main function to run genetic algorithm and process dataset
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


    def evaluate_models_lambda(params):
        metrics = evaluate_models(params, model, base_model, dataset_path, screen, font)
        # Return weighted average of detection proportion and re-id precision
        return 0.7 * metrics['detection_proportion'] + 0.3 * metrics['reid_precision']


    generations = 5
    best_params, best_fitness = genetic_algorithm(
        pop_size=10,
        generations=generations,
        num_select=4,
        evaluate_models=evaluate_models_lambda,
        initial_params=initial_params,
        screen=screen,
        font=font,
        progress_file='genetic_progress.csv'
    )

    print(f"Best parameters found: {best_params}", f"Best fitness: {best_fitness}")
    draw_frame(screen, font, None, generations, best_params, best_fitness)
