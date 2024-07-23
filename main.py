import os
import cv2
import numpy as np
import torch
from tensorflow.keras.applications import MobileNetV2
from deep_sort_realtime.deepsort_tracker import DeepSort
from genetic_algorithm import genetic_algorithm
from processing_functions import process_frame, preprocess_image, evaluate_metrics
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


def draw_bounding_boxes(frame, detections):
    for detection in detections:
        bbox = detection.to_tlbr()  # Adjusted to use the Track object method
        track_id = detection.track_id  # Adjusted to use the Track object attribute
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)


def evaluate_models(params, detection_model, feature_extraction_model, dataset_path, screen, font):
    deepsort = initialize_deepsort(params['max_cosine_distance'])
    metrics_list = []
    processed_folders = set()
    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        print(f"Processing images and videos for {person_folder}...")
        if not os.path.isdir(person_path):
            continue  # Skip non-directory files
        if person_folder in processed_folders:
            continue  # Skip already processed folders
        processed_folders.add(person_folder)

        person_detections = []
        ground_truth_ids = [person_folder] * len(os.listdir(person_path))  # All frames have the same ground truth ID
        for filename in os.listdir(person_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(person_path, filename)
                _, detections = process_image(
                    image_path, deepsort, detection_model, feature_extraction_model,
                    params['confidence_threshold'], (params['image_resize_dim'], params['image_resize_dim']),
                    params['brightness'], params['contrast'], screen, font
                )
                person_detections.extend(detections)
            elif filename.endswith(".mp4") or filename.endswith(".avi"):
                video_path = os.path.join(person_path, filename)
                _, detections = process_video(
                    video_path, deepsort, detection_model, feature_extraction_model,
                    params['confidence_threshold'], (params['image_resize_dim'], params['image_resize_dim']),
                    params['brightness'], params['contrast'], screen, font
                )
                person_detections.extend(detections)
        metrics = evaluate_metrics(person_detections, ground_truth_ids)
        print(f"Metrics for {person_folder}: {metrics}")
        metrics_list.append(metrics)

    # Calculate the average of each metric
    average_metrics = {
        "detection_accuracy": np.mean([m["detection_accuracy"] for m in metrics_list]),
        "reid_accuracy": np.mean([m["reid_accuracy"] for m in metrics_list]),
        "precision": np.mean([m["precision"] for m in metrics_list]),
        "recall": np.mean([m["recall"] for m in metrics_list]),
        "f1_score": np.mean([m["f1_score"] for m in metrics_list]),
        "mAP": np.mean([m["mAP"] for m in metrics_list]),
        "cmc_rank1": np.mean([m["cmc_rank1"] for m in metrics_list]),
    }

    print(f"Average metrics: {average_metrics}")
    return average_metrics


def process_image(image_path, deepsort, detection_model, feature_extraction_model, confidence_threshold,
                  image_resize_dim, brightness, contrast, screen, font):
    frame = cv2.imread(image_path)
    print(f"Processing image: {image_path}")
    if frame is None:
        print(f"Error: Unable to read the image from {image_path}. Please check the file path and integrity.")
        return None, []

    frame = preprocess_image(frame, brightness, contrast)
    frame, detections = process_frame(frame, detection_model, feature_extraction_model, confidence_threshold, deepsort,
                                      image_resize_dim, brightness, contrast)
    draw_bounding_boxes(frame, detections)  # Draw bounding boxes
    draw_frame(screen, font, frame)  # Visualize the frame
    return frame, detections


def process_video(video_path, deepsort, detection_model, feature_extraction_model, confidence_threshold,
                  image_resize_dim, brightness, contrast, screen, font):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Process 1 frame per second
    frame_count = 0
    detections = []
    print(f"Processing video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:  # Process this frame
            frame = preprocess_image(frame, brightness, contrast)
            frame, frame_detections = process_frame(frame, detection_model, feature_extraction_model,
                                                    confidence_threshold, deepsort, image_resize_dim, brightness,
                                                    contrast)
            draw_bounding_boxes(frame, frame_detections)  # Draw bounding boxes
            detections.extend(frame_detections)
            draw_frame(screen, font, frame)  # Visualize the frame
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return video_path, detections


# Main function to run genetic algorithm and process dataset
if __name__ == "__main__":
    dataset_path = "dataset"  # Update this path to your dataset

    # Initial parameters
    initial_params = {
        'confidence_threshold': 0.5,
        'max_cosine_distance': 0.2,
        'image_resize_dim': 224,
        'brightness': 1.0,
        'contrast': 1.0,
    }

    screen, font = initialize_pygame()  # Initialize Pygame screen and font


    def evaluate_models_lambda(params):
        return evaluate_models(params, model, base_model, dataset_path, screen, font)


    generations = 5  # Define the number of generations
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

    print(f"Best parameters found: {best_params}")
    draw_frame(screen, font, None, generations, best_params, best_fitness)  # Display final best parameters and fitness
