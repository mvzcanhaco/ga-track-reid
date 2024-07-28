import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score
from tensorflow.keras.preprocessing import image as keras_image
from visualization import draw_frame, draw_bounding_boxes


def preprocess_image(frame, brightness, contrast):
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    return frame


def process_frame(frame, detection_model, feature_extraction_model, confidence_threshold, deepsort, scale_factor,
                  brightness, contrast):
    frame = preprocess_image(frame, brightness, contrast)
    height, width = frame.shape[:2]
    new_dim = (int(width * scale_factor), int(height * scale_factor))
    frame_resized = cv2.resize(frame, new_dim)
    print("Processing frame...")
    detections = detect_people(frame_resized, detection_model, confidence_threshold)
    if len(detections) == 0:
        print("No detections in frame.")
        return frame, []

    bboxes = [det[0] for det in detections]
    confidences = [det[1] for det in detections]
    features = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        person_image = frame[y1:y2, x1:x2]
        if person_image.size == 0:
            print("Empty crop detected, skipping this bbox.")
            continue
        feature = extract_features(person_image, bbox, feature_extraction_model)
        print(f"Extracted feature shape: {feature.shape}")
        features.append(feature)

    features = np.array(features)
    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1)
    print(f"Features reshaped for DeepSORT: {features.shape}")

    formatted_detections = [[bbox, conf, 'person'] for bbox, conf in zip(bboxes, confidences)]
    tracks = deepsort.update_tracks(raw_detections=formatted_detections, embeds=features, frame=frame_resized)
    print("Frame processing complete.")

    for track in tracks:
        print(f"Track ID: {track.track_id}, BBox: {track.to_tlbr()}")

    return frame_resized, tracks


def process_image(image_path, deepsort, detection_model, feature_extraction_model, confidence_threshold,
                  scale_factor, brightness, contrast, screen, font):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Unable to read the image from {image_path}. Please check the file path and integrity.")
        return None, []

    frame, detections = process_frame(frame, detection_model, feature_extraction_model, confidence_threshold, deepsort,
                                      scale_factor, brightness, contrast)
    draw_bounding_boxes(frame, detections)
    draw_frame(screen, font, frame)
    return frame, detections


def process_video(video_path, deepsort, detection_model, feature_extraction_model, confidence_threshold,
                  scale_factor, brightness, contrast, screen, font):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)
    frame_count = 0
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame, frame_detections = process_frame(frame, detection_model, feature_extraction_model,
                                                    confidence_threshold, deepsort, scale_factor, brightness, contrast)
            draw_bounding_boxes(frame, frame_detections)
            detections.append(frame_detections)
            draw_frame(screen, font, frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return video_path, detections


def detect_people(frame, model, confidence_threshold):
    results = model(frame)
    detections = []
    for result in results[0].boxes:
        if result.conf > confidence_threshold:  # Use the confidence score from the result
            bbox = result.xyxy[0].cpu().numpy()  # Bounding box coordinates
            score = result.conf.item()  # Confidence score
            class_id = int(result.cls.item())  # Class ID
            detections.append((bbox, score, class_id))
    return detections


def extract_features(image, bbox, feature_extraction_model):
    crop = image
    if crop.size == 0:
        print("Error: The crop is empty.")
        return np.zeros((1, 1280))
    img = cv2.resize(crop, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    features = feature_extraction_model.predict(img)
    return features


def simplified_evaluate_metrics(detections, processed_folders):
    total_frames = len(detections)
    frames_with_detections = sum(1 for d in detections if isinstance(d, list) and len(d) > 0)
    detection_proportion = frames_with_detections / total_frames if total_frames > 0 else 0

    unique_ids = set()
    all_ids_dict = {}

    for track_list in detections:
        if isinstance(track_list, list):
            for track in track_list:
                unique_ids.add(track.track_id)
                current_folder = tuple(processed_folders)
                if current_folder not in all_ids_dict:
                    all_ids_dict[current_folder] = set()
                all_ids_dict[current_folder].add(track.track_id)
        else:
            unique_ids.add(track_list.track_id)
            current_folder = tuple(processed_folders)
            if current_folder not in all_ids_dict:
                all_ids_dict[current_folder] = set()
            all_ids_dict[current_folder].add(track_list.track_id)

    reid_precision = 1 / len(unique_ids) if unique_ids else 0

    inter_folder_precision = 0
    all_ids = set()
    for folder_ids in all_ids_dict.values():
        if len(all_ids.intersection(folder_ids)) > 0:
            inter_folder_precision += 1
        all_ids.update(folder_ids)
    inter_folder_precision = 1 - inter_folder_precision / len(all_ids_dict) if all_ids_dict else 0

    return detection_proportion, reid_precision, inter_folder_precision
