import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score
from tensorflow.keras.preprocessing import image as keras_image
from visualization import draw_frame, draw_bounding_boxes


def preprocess_image(frame, brightness, contrast):
    # Ajuste de brilho e contraste
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    return frame


def process_frame(frame, detection_model, feature_extraction_model, confidence_threshold, deepsort, scale_factor,
                  brightness, contrast):
    frame = preprocess_image(frame, brightness, contrast)
    height, width = frame.shape[:2]
    new_dim = (int(width * scale_factor), int(height * scale_factor))
    frame_resized = cv2.resize(frame, new_dim)
    detections = detect_people(frame_resized, detection_model, confidence_threshold)
    if len(detections) == 0:
        return frame, []

    bboxes = [det[0] for det in detections]
    confidences = [det[1] for det in detections]
    features = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)  # Ensure the coordinates are integers
        # Ensure bounding box is within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        person_image = frame[y1:y2, x1:x2]

        # Handle empty crop
        if person_image.size == 0:
            print("Empty crop detected, skipping this bbox.")
            continue

        feature = extract_features(person_image, bbox, feature_extraction_model)
        features.append(feature)

    # Ensure features are in the correct shape for DeepSORT
    features = np.array(features)
    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1)

    formatted_detections = [[bbox, conf, 'person'] for bbox, conf in zip(bboxes, confidences)]
    tracks = deepsort.update_tracks(raw_detections=formatted_detections, embeds=features, frame=frame)

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
    for result in results.xyxy[0].cpu().numpy():
        if result[4] > confidence_threshold:  # Confidence score
            bbox = result[:4]  # Bounding box coordinates
            score = result[4]  # Confidence score
            class_id = int(result[5])  # Class ID
            detections.append((bbox, score, class_id))
    return detections


def extract_features(image, bbox, feature_extraction_model):
    # Preprocess the image for feature extraction
    crop = image
    if crop.size == 0:
        print("Error: The crop is empty.")
        return np.zeros((1, 1280))  # Return a dummy feature vector
    img = cv2.resize(crop, (224, 224))
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Extract features using the model
    features = feature_extraction_model.predict(img)
    return features


def calculate_reid_metrics(detections, ground_truth_ids):
    all_labels = []
    all_preds = []
    for track_list in detections:
        if isinstance(track_list, list):
            for track in track_list:
                all_labels.append(track.track_id in ground_truth_ids)
                all_preds.append(
                    track.conf if hasattr(track, 'conf') else 1.0)  # Use a default confidence of 1.0 if not available
        else:
            all_labels.append(track_list.track_id in ground_truth_ids)
            all_preds.append(track_list.conf if hasattr(track_list, 'conf') else 1.0)

    # Ensure all_labels and all_preds are not empty
    if not all_labels or not all_preds:
        return 0.0, 0.0

    # Calculate average precision
    average_precision = average_precision_score(all_labels, all_preds)

    # For simplicity, consider the CMC Rank-1 as the ratio of correct IDs at the first position
    cmc_rank1 = sum(all_labels) / len(all_labels) if all_labels else 0

    return average_precision, cmc_rank1


def simplified_evaluate_metrics(detections):
    """
    Avalia métricas simplificadas.
    """
    total_frames = len(detections)  # Número total de frames processados (imagens e frames de vídeo)
    frames_with_detections = sum(
        1 for d in detections if isinstance(d, list) and len(d) > 0)  # Frames com pelo menos uma detecção
    detection_proportion = frames_with_detections / total_frames if total_frames > 0 else 0

    unique_ids = set()
    for track_list in detections:
        if isinstance(track_list, list):
            for track in track_list:
                unique_ids.add(track.track_id)
        else:
            unique_ids.add(track_list.track_id)

    reid_precision = 1 / len(unique_ids) if unique_ids else 0

    return detection_proportion, reid_precision
