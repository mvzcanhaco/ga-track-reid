import cv2
import numpy as np
from deep_sort_realtime.deep_sort.track import Track
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def preprocess_image(frame, brightness, contrast):
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    return frame


def process_frame(frame, detection_model, feature_extraction_model, confidence_threshold, deepsort, image_resize_dim,
                  brightness, contrast):
    frame_resized = cv2.resize(frame, image_resize_dim)
    detections = detect_people(frame_resized, detection_model, confidence_threshold)
    tracks = deepsort.update_tracks(detections, frame=frame_resized)  # Update DeepSort with detections
    return frame_resized, tracks


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


def extract_features(frame, bbox, feature_extraction_model):
    x1, y1, x2, y2 = bbox
    crop = frame[int(y1):int(y2), int(x1):int(x2)]
    img = cv2.resize(crop, (224, 224))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    features = feature_extraction_model.predict(img)
    return features


def calculate_reid_metrics(detections, ground_truth_ids):
    all_labels = []
    all_preds = []
    for track_list in detections:
        if isinstance(track_list, list):
            for track in track_list:
                all_labels.append(track.track_id in ground_truth_ids)
                all_preds.append(track.confidence if hasattr(track, 'confidence') else 1.0)  # Use a default confidence of 1.0 if not available
        else:
            all_labels.append(track_list.track_id in ground_truth_ids)
            all_preds.append(track_list.confidence if hasattr(track_list, 'confidence') else 1.0)

    # Calculate average precision
    average_precision = average_precision_score(all_labels, all_preds)

    # For simplicity, consider the CMC Rank-1 as the ratio of correct IDs at the first position
    cmc_rank1 = sum(all_labels) / len(all_labels) if all_labels else 0

    return average_precision, cmc_rank1


def calculate_reid_metrics(detections, ground_truth_ids):
    all_labels = []
    all_preds = []
    for track_list in detections:
        if isinstance(track_list, list):
            for track in track_list:
                all_labels.append(track.track_id in ground_truth_ids)
                all_preds.append(track.confidence if hasattr(track,
                                                             'confidence') else 1.0)  # Use a default confidence of 1.0 if not available
        else:
            all_labels.append(track_list.track_id in ground_truth_ids)
            all_preds.append(track_list.confidence if hasattr(track_list, 'confidence') else 1.0)

    # Ensure all_labels and all_preds are not empty
    if not all_labels or not all_preds:
        return 0.0, 0.0

    # Calculate average precision
    average_precision = average_precision_score(all_labels, all_preds)

    # For simplicity, consider the CMC Rank-1 as the ratio of correct IDs at the first position
    cmc_rank1 = sum(all_labels) / len(all_labels) if all_labels else 0

    return average_precision, cmc_rank1


def evaluate_metrics(detections, ground_truth_ids):
    """
    Avalia as métricas de detecção e reidentificação.
    """
    total_frames = len(detections)
    frames_with_detections = sum([1 for d in detections if isinstance(d, list) and len(d) > 0 or isinstance(d, Track)])
    detection_accuracy = frames_with_detections / total_frames if total_frames > 0 else 0

    unique_ids = set()
    for track_list in detections:
        if isinstance(track_list, list):
            for track in track_list:
                unique_ids.add(track.track_id)
        else:
            unique_ids.add(track_list.track_id)

    reid_accuracy = len(unique_ids) / total_frames if total_frames > 0 else 0

    # Calculate precision, recall, and F1-score assuming ground truth has exactly one person per frame
    y_true = [1] * total_frames
    y_pred = [1 if isinstance(d, list) and len(d) > 0 or isinstance(d, Track) else 0 for d in detections]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    # Calculate mAP and CMC Rank-1
    mAP, cmc_rank1 = calculate_reid_metrics(detections, ground_truth_ids)

    return {
        "detection_accuracy": detection_accuracy,
        "reid_accuracy": reid_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mAP": mAP,
        "cmc_rank1": cmc_rank1
    }
