import numpy as np
import cv2
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc, confusion_matrix
def compute_threshold(anomaly_scores, ground_truth, method='f1'):
    if method == 'f1':
        precision, recall, thresholds = precision_recall_curve(ground_truth.flatten(),
                                                               anomaly_scores.flatten())
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]

    elif method == 'otsu':
        anomaly_scores_normalized = ((anomaly_scores - anomaly_scores.min()) /
                                     (anomaly_scores.max() - anomaly_scores.min() + 1e-10) * 255).astype(np.uint8)
        threshold, _ = cv2.threshold(anomaly_scores_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold / 255.0 * (anomaly_scores.max() - anomaly_scores.min()) + anomaly_scores.min()

    else:
        raise ValueError(f"Unknown threshold method: {method}")


def evaluate_anomaly_detection(anomaly_scores, ground_truth, threshold=None):
    anomaly_scores_flat = anomaly_scores.flatten()
    ground_truth_flat = ground_truth.flatten()
    if threshold is None:
        threshold = compute_threshold(anomaly_scores, ground_truth)

    predictions = (anomaly_scores_flat > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ground_truth_flat, predictions).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    fpr, tpr, _ = roc_curve(ground_truth_flat, anomaly_scores_flat)
    roc_auc = auc(fpr, tpr)

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    }


def evaluate_segmentation(pred_masks, true_masks, threshold=0.5):
    results = []

    for i in range(len(pred_masks)):
        pred_mask = (pred_masks[i] > threshold).astype(np.uint8)
        true_mask = true_masks[i].astype(np.uint8)
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        iou = intersection / (union + 1e-10)
        dice = 2 * intersection / (pred_mask.sum() + true_mask.sum() + 1e-10)
        correct_pixels = (pred_mask == true_mask).sum()
        total_pixels = pred_mask.size
        pixel_accuracy = correct_pixels / total_pixels

        results.append({
            'iou': iou,
            'dice': dice,
            'pixel_accuracy': pixel_accuracy
        })

    avg_results = {
        'mean_iou': np.mean([r['iou'] for r in results]),
        'mean_dice': np.mean([r['dice'] for r in results]),
        'mean_pixel_accuracy': np.mean([r['pixel_accuracy'] for r in results])
    }

    return avg_results, results