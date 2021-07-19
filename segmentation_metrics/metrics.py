from typing import List

import numpy as np

from segmentation_metrics.core import BinarySegmentationMetrics


def calculate(masks: List[np.ndarray], predicted_masks: List[np.ndarray], jaccard_threshold: float = 0.65) -> dict:
    """
    Calculates the metrics.

    :param masks: list
        List of masks (ground truth)
    :param predicted_masks:
        List of predicted masks
    :param jaccard_threshold:
        Threshold Jaccard Index will return 0 for values below the threshold
    :return: dict
        Calculate metrics


    n_true_positives_%: percentage of true positives (out of all true positives)
    n_true_negatives_%: percentage of true negatives (out of all true negatives)
    """
    assert masks is not None and predicted_masks is not None, "Masks and predicted masks should not be None."

    metrics = BinarySegmentationMetrics(jaccard_threshold=jaccard_threshold)

    mask_pixels, background_pixels, tp, tn, fp, fn = [], [], [], [], [], []
    threshold_jaccard_indexes, jaccard_similarity_indexes, dice_scores = [], [], []
    accuracies, sensitivities, specificities = [], [], []

    for i in range(len(masks)):
        mask = masks[i]
        predicted_mask = predicted_masks[i]

        metrics.calculate(mask=mask, predicted_mask=predicted_mask)
        mask_pixels.append(metrics.n_mask_pixels)
        background_pixels.append(metrics.n_background_pixels)
        tp.append(metrics.tp)
        tn.append(metrics.tn)
        fp.append(metrics.fp)
        fn.append(metrics.fn)
        threshold_jaccard_indexes.append(metrics.threshold_jaccard_index)
        jaccard_similarity_indexes.append(metrics.jaccard_similarity_index)
        dice_scores.append(metrics.dice)
        sensitivities.append(metrics.sensitivity)
        specificities.append(metrics.specificity)
        accuracies.append(metrics.accuracy)

    return {
        "n_images": len(masks),
        "n_true_positives": sum(tp),
        "n_true_positives_%": sum(tp) / sum(mask_pixels),
        "n_true_negatives": sum(tn),
        "n_true_negatives_%": sum(tn) / sum(background_pixels),
        "n_false_positives": sum(fp),
        "n_false_negatives": sum(fn),
        "threshold_jaccard_index": np.mean(threshold_jaccard_indexes),
        "jaccard_similarity_index_(iou_score)": np.mean(jaccard_similarity_indexes),
        "dice_coefficient": np.mean(dice_scores),
        "sensitivity": np.mean(sensitivities),
        "specificity": np.mean(specificities),
        "accuracy": np.mean(accuracies)
    }
