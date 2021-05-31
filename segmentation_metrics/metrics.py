
import numpy as np

from segmentation_metrics.core import BinarySegmentationMetrics


class Metrics:

    def calculate(self, mask, predicted_mask, jaccard_similarity_index_threshold=0.65, normalize=False):
        assert mask is not None and predicted_mask is not None

        if normalize:
            mask = mask / 255.0
            predicted_mask = predicted_mask / 255.0

        metrics = BinarySegmentationMetrics(
            jaccard_similarity_index_threshold=jaccard_similarity_index_threshold
        )
        metrics.calculate(mask=mask, predicted_mask=predicted_mask)
        return {
            "n_images": 1,
            "n_true_positives": metrics.tp,
            "n_true_positives_%": metrics.tp / metrics.n_mask_pixels,
            "n_true_negatives": metrics.tn,
            "n_true_negatives_%": metrics.tn / metrics.n_background_pixels,
            "n_false_positives": metrics.fp,
            "n_false_negatives": metrics.fn,
            "iou_score": metrics.iou_score,
            "threshold_jaccard_index": metrics.threshold_jaccard_index,
            "jaccard_similarity_index": metrics.jaccard_similarity_index,
            "dice": metrics.dice,
            "f1_score": metrics.f1_score,
            "sensitivity": metrics.sensitivity,
            "specificity": metrics.specificity,
            "accuracy": metrics.accuracy
        }

    def calculate_batch(self, masks, predicted_masks, jaccard_similarity_index_threshold=0.65, normalize=False):
        assert masks is not None and predicted_masks is not None

        metrics = BinarySegmentationMetrics(
            jaccard_similarity_index_threshold=jaccard_similarity_index_threshold
        )

        mask_pixels = []
        background_pixels = []
        tp = []
        tn = []
        fp = []
        fn = []
        iou_scores = []
        threshold_jaccard_indexes = []
        jaccard_similarity_indexes = []
        dice_scores = []
        f1_scores = []
        sensitivities = []
        specificities = []
        accuracies = []

        for i in range(len(masks)):
            mask = masks[i]
            predicted_mask = predicted_masks[i]

            if normalize:
                mask = mask / 255.0
                predicted_mask = predicted_mask / 255.0

            metrics.calculate(mask=mask, predicted_mask=predicted_mask)
            mask_pixels.append(metrics.n_mask_pixels)
            background_pixels.append(metrics.n_background_pixels)
            tp.append(metrics.tp)
            tn.append(metrics.tn)
            fp.append(metrics.fp)
            fn.append(metrics.fn)
            iou_scores.append(metrics.iou_score)
            threshold_jaccard_indexes.append(metrics.threshold_jaccard_index)
            jaccard_similarity_indexes.append(metrics.jaccard_similarity_index)
            dice_scores.append(metrics.dice)
            f1_scores.append(metrics.f1_score)
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
            "iou_score": np.mean(iou_scores),
            "threshold_jaccard_index": np.mean(threshold_jaccard_indexes),
            "jaccard_similarity_index": np.mean(jaccard_similarity_indexes),
            "dice": np.mean(dice_scores),
            "f1_score": np.mean(f1_scores),
            "sensitivity": np.mean(sensitivities),
            "specificity": np.mean(specificities),
            "accuracy": np.mean(accuracies)
        }
