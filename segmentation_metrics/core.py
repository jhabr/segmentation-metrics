import numpy as np


class BinarySegmentationMetrics:

    def __init__(self, jaccard_similarity_index_threshold=0.0):
        self.n_mask_pixels = 0
        self.n_background_pixels = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.iou_score = 0
        self.jaccard_similarity_index_threshold = jaccard_similarity_index_threshold

    def calculate(self, mask, predicted_mask):
        self.__calculate_simple_metrics(mask, predicted_mask)
        self.__calculate_iou(mask, predicted_mask)

    def __calculate_simple_metrics(self, mask, predicted_mask):
        assert mask.shape == predicted_mask.shape
        assert len(mask.shape) == len(predicted_mask.shape) == 3
        # assert binary mask
        assert mask.shape[-1] == 1 and predicted_mask.shape[-1] == 1
        # reshape to only 2 dimensions
        mask = mask.squeeze()
        predicted_mask = predicted_mask.squeeze()

        self.n_mask_pixels = np.count_nonzero(mask == 1.0)
        self.n_background_pixels = np.count_nonzero(mask == 0.0)

        height, width = mask.shape

        true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

        for i in range(height):
            for j in range(width):
                mask_pixel_value = mask[i][j]
                predicted_mask_pixel_value = predicted_mask[i][j]
                if mask_pixel_value == predicted_mask_pixel_value:
                    if mask_pixel_value == 1:
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if predicted_mask_pixel_value == 0:
                        false_negatives += 1
                    else:
                        false_positives += 1

        assert true_positives + \
               true_negatives + \
               false_positives + \
               false_negatives == height * width

        self.true_positives = true_positives
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.false_negatives = false_negatives

    @property
    def tp(self):
        """TP: pixels correctly segmented as foreground"""
        return self.true_positives

    @property
    def tn(self):
        """TN: pixels correctly detected as background"""
        return self.true_negatives

    @property
    def fp(self):
        """FP: pixels falsely segmented as foreground"""
        return self.false_positives

    @property
    def fn(self):
        """FN: pixels falsely detected as background"""
        return self.false_negatives

    def __calculate_iou(self, mask, predicted_mask):
        intersection = np.logical_and(mask, predicted_mask)
        union = np.logical_or(mask, predicted_mask)
        self.iou_score = np.sum(intersection) / np.sum(union)

    @property
    def jaccard_similarity_index(self):
        denominator = (self.tp + self.fn + self.fp)
        if denominator == 0:
            return 0
        return self.tp / denominator

    @property
    def threshold_jaccard_index(self):
        """
        Based on https://clusteval.sdu.dk/1/clustering_quality_measures/7
        """
        if self.jaccard_similarity_index >= self.jaccard_similarity_index_threshold:
            return self.jaccard_similarity_index
        else:
            return 0.0

    @property
    def dice(self):
        denominator = (2 * self.tp + self.fn + self.fp)
        if denominator == 0:
            return 0
        return (2 * self.tp) / denominator

    @property
    def f1_score(self):
        denominator = (2 * self.tp + self.fn + self.fp)
        if denominator == 0:
            return 0
        return self.tp / denominator

    @property
    def sensitivity(self):
        denominator = (self.tp + self.fn)
        if denominator == 0:
            return 0
        return self.tp / denominator

    @property
    def specificity(self):
        denominator = (self.tn + self.fp)
        if denominator == 0:
            return 0
        return self.tn / denominator

    @property
    def accuracy(self):
        denominator = (self.tp + self.fp + self.tn + self.fn)
        if denominator == 0:
            return 0
        return (self.tp + self.tn) / denominator
