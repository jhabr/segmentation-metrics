import numpy as np


class BinarySegmentationMetrics:
    """
    This class is responsible for calculating simple metrics for one pair of ground truth mask and its predicted mask.

    :param jaccard_threshold: float
        Threshold value for the jaccard index. Values below this value will be calculated as 0.

    TP (true positives): pixels correctly segmented as foreground
    TN (true negatives): pixels correctly detected as background
    FP (false positives): pixels falsely segmented as foreground
    FN (false negatives): pixels falsely detected as background
    """

    def __init__(self, jaccard_threshold: float = 0.0):
        self.n_mask_pixels = 0
        self.n_background_pixels = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.jaccard_threshold = jaccard_threshold

    def calculate(self, mask: np.ndarray, predicted_mask: np.ndarray) -> 'BinarySegmentationMetrics':
        """
        Calculate pixel-wise tp, tn, fp and fn.

        :param mask: np.ndarray
            The ground truth mask.
        :param predicted_mask: np.ndarray
            The predicted mask.
        :return: BinarySegmentationMetrics
            Update instance of BinarySegmentationMetrics
        """
        assert mask is not None and predicted_mask is not None, "Mask and predicted mask shall not be None."

        self.__calculate_positives_negatives(mask, predicted_mask)
        return self

    def __calculate_positives_negatives(self, mask: np.ndarray, predicted_mask: np.ndarray):
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

        tp, tn, fp, fn = 0, 0, 0, 0

        for i in range(height):
            for j in range(width):
                mask_pixel_value = mask[i][j]
                predicted_mask_pixel_value = predicted_mask[i][j]
                if mask_pixel_value == predicted_mask_pixel_value:
                    if mask_pixel_value == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if predicted_mask_pixel_value == 0:
                        fn += 1
                    else:
                        fp += 1

        assert tp + tn + fp + fn == height * width, "Sum of all pixels is not equal to the resolutions of the image."

        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    @property
    def jaccard_similarity_index(self) -> float:
        denominator = (self.tp + self.fp + self.fn)
        if denominator == 0:
            return 0
        return self.tp / denominator

    @property
    def threshold_jaccard_index(self) -> float:
        if self.jaccard_similarity_index >= self.jaccard_threshold:
            return self.jaccard_similarity_index
        else:
            return 0.0

    @property
    def dice(self) -> float:
        denominator = (2 * self.tp + self.fn + self.fp)
        if denominator == 0:
            return 0
        return (2 * self.tp) / denominator

    @property
    def sensitivity(self) -> float:
        denominator = (self.tp + self.fn)
        if denominator == 0:
            return 0
        return self.tp / denominator

    @property
    def specificity(self) -> float:
        denominator = (self.tn + self.fp)
        if denominator == 0:
            return 0
        return self.tn / denominator

    @property
    def accuracy(self) -> float:
        denominator = (self.tp + self.fp + self.tn + self.fn)
        if denominator == 0:
            return 0
        return (self.tp + self.tn) / denominator
