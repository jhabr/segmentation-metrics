import unittest

import numpy as np

from segmentation_metrics.metrics import Metrics


class SegmentationMetricsTests(unittest.TestCase):

    @property
    def mask(self):
        mask = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]]
        )
        mask = np.expand_dims(mask, axis=2)
        return mask

    @property
    def mask2(self):
        mask2 = np.array(
            [[0., 0., 0., 0., 0.],
             [0., 1., 1., 1., 0.],
             [0., 1., 1., 1., 0.],
             [0., 1., 1., 1., 0.],
             [0., 0., 0., 0., 0.]]
        )
        mask2 = np.expand_dims(mask2, axis=2)
        return mask2

    @property
    def predicted_mask(self):
        predicted_mask = np.array(
            [[1., 1., 1., 1., 1.],  # 5 true positives
             [1., 0., 0., 0., 0.],  # 1 true positive, 4 false negatives
             [0., 0., 0., 0., 0.],  # 5 true negatives
             [0., 0., 0., 0., 0.],  # 5 true negatives
             [1., 1., 1., 0., 0.]]  # 3 false positives, 2 true negatives
        )
        predicted_mask = np.expand_dims(predicted_mask, axis=2)
        return predicted_mask

    @property
    def predicted_mask2(self):
        predicted_mask2 = np.array(
            [[0., 0., 0., 0., 0.],  # 5 true negatives
             [0., 1., 1., 1., 0.],  # 2 true negatives, 3 true positives
             [0., 1., 1., 1., 0.],  # 2 true negatives, 3 true positives
             [0., 1., 1., 1., 0.],  # 2 true negatives, 3 true positives
             [0., 0., 0., 0., 0.]]  # 5 true negatives
        )
        predicted_mask2 = np.expand_dims(predicted_mask2, axis=2)
        return predicted_mask2

    def test_segmentation_metrics(self):
        metrics = Metrics.calculate([self.mask], [self.predicted_mask])
        self.assertEqual(metrics["n_images"], 1)
        self.assertEqual(metrics["n_true_positives"], 6)
        self.assertEqual(metrics["n_true_positives_%"], 0.6)
        self.assertEqual(metrics["n_true_negatives"], 12)
        self.assertEqual(metrics["n_true_negatives_%"], 0.8)
        self.assertEqual(metrics["n_false_positives"], 3)
        self.assertEqual(metrics["n_false_negatives"], 4)
        self.assertEqual(metrics["threshold_jaccard_index"], 0.0)
        self.assertEqual(metrics["jaccard_similarity_index (iou_score)"], 0.46153846153846156)
        self.assertEqual(metrics["dice"], 0.631578947368421)
        self.assertEqual(metrics["sensitivity"], 0.6)
        self.assertEqual(metrics["specificity"], 0.8)
        self.assertEqual(metrics["accuracy"], 0.72)

    def test_segmentation_metrics_2(self):
        metrics = Metrics.calculate([self.mask2], [self.predicted_mask2])
        self.assertEqual(metrics["n_images"], 1)
        self.assertEqual(metrics["n_true_positives"], 9)
        self.assertEqual(metrics["n_true_positives_%"], 1.0)
        self.assertEqual(metrics["n_true_negatives"], 16)
        self.assertEqual(metrics["n_true_negatives_%"], 1.0)
        self.assertEqual(metrics["n_false_positives"], 0)
        self.assertEqual(metrics["n_false_negatives"], 0)
        self.assertEqual(metrics["threshold_jaccard_index"], 1.0)
        self.assertEqual(metrics["jaccard_similarity_index (iou_score)"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_segmentation_metrics_batch(self):
        masks = [self.mask, self.mask2]
        predicted_masks = [self.predicted_mask, self.predicted_mask2]
        metrics = Metrics.calculate(masks, predicted_masks)
        self.assertEqual(metrics["n_images"], 2)
        self.assertEqual(metrics["n_true_positives"], 15)
        self.assertEqual(metrics["n_true_positives_%"], 0.7894736842105263)
        self.assertEqual(metrics["n_true_negatives"], 28)
        self.assertEqual(metrics["n_true_negatives_%"], 0.9032258064516129)
        self.assertEqual(metrics["n_false_positives"], 3)
        self.assertEqual(metrics["n_false_negatives"], 4)
        self.assertEqual(metrics["threshold_jaccard_index"], 0.5)
        self.assertEqual(metrics["jaccard_similarity_index (iou_score)"], 0.7307692307692308)
        self.assertEqual(metrics["dice"], 0.8157894736842105)
        self.assertEqual(metrics["sensitivity"], 0.8)
        self.assertEqual(metrics["specificity"], 0.9)
        self.assertEqual(metrics["accuracy"], 0.86)


if __name__ == '__main__':
    unittest.main()
