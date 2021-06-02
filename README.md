# Segmentation Metrics

A python library to help you calculate:

- TP (true positives)
- TN (true negatives)
- FP (false positives)
- FN (false negatives)
- Jaccard Similarity Index (IoU)
- Threshold Jaccard Index
- Dice Coefficient
- Sensitivity
- Specificity
- Accuracy

for a binary segmentation task.

## Example

```python
import cv2
import segmentation_metrics as sm

mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
predicted_mask = cv2.imread("predicted_mask.png", cv2.IMREAD_GRAYSCALE)

metrics = sm.calculate(
    masks=[mask],
    predicted_masks=[predicted_mask],
    jaccard_threshold=0.65
)

print(metrics)
```

Example Output (but not ☝ that one)️:
```python
{
    'n_images': 300,
    'n_true_positives': 3086436,
    'n_true_positives_%': 0.9052690301820284,
    'n_true_negatives': 11023919,
    'n_true_negatives_%': 0.9724538771281737,
    'n_false_positives': 312268,
    'n_false_negatives': 322977,
    'threshold_jaccard_index': 0.7551666249674276,
    'jaccard_similarity_index_(iou_score)': 0.8186671808348286,
    'dice_coefficient': 0.8902348083199045,
    'sensitivity': 0.9189991951163548,
    'specificity': 0.9669434672587298,
    'accuracy': 0.9569196912977429
}
```

### Future Work
- Add support for multiclass segmentation metrics