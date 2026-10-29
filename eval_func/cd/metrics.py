import numpy as np


def get_confuse_matrix(n_class, label_gts, label_preds):
    def fast_hist(label_gt, label_pred):
        mask = (label_gt >= 0) & (label_gt < n_class)
        hist = np.bincount(
            n_class * label_gt[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    confusion_matrix = np.zeros((n_class, n_class))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def cm2score(confusion_matrix):
    """Per-class Precision/Recall/F1/IoU plus OA and Kappa; rows = GT, cols = prediction."""
    cm = confusion_matrix.astype(np.float64)
    eps = np.finfo(np.float32).eps
    n_class = cm.shape[0]

    scores = {}
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    total = cm.sum()

    for c in range(n_class):
        tp = cm[c, c]
        fp = col_sums[c] - tp
        fn = row_sums[c] - tp

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)

        scores[f'class_{c}'] = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'IoU': iou,
        }

    scores['OA'] = np.trace(cm) / (total + eps)

    if total > 0:
        po = np.trace(cm) / total
        pe = (row_sums * col_sums).sum() / (total * total + eps)
        scores['Kappa'] = (po - pe) / (1 - pe + eps)
    else:
        scores['Kappa'] = 0.0

    return scores


class ConfuseMatrixMeter:
    """Accumulates predictions into a confusion matrix and reports scores from it."""

    def __init__(self, n_class=2):
        self.n_class = n_class
        self.sum = np.zeros((n_class, n_class))

    def update_cm(self, pr, gt, weight=1):
        self.sum += weight * get_confuse_matrix(self.n_class, gt, pr)

    def get_scores(self):
        return cm2score(self.sum)
