import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def nss(pred_map, fixation_map, eps=1e-7):
    """Normalized Scanpath Saliency: mean of the z-scored prediction at fixations.

    pred_map and fixation_map are [B, 1, H, W]; fixation_map is binary.
    """
    mean = pred_map.mean(dim=(2, 3), keepdim=True)
    std = pred_map.std(dim=(2, 3), keepdim=True)
    pred_normalized = (pred_map - mean) / (std + eps)

    fixation_values = pred_normalized * fixation_map
    nss_scores = fixation_values.sum(dim=(2, 3)) / (fixation_map.sum(dim=(2, 3)) + eps)

    return nss_scores.mean().item()

def auc_judd(pred_map, fixation_map):
    """AUC-Judd: saliency as fixated-vs-not classification, averaged over the batch."""
    b, c, h, w = pred_map.size()
    auc_scores = []

    for i in range(b):
        y_true = fixation_map[i].view(-1).cpu().numpy()
        y_score = pred_map[i].view(-1).detach().cpu().numpy()

        # roc_auc_score needs both classes present.
        if y_true.sum() > 0 and y_true.sum() < (h * w):
            auc = roc_auc_score(y_true, y_score)
            auc_scores.append(auc)

    if not auc_scores:
        return 0.0

    return sum(auc_scores) / len(auc_scores)

def information_gain(pred_prob, fixation_map, baseline_prob, eps=1e-7):
    """Information Gain in bits over a center-bias baseline, measured at fixations.

    All inputs are [B, 1, H, W]; pred_prob and baseline_prob are probability maps.
    """
    # IG = mean over fixations of log2(P) - log2(baseline).
    log_pred = torch.log2(pred_prob + eps) * fixation_map
    log_base = torch.log2(baseline_prob + eps) * fixation_map

    ig_scores = (log_pred - log_base).sum(dim=(2, 3)) / (fixation_map.sum(dim=(2, 3)) + eps)

    return ig_scores.mean().item()
