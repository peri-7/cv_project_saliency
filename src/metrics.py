import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def nss(pred_map, fixation_map, eps=1e-7):
    """
    Normalized Scanpath Saliency (NSS).
    Measures the average normalized saliency value at human fixation locations.
    
    Args:
        pred_map (torch.Tensor): The model's raw spatial output [B, 1, H, W].
        fixation_map (torch.Tensor): Binary ground truth fixations [B, 1, H, W].
    Returns:
        float: The average NSS score across the batch.
    """
    # 1. Z-score normalize the predicted map to have zero mean and unit variance
    mean = pred_map.mean(dim=(2, 3), keepdim=True)
    std = pred_map.std(dim=(2, 3), keepdim=True)
    pred_normalized = (pred_map - mean) / (std + eps)
    
    # 2. Extract values at fixation points and compute the average
    # We multiply the normalized map by the binary fixation map (0s and 1s)
    fixation_values = pred_normalized * fixation_map
    nss_scores = fixation_values.sum(dim=(2, 3)) / (fixation_map.sum(dim=(2, 3)) + eps)
    
    return nss_scores.mean().item()

def auc_judd(pred_map, fixation_map):
    """
    Area Under the ROC Curve (AUC - Judd implementation).
    Treats saliency prediction as a binary classification task (fixated vs. non-fixated).
    
    Args:
        pred_map (torch.Tensor): The model's spatial output [B, 1, H, W].
        fixation_map (torch.Tensor): Binary ground truth fixations [B, 1, H, W].
    Returns:
        float: The average AUC score across the batch.
    """
    b, c, h, w = pred_map.size()
    auc_scores = []
    
    for i in range(b):
        # Flatten the tensors for scikit-learn
        y_true = fixation_map[i].view(-1).cpu().numpy()
        y_score = pred_map[i].view(-1).detach().cpu().numpy()
        
        # Ensure there are both positive and negative samples
        if y_true.sum() > 0 and y_true.sum() < (h * w):
            auc = roc_auc_score(y_true, y_score)
            auc_scores.append(auc)
            
    if not auc_scores:
        return 0.0
        
    return sum(auc_scores) / len(auc_scores)

def information_gain(pred_prob, fixation_map, baseline_prob, eps=1e-7):
    """
    Information Gain (IG).
    Measures how much better the model is than a center-bias baseline, reported in "bits".
    
    Args:
        pred_prob (torch.Tensor): The model's probability distribution [B, 1, H, W].
        fixation_map (torch.Tensor): Binary ground truth fixations [B, 1, H, W].
        baseline_prob (torch.Tensor): A center-bias probability map [B, 1, H, W].
    Returns:
        float: The IG score in bits.
    """
    # IG = mean( log2( P(fix) ) - log2( B(fix) ) )
    # Extract the log probabilities at the exact fixation pixels
    log_pred = torch.log2(pred_prob + eps) * fixation_map
    log_base = torch.log2(baseline_prob + eps) * fixation_map
    
    # Calculate the mean information gain over all fixations
    ig_scores = (log_pred - log_base).sum(dim=(2, 3)) / (fixation_map.sum(dim=(2, 3)) + eps)
    
    return ig_scores.mean().item()