import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def normalize_tensor(tensor, eps=1e-7):
    """
    Ensures a spatial map sums to 1.0 across the spatial dimensions (H, W),
    formatting it as a strict probability distribution.
    """
    b, c, h, w = tensor.size()
    # Sum across Height and Width
    sum_tensor = torch.sum(tensor, dim=(2, 3), keepdim=True)
    return tensor / (sum_tensor + eps)


# ---------------------------------------------------------
# Individual Metric Losses
# ---------------------------------------------------------
class KLD_Loss(nn.Module):
    """
    Computes the KLD loss. Penalizes false negatives extremely heavily.
    Requires both inputs to be strictly positive probability distributions.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_prob, target_map):
        # Ensure targets are properly normalized probability distributions
        target_prob = normalize_tensor(target_map, self.eps)
        
        # KLD = sum( Q * log( Q / P ) )
        # where Q is the target distribution and P is the predicted distribution
        kld = torch.sum(
            target_prob * torch.log((target_prob + self.eps) / (pred_prob + self.eps)),
            dim=(1, 2, 3)
        )
        return kld.mean()


class CC_Loss(nn.Module):
    """
    Computes Pearson's Correlation Coefficient. 
    It normalizes its own inputs (mean subtraction and std division), 
    so it can accept raw logits directly without softmax.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target_map):
        # Flatten spatial dimensions
        b, c, h, w = pred.size()
        pred_flat = pred.view(b, -1)
        target_flat = target_map.view(b, -1)
        
        # Mean subtraction
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        # Covariance and Standard Deviation
        covar = (pred_centered * target_centered).sum(dim=1)
        pred_var = (pred_centered ** 2).sum(dim=1).sqrt()
        target_var = (target_centered ** 2).sum(dim=1).sqrt()
        
        # Calculate CC (ranges from -1 to 1)
        cc = covar / (pred_var * target_var + self.eps)
        return cc.mean()


class SIM_Loss(nn.Module):
    """
    Computes the Similarity (SIM) metric. 
    Measures the sum of minimum values between two probability distributions.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_prob, target_map):
        target_prob = normalize_tensor(target_map, self.eps)
        # SIM = sum( min(P, Q) )
        sim = torch.sum(torch.min(pred_prob, target_prob), dim=(1, 2, 3))
        return sim.mean()


# ---------------------------------------------------------
# The Composite Optimization Strategy
# ---------------------------------------------------------
class UNETRSal_Loss(nn.Module):
    """
    The master loss function. It orchestrates the mathematical tug-of-war 
    between statistical alignment and probabilistic smoothness.
    """
    def __init__(self):
        super().__init__()
        self.kld = KLD_Loss()
        self.cc = CC_Loss()
        self.sim = SIM_Loss()
        
    def forward(self, pred_logits, target_map):
        """
        Args:
            pred_logits: Raw, unnormalized outputs directly from the decoding head.
            target_map: The continuous SALICON ground truth map.
        """
        b, c, h, w = pred_logits.size()
        
        # 1. Prepare Probability Space (For KLD and SIM)
        # We apply a spatial softmax by flattening the spatial dims, applying softmax, and reshaping
        pred_probs = F.softmax(pred_logits.view(b, c, -1), dim=2).view(b, c, h, w)
        
        # 2. Calculate Individual Losses
        # Notice we pass the PROBABILITIES to KLD and SIM
        loss_kld = self.kld(pred_probs, target_map)
        loss_sim = self.sim(pred_probs, target_map)
        
        # Notice we pass the RAW LOGITS to CC
        loss_cc = self.cc(pred_logits, target_map)
        
        # 3. The Weighted Optimization Formula
        # We want to MINIMIZE KLD. We want to MAXIMIZE CC and SIM. 
        # Therefore, CC and SIM receive negative weights in the loss formula.
        total_loss = (10.0 * loss_kld) - (1.0 * loss_cc) - (1.0 * loss_sim)
        
        # Returning the individual components is highly recommended so you can track 
        # them individually in your Colab/TensorBoard logs to ensure they are balanced.
        return total_loss, loss_kld, loss_cc, loss_sim