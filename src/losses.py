import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_tensor(tensor, eps=1e-7):
    """Normalize a spatial map to sum to 1 over (H, W) so it is a probability map."""
    b, c, h, w = tensor.size()
    sum_tensor = torch.sum(tensor, dim=(2, 3), keepdim=True)
    return tensor / (sum_tensor + eps)


class KLD_Loss(nn.Module):
    """KL divergence between target and predicted saliency distributions.

    Heavily penalizes false negatives. Both inputs must be positive and
    normalized as probability maps.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_prob, target_map):
        target_prob = normalize_tensor(target_map, self.eps)
        # KLD = sum( Q * log(Q / P) ), Q = target, P = prediction
        kld = torch.sum(
            target_prob * torch.log((target_prob + self.eps) / (pred_prob + self.eps)),
            dim=(1, 2, 3)
        )
        return kld.mean()


class CC_Loss(nn.Module):
    """Pearson correlation coefficient.

    Centers and scales its inputs internally, so it takes raw logits (no softmax).
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target_map):
        b, c, h, w = pred.size()
        pred_flat = pred.view(b, -1)
        target_flat = target_map.view(b, -1)

        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean

        covar = (pred_centered * target_centered).sum(dim=1)
        pred_var = (pred_centered ** 2).sum(dim=1).sqrt()
        target_var = (target_centered ** 2).sum(dim=1).sqrt()

        cc = covar / (pred_var * target_var + self.eps)  # in [-1, 1]
        return cc.mean()


class SIM_Loss(nn.Module):
    """Similarity metric: sum of the per-pixel minimum of two probability maps."""
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_prob, target_map):
        target_prob = normalize_tensor(target_map, self.eps)
        sim = torch.sum(torch.min(pred_prob, target_prob), dim=(1, 2, 3))
        return sim.mean()


class Composite_Loss(nn.Module):
    """Training loss: 10*KLD - CC - SIM.

    KLD/SIM run on softmax probabilities, CC on the raw logits.
    """
    def __init__(self):
        super().__init__()
        self.kld = KLD_Loss()
        self.cc = CC_Loss()
        self.sim = SIM_Loss()

    def forward(self, pred_logits, target_map):
        # pred_logits: raw decoder output; target_map: continuous SALICON map.
        b, c, h, w = pred_logits.size()

        # Spatial softmax over the flattened HxW grid for the probability-based terms.
        pred_probs = F.softmax(pred_logits.view(b, c, -1), dim=2).view(b, c, h, w)

        loss_kld = self.kld(pred_probs, target_map)
        loss_sim = self.sim(pred_probs, target_map)
        loss_cc = self.cc(pred_logits, target_map)

        # Minimize KLD, maximize CC and SIM (hence the negative weights).
        total_loss = (10.0 * loss_kld) - (1.0 * loss_cc) - (1.0 * loss_sim)

        # Return the components too, to track them separately during training.
        return total_loss, loss_kld, loss_cc, loss_sim
