import torch
import torch.nn.functional as F
from src.metrics import nss, auc_judd, information_gain


def train_one_epoch(decoder, dataloader, optimizer, criterion, device):
    """One pass over the training data. Returns average loss, KLD and CC."""
    decoder.train()

    total_loss_accum = 0.0
    kld_accum = 0.0
    cc_accum = 0.0

    for batch_idx, (features, target_map, _) in enumerate(dataloader):

        # features is a list/dict of tensors, so move each one.
        if isinstance(features, dict):
            features = [f.to(device) for f in features.values()]
        else:
            features = [f.to(device) for f in features]

        target_map = target_map.to(device)

        optimizer.zero_grad()

        raw_logits = decoder(features)

        # Upsample the prediction to the ground-truth resolution before the loss.
        target_height, target_width = target_map.shape[2], target_map.shape[3]

        matched_logits = F.interpolate(
            raw_logits,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )

        loss, loss_kld, loss_cc, loss_sim = criterion(matched_logits, target_map)

        loss.backward()
        optimizer.step()

        total_loss_accum += loss.item()
        kld_accum += loss_kld.item()
        cc_accum += loss_cc.item()

    avg_loss = total_loss_accum / len(dataloader)
    avg_kld = kld_accum / len(dataloader)
    avg_cc = cc_accum / len(dataloader)

    return avg_loss, avg_kld, avg_cc


@torch.no_grad()
def evaluate_model(decoder, dataloader, criterion, device):
    """Validation loss only (no backprop)."""
    decoder.eval()

    total_loss_accum = 0.0

    for features, target_map, fixation_map in dataloader:

        if isinstance(features, dict):
            features = [f.to(device) for f in features.values()]
        else:
            features = [f.to(device) for f in features]

        target_map = target_map.to(device)

        raw_logits = decoder(features)

        target_height, target_width = target_map.shape[2], target_map.shape[3]
        matched_logits = F.interpolate(
            raw_logits,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )

        loss, _, _, _ = criterion(matched_logits, target_map)
        total_loss_accum += loss.item()

    avg_loss = total_loss_accum / len(dataloader)
    return avg_loss


@torch.no_grad()
def test_model(decoder, dataloader, criterion, device, baseline_prob=None):
    """Full test-set pass: continuous losses plus the discrete metrics.

    baseline_prob is the center-bias prior for Information Gain; if None a
    uniform prior is used. Returns
    (avg_loss, avg_kld, avg_cc, avg_sim, avg_nss, avg_auc, avg_ig).
    """
    decoder.eval()

    total_loss, total_kld, total_cc, total_sim = 0.0, 0.0, 0.0, 0.0
    total_nss, total_auc, total_ig = 0.0, 0.0, 0.0

    for features, target_map, fixation_map in dataloader:

        if isinstance(features, dict):
            features = [f.to(device) for f in features.values()]
        else:
            features = [f.to(device) for f in features]

        target_map = target_map.to(device)
        fixation_map = fixation_map.to(device)  # needed by the discrete metrics

        raw_logits = decoder(features)

        b, c, target_height, target_width = target_map.shape
        matched_logits = F.interpolate(
            raw_logits,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )

        loss, kld, cc, sim = criterion(matched_logits, target_map)
        total_loss += loss.item()
        total_kld += kld.item()
        total_cc += cc.item()
        total_sim += sim.item()

        # Information Gain needs a normalized probability map.
        pred_prob = F.softmax(matched_logits.view(b, 1, -1), dim=2).view(b, 1, target_height, target_width)

        if baseline_prob is None:
            batch_baseline = torch.ones(b, 1, target_height, target_width).to(device) / (target_height * target_width)
        else:
            batch_baseline = baseline_prob.expand(b, -1, -1, -1).to(device)

        total_nss += nss(matched_logits, fixation_map)
        total_auc += auc_judd(matched_logits, fixation_map)
        total_ig += information_gain(pred_prob, fixation_map, batch_baseline)

    num_batches = len(dataloader)

    avg_loss = total_loss / num_batches
    avg_kld = total_kld / num_batches
    avg_cc = total_cc / num_batches
    avg_sim = total_sim / num_batches
    avg_nss = total_nss / num_batches
    avg_auc = total_auc / num_batches
    avg_ig = total_ig / num_batches

    return avg_loss, avg_kld, avg_cc, avg_sim, avg_nss, avg_auc, avg_ig
