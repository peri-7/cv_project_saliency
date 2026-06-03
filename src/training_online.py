import torch
import torch.nn.functional as F
from src.metrics import nss, auc_judd, information_gain



def train_one_epoch_online(extractor, decoder, dataloader, optimizer, criterion, device):
    """
    Executes an end-to-end pass. 
    Extractor is frozen (no gradients). Decoder is actively trained.
    """
    extractor.eval() # Freeze BatchNorm/Dropout in backbone
    decoder.train()  # Enable training mode for decoder
    
    total_loss_accum, kld_accum, cc_accum = 0.0, 0.0, 0.0
    
    for images, target_map, _ in dataloader:
        
        images = images.to(device)
        target_map = target_map.to(device)
        
        # 1. ONLINE EXTRACTION (Strictly No Gradients)
        with torch.no_grad():
            features_dict = extractor(images)
            # Convert dictionary of batch tensors to list of batch tensors
            features = list(features_dict.values())
        
        # 2. DECODER OPTIMIZATION
        optimizer.zero_grad()
        raw_logits = decoder(features)
        
        matched_logits = F.interpolate(
            raw_logits, 
            size=(target_map.shape[2], target_map.shape[3]), 
            mode='bilinear', align_corners=False
        )
        
        loss, loss_kld, loss_cc, _ = criterion(matched_logits, target_map)
        
        loss.backward()
        optimizer.step()
        
        total_loss_accum += loss.item()
        kld_accum += loss_kld.item()
        cc_accum += loss_cc.item()
        
    L = len(dataloader)
    avg_loss = total_loss_accum / L
    avg_kld = kld_accum / L
    avg_cc = cc_accum / L
    
    return avg_loss, avg_kld, avg_cc




@torch.no_grad()
def evaluate_model_online(extractor, decoder, dataloader, criterion, device):
    """
    Evaluates the model on the validation set end-to-end.
    Both extractor and decoder are strictly frozen.
    """
    extractor.eval()
    decoder.eval()
    
    total_loss_accum = 0.0
    
    for images, target_map, _ in dataloader:
        
        # 1. Move to GPU
        images = images.to(device)
        target_map = target_map.to(device)
        
        # 2. Extract Features
        features_dict = extractor(images)
        features = list(features_dict.values())
        
        # 3. Decode
        raw_logits = decoder(features)
        
        # 4. Upsample
        matched_logits = F.interpolate(
            raw_logits, 
            size=(target_map.shape[2], target_map.shape[3]), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 5. Calculate Loss
        loss, _, _, _ = criterion(matched_logits, target_map)
        total_loss_accum += loss.item()
        
    avg_loss = total_loss_accum / len(dataloader)
    return avg_loss



@torch.no_grad()
def test_model_online(extractor, decoder, dataloader, criterion, device, baseline_prob=None):
    """
    Tests the model end-to-end, computing both continuous losses and discrete metrics.
    """
    extractor.eval()
    decoder.eval()
    
    total_loss, total_kld, total_cc, total_sim = 0.0, 0.0, 0.0, 0.0
    total_nss, total_auc, total_ig = 0.0, 0.0, 0.0

    for images, target_map, fixation_map in dataloader:

        images = images.to(device)
        target_map = target_map.to(device)
        fixation_map = fixation_map.to(device)

        # 1. Extract Features
        features_dict = extractor(images)
        features = list(features_dict.values())

        # 2. Decode
        raw_logits = decoder(features)

        # 3. Upsample
        b, c, target_height, target_width = target_map.shape
        matched_logits = F.interpolate(
            raw_logits,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )

        # 4. Continuous Metrics
        loss, kld, cc, sim = criterion(matched_logits, target_map)
        total_loss += loss.item()
        total_kld += kld.item()
        total_cc += cc.item()
        total_sim += sim.item()
        
        # 5. Discrete Metrics formatting
        pred_prob = F.softmax(matched_logits.view(b, 1, -1), dim=2).view(b, 1, target_height, target_width)
        
        if baseline_prob is None:
            batch_baseline = torch.ones(b, 1, target_height, target_width).to(device) / (target_height * target_width)
        else:
            batch_baseline = baseline_prob.expand(b, -1, -1, -1).to(device)
            
        # NOTE: Make sure nss, auc_judd, and information_gain are imported 
        # from your metrics.py file into training.py!
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