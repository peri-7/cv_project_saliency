import torch
import torch.nn.functional as F
from src.metrics import nss, auc_judd, information_gain



def train_one_epoch(decoder, dataloader, optimizer, criterion, device):
    """
    Executes one full pass over the training data.
    """
    decoder.train()
    
    total_loss_accum = 0.0
    kld_accum = 0.0
    cc_accum = 0.0
    
    for batch_idx, (features, target_map, _) in enumerate(dataloader):
        
        # 1. Move data to GPU/CPU
        # Note: 'features' is a list/dict of tensors, so we iterate to move them
        if isinstance(features, dict):
            features = [f.to(device) for f in features.values()]
        else:
            features = [f.to(device) for f in features]
            
        target_map = target_map.to(device)
        
        # 2. Zero the gradients
        optimizer.zero_grad()
        
        # 3. Forward Pass: Generate raw logits from the multi-scale features
        raw_logits = decoder(features)
        
        # 4. METHODOLOGICAL RULE: Upsample prediction to immutable ground truth size
        target_height, target_width = target_map.shape[2], target_map.shape[3]
        
        matched_logits = F.interpolate(
            raw_logits, 
            size=(target_height, target_width), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 5. Compute the Composite Loss ($10 \times KLD - 1 \times CC - 1 \times SIM$)
        loss, loss_kld, loss_cc, loss_sim = criterion(matched_logits, target_map)
        
        # 6. Backpropagation and Optimization
        loss.backward()
        optimizer.step()
        
        # Logging metrics
        total_loss_accum += loss.item()
        kld_accum += loss_kld.item()
        cc_accum += loss_cc.item()
        
    avg_loss = total_loss_accum / len(dataloader)
    avg_kld = kld_accum / len(dataloader)
    avg_cc = cc_accum / len(dataloader)
    
    return avg_loss, avg_kld, avg_cc



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
def evaluate_model(decoder, dataloader, criterion, device):
    """
    Evaluates the model on the validation/test set.
    """
    decoder.eval() # Ensure dropout/batchnorm are locked
    
    total_loss_accum = 0.0
    
    for features, target_map, fixation_map in dataloader:
        
        # Move data to device
        if isinstance(features, dict):
            features = [f.to(device) for f in features.values()]
        else:
            features = [f.to(device) for f in features]
            
        target_map = target_map.to(device)
        # Note: For strict metric calculation, we need the discrete fixations too
        
        # Forward pass
        raw_logits = decoder(features)
        
        # Upsample to match ground truth
        target_height, target_width = target_map.shape[2], target_map.shape[3]
        matched_logits = F.interpolate(
            raw_logits, 
            size=(target_height, target_width), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Compute loss (without backprop)
        loss, _, _, _ = criterion(matched_logits, target_map)
        total_loss_accum += loss.item()
        
        
    avg_loss = total_loss_accum / len(dataloader)
    return avg_loss
    


@torch.no_grad()
def test_model(decoder, dataloader, criterion, device, baseline_prob=None):
    """
    Tests the model on the test set, computing both continuous losses and discrete metrics.
    
    Args:
        decoder: The neural network decoding head.
        dataloader: The DataLoader containing the test split.
        criterion: The UNETRSalCompositeLoss function.
        device: CPU or CUDA device.
        baseline_prob (torch.Tensor, optional): The center-bias prior for Information Gain. 
                                                If None, defaults to a uniform distribution.
                                                
    Returns:
        tuple: (avg_loss, avg_kld, avg_cc, avg_nss, avg_auc, avg_ig)
    """
    decoder.eval() # Ensure dropout/batchnorm are strictly locked
    
    # 1. Initialize Accumulators
    total_loss, total_kld, total_cc = 0.0, 0.0, 0.0
    total_nss, total_auc, total_ig = 0.0, 0.0, 0.0
    
    for features, target_map, fixation_map in dataloader:
        
        # 2. Move data to device
        if isinstance(features, dict):
            features = [f.to(device) for f in features.values()]
        else:
            features = [f.to(device) for f in features]
            
        target_map = target_map.to(device)
        fixation_map = fixation_map.to(device) # CRITICAL: Required for metric calculation
        
        # 3. Forward pass
        raw_logits = decoder(features)
        
        # 4. Upsample to match ground truth
        b, c, target_height, target_width = target_map.shape
        matched_logits = F.interpolate(
            raw_logits, 
            size=(target_height, target_width), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 5. Compute continuous losses
        loss, kld, cc, sim = criterion(matched_logits, target_map)
        total_loss += loss.item()
        total_kld += kld.item()
        total_cc += cc.item()
        
        # 6. Format predictions for discrete metrics
        # Information Gain strictly requires a probability distribution summing to 1.0
        pred_prob = F.softmax(matched_logits.view(b, 1, -1), dim=2).view(b, 1, target_height, target_width)
        
        # 7. Configure Information Gain Baseline
        if baseline_prob is None:
            # Fallback: Assume uniform probability across all pixels if no prior is provided
            batch_baseline = torch.ones(b, 1, target_height, target_width).to(device) / (target_height * target_width)
        else:
            # Expand the provided prior to match the current batch size
            batch_baseline = baseline_prob.expand(b, -1, -1, -1).to(device)
            
        # 8. Compute discrete spatial metrics
        total_nss += nss(matched_logits, fixation_map)
        total_auc += auc_judd(matched_logits, fixation_map)
        total_ig += information_gain(pred_prob, fixation_map, batch_baseline)
        
    # 9. Calculate Final Averages
    num_batches = len(dataloader)
    
    avg_loss = total_loss / num_batches
    avg_kld = total_kld / num_batches
    avg_cc = total_cc / num_batches
    avg_nss = total_nss / num_batches
    avg_auc = total_auc / num_batches
    avg_ig = total_ig / num_batches

    return avg_loss, avg_kld, avg_cc, avg_nss, avg_auc, avg_ig