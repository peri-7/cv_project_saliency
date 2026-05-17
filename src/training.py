import torch
import torch.nn.functional as F

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
        
        # NOTE: In your final Jupyter Notebook, you will import the functions 
        # from metrics.py here to calculate and print the NSS, AUC, and IG scores.
        
    avg_loss = total_loss_accum / len(dataloader)
    return avg_loss