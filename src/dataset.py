import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# Loads the raw RGB images and feeds them to the backbone
class RawDataset(Dataset):

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image and ensure it has 3 RGB channels
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # We return the filename so we know what to name the saved feature tensor later
        return image, img_name



# Loads the features extracted by the backbone along with the GT and feeds them to the decoding head
class FeatureDataset(Dataset):

    def __init__(self, features_dir, maps_dir, fixations_dir, map_transform=None):
        self.features_dir = features_dir
        self.maps_dir = maps_dir
        self.fixations_dir = fixations_dir
        
        # We assume every saved feature file has a corresponding map and fixation file
        self.feature_filenames = sorted(os.listdir(features_dir))
        self.map_transform = map_transform

    def __len__(self):
        return len(self.feature_filenames)

    def __getitem__(self, idx):
        feat_name = self.feature_filenames[idx]
        base_name = os.path.splitext(feat_name)[0] # Strip the .pt extension
        
        # 1. Load the pre-extracted multi-scale feature dictionary
        feat_path = os.path.join(self.features_dir, feat_name)
        extracted_features = torch.load(feat_path)
        # Strip the dummy batch dimension: [1, C, H, W] becomes [C, H, W]
        extracted_features = {k: v.squeeze(0) for k, v in extracted_features.items()}
        
        # 2. Load the Continuous Saliency Map (For KLD, CC, SIM loss)
        # Saliency maps are usually provided as grayscale images
        map_path = os.path.join(self.maps_dir, base_name + '.png')
        saliency_map = Image.open(map_path).convert('L')
        
        # 3. Load the Discrete Fixation Map (For NSS, AUC, IG evaluation)
        # Fixation maps are binary (0 for no fixation, 255 for fixation)
        fix_path = os.path.join(self.fixations_dir, base_name + '.png')
        fixation_map = Image.open(fix_path).convert('L')
        
        # 4. Transform maps to PyTorch Tensors [1, H, W]
        if self.map_transform:
            saliency_map = self.map_transform(saliency_map)
            fixation_map = self.map_transform(fixation_map)
        else:
            # Default fallback: convert PIL image to tensor and normalize to [0, 1]
            saliency_map = transforms.ToTensor()(saliency_map)
            fixation_map = transforms.ToTensor()(fixation_map)
            
        # Ensure fixation map is strictly binary (0.0 or 1.0)
        fixation_map = (fixation_map > 0.5).float()
            
        return extracted_features, saliency_map, fixation_map
        
      
# Loads images from MS COCO + the GT and feeds them to the full model 
class LoraDataset(Dataset):
    
    def __init__(self, image_dir, maps_dir, fixations_dir, image_transform=None, map_transform=None):
        self.image_dir = image_dir
        self.maps_dir = maps_dir
        self.fixations_dir = fixations_dir
        
        self.image_filenames = sorted(os.listdir(image_dir))
        self.image_transform = image_transform
        self.map_transform = map_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # 1. Load and Transform the Raw RGB Image (for the LoRA-adapted backbone)
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.image_transform:
            image = self.image_transform(image)
            
        # 2. Load the Continuous Saliency Map (For KLD, CC, SIM loss)
        map_path = os.path.join(self.maps_dir, base_name + '.png')
        saliency_map = Image.open(map_path).convert('L')
        
        # 3. Load the Discrete Fixation Map (For Evaluation Metrics)
        fix_path = os.path.join(self.fixations_dir, base_name + '.png')
        fixation_map = Image.open(fix_path).convert('L')
        
        # 4. Transform Target Maps
        if self.map_transform:
            saliency_map = self.map_transform(saliency_map)
            fixation_map = self.map_transform(fixation_map)
        else:
            saliency_map = transforms.ToTensor()(saliency_map)
            fixation_map = transforms.ToTensor()(fixation_map)
            
        # Enforce strict binary values for the evaluation matrix
        fixation_map = (fixation_map > 0.5).float()
            
        return image, saliency_map, fixation_map