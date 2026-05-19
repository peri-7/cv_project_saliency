import os
import torch
import numpy as np
from PIL import Image
import scipy.io as sio
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
        #extracted_features = {k: v.squeeze(0) for k, v in extracted_features.items()}
        
        # 2. Load the Continuous Saliency Map (For KLD, CC, SIM loss)
        # Saliency maps are usually provided as grayscale images
        map_path = os.path.join(self.maps_dir, base_name + '.png')
        saliency_map = Image.open(map_path).convert('L')
        
        # 3. Load the Discrete Fixation Map (.mat file)
        fix_path = os.path.join(self.fixations_dir, base_name + '.mat')
        mat_data = sio.loadmat(fix_path)
        
        # We synthesize our own blank ground-truth matrix
        fixation_matrix = np.zeros((480, 640), dtype=np.float32)
        
        # Strategy A: Official SALICON Struct Format (gaze coordinates)
        if 'gaze' in mat_data:
            gaze_data = mat_data['gaze']
            # Loop through each human subject's data
            for i in range(gaze_data.shape[1]):
                
                # Extract the fixations for this subject
                subject_fixations = gaze_data[0, i]['fixations']
                
                # 1. THE UNWRAPPER: If SciPy wrapped it in an object shell, break it open.
                # If it didn't, this safely ignores it.
                if subject_fixations.dtype == object and subject_fixations.size == 1:
                    subject_fixations = subject_fixations[0, 0]
                    
                # 2. EDGE CASE: If a subject only made 1 single fixation, 
                # numpy flattens shape (1, 2) into just (2,). We force it back to 2D.
                if subject_fixations.ndim == 1 and subject_fixations.size >= 2:
                    subject_fixations = subject_fixations.reshape(-1, 2)
                    
                # 3. PAINT THE MATRIX
                for fix in subject_fixations:
                    x, y = int(fix[0]), int(fix[1])
                    if 0 <= y < 480 and 0 <= x < 640:
                        fixation_matrix[y, x] = 1.0
                        
        # Strategy B: Fallback (in case the mini-dataset already converted them to binary matrices)
        else:
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray) and value.shape == (480, 640):
                    fixation_matrix = value
                    break

        # Convert our pristine binary matrix into a PIL Image for the transform pipeline
        fixation_map = Image.fromarray((fixation_matrix * 255.0).astype(np.uint8)).convert('L')
        
        # 4. Transform maps to PyTorch Tensors [1, H, W]
        if self.map_transform:
            saliency_map = self.map_transform(saliency_map)
            fixation_map = self.map_transform(fixation_map)
        else:
            # Default fallback: convert PIL image to tensor and normalize to [0, 1]
            saliency_map = transforms.ToTensor()(saliency_map)
            fixation_map = transforms.ToTensor()(fixation_map)
            
        # Ensure fixation map is strictly binary (0.0 or 1.0)
        fixation_map = (fixation_map > 0.0).float()
            
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
        
        # 3. Load the Discrete Fixation Map (.mat file)
        fix_path = os.path.join(self.fixations_dir, base_name + '.mat')
        mat_data = sio.loadmat(fix_path)
        
        # We synthesize our own blank ground-truth matrix
        fixation_matrix = np.zeros((480, 640), dtype=np.float32)
        
        # Strategy A: Official SALICON Struct Format (gaze coordinates)
        if 'gaze' in mat_data:
            gaze_data = mat_data['gaze']
            # Loop through each human subject's data
            for i in range(gaze_data.shape[1]):
                
                # Extract the fixations for this subject
                subject_fixations = gaze_data[0, i]['fixations']
                
                # 1. THE UNWRAPPER: If SciPy wrapped it in an object shell, break it open.
                # If it didn't, this safely ignores it.
                if subject_fixations.dtype == object and subject_fixations.size == 1:
                    subject_fixations = subject_fixations[0, 0]
                    
                # 2. EDGE CASE: If a subject only made 1 single fixation, 
                # numpy flattens shape (1, 2) into just (2,). We force it back to 2D.
                if subject_fixations.ndim == 1 and subject_fixations.size >= 2:
                    subject_fixations = subject_fixations.reshape(-1, 2)
                    
                # 3. PAINT THE MATRIX
                for fix in subject_fixations:
                    x, y = int(fix[0]), int(fix[1])
                    if 0 <= y < 480 and 0 <= x < 640:
                        fixation_matrix[y, x] = 1.0
                        
        # Strategy B: Fallback (in case the mini-dataset already converted them to binary matrices)
        else:
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray) and value.shape == (480, 640):
                    fixation_matrix = value
                    break

        # Convert our pristine binary matrix into a PIL Image for the transform pipeline
        fixation_map = Image.fromarray((fixation_matrix * 255.0).astype(np.uint8)).convert('L')
        
        # 4. Transform Target Maps
        if self.map_transform:
            saliency_map = self.map_transform(saliency_map)
            fixation_map = self.map_transform(fixation_map)
        else:
            saliency_map = transforms.ToTensor()(saliency_map)
            fixation_map = transforms.ToTensor()(fixation_map)
            
        # Enforce strict binary values for the evaluation matrix
        fixation_map = (fixation_map > 0.0).float()
            
        return image, saliency_map, fixation_map