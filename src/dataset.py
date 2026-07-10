import os
import torch
import numpy as np
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# Raw RGB images fed to the backbone (feature-caching pass).
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

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Filename is returned so the cached feature file can reuse it.
        return image, img_name


# Pre-extracted backbone features + ground truth, fed to the decoder.
class FeatureDataset(Dataset):

    def __init__(self, features_dir, maps_dir, fixations_dir, map_transform=None):
        self.features_dir = features_dir
        self.maps_dir = maps_dir
        self.fixations_dir = fixations_dir

        # Assumes every feature file has a matching map and fixation file.
        self.feature_filenames = sorted(os.listdir(features_dir))
        self.map_transform = map_transform

    def __len__(self):
        return len(self.feature_filenames)

    def __getitem__(self, idx):
        feat_name = self.feature_filenames[idx]
        base_name = os.path.splitext(feat_name)[0]

        feat_path = os.path.join(self.features_dir, feat_name)
        extracted_features = torch.load(feat_path)
        #extracted_features = {k: v.squeeze(0) for k, v in extracted_features.items()}

        # Continuous saliency map (grayscale), used by the loss.
        map_path = os.path.join(self.maps_dir, base_name + '.png')
        saliency_map = Image.open(map_path).convert('L')

        # Discrete fixations from the .mat file.
        fix_path = os.path.join(self.fixations_dir, base_name + '.mat')
        mat_data = sio.loadmat(fix_path)

        fixation_matrix = np.zeros((480, 640), dtype=np.float32)

        # Official SALICON format: a (num_subjects, 1) gaze struct.
        if 'gaze' in mat_data:
            gaze_data = mat_data['gaze']
            num_subjects = gaze_data.shape[0]
            for i in range(num_subjects):

                subject_fixations = gaze_data[i, 0]['fixations']

                # SciPy may wrap the array in a 1-element object array; unwrap it.
                if subject_fixations.dtype == object and subject_fixations.size == 1:
                    subject_fixations = subject_fixations[0, 0]

                # A single fixation collapses (1, 2) to (2,); force it back to 2D.
                if subject_fixations.ndim == 1 and subject_fixations.size >= 2:
                    subject_fixations = subject_fixations.reshape(-1, 2)

                for fix in subject_fixations:
                    x, y = int(fix[0]), int(fix[1])
                    if 0 <= y < 480 and 0 <= x < 640:
                        fixation_matrix[y, x] = 1.0

        # Fallback: the mini-dataset may already store a binary matrix.
        else:
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray) and value.shape == (480, 640):
                    fixation_matrix = value
                    break

        # Wrap the binary matrix as a PIL image so the transform pipeline applies.
        fixation_map = Image.fromarray((fixation_matrix * 255.0).astype(np.uint8)).convert('L')

        if self.map_transform:
            saliency_map = self.map_transform(saliency_map)
            fixation_map = self.map_transform(fixation_map)
        else:
            saliency_map = transforms.ToTensor()(saliency_map)
            fixation_map = transforms.ToTensor()(fixation_map)

        # Keep the fixation map strictly binary.
        fixation_map = (fixation_map > 0.0).float()

        return extracted_features, saliency_map, fixation_map


# Raw images + ground truth fed to the full LoRA-adapted model.
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

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.image_transform:
            image = self.image_transform(image)

        # Continuous saliency map (grayscale), used by the loss.
        map_path = os.path.join(self.maps_dir, base_name + '.png')
        saliency_map = Image.open(map_path).convert('L')

        # Discrete fixations from the .mat file.
        fix_path = os.path.join(self.fixations_dir, base_name + '.mat')
        mat_data = sio.loadmat(fix_path)

        fixation_matrix = np.zeros((480, 640), dtype=np.float32)

        # Official SALICON format: a (num_subjects, 1) gaze struct.
        if 'gaze' in mat_data:
            gaze_data = mat_data['gaze']
            num_subjects = gaze_data.shape[0]
            for i in range(num_subjects):

                subject_fixations = gaze_data[i, 0]['fixations']

                # SciPy may wrap the array in a 1-element object array; unwrap it.
                if subject_fixations.dtype == object and subject_fixations.size == 1:
                    subject_fixations = subject_fixations[0, 0]

                # A single fixation collapses (1, 2) to (2,); force it back to 2D.
                if subject_fixations.ndim == 1 and subject_fixations.size >= 2:
                    subject_fixations = subject_fixations.reshape(-1, 2)

                for fix in subject_fixations:
                    x, y = int(fix[0]), int(fix[1])
                    if 0 <= y < 480 and 0 <= x < 640:
                        fixation_matrix[y, x] = 1.0

        # Fallback: the mini-dataset may already store a binary matrix.
        else:
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray) and value.shape == (480, 640):
                    fixation_matrix = value
                    break

        # Wrap the binary matrix as a PIL image so the transform pipeline applies.
        fixation_map = Image.fromarray((fixation_matrix * 255.0).astype(np.uint8)).convert('L')

        if self.map_transform:
            saliency_map = self.map_transform(saliency_map)
            fixation_map = self.map_transform(fixation_map)
        else:
            saliency_map = transforms.ToTensor()(saliency_map)
            fixation_map = transforms.ToTensor()(fixation_map)

        # Keep the fixation map strictly binary.
        fixation_map = (fixation_map > 0.0).float()

        return image, saliency_map, fixation_map
