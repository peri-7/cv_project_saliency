"""Compute an empirical center-bias baseline from the SALICON training fixations.

Reads only the .mat fixation files (no images or saliency maps) and saves the
normalized average fixation map to ./saved_models/center_bias_baseline.pt. This
baseline is what scripts/evaluate_ig.py scores Information Gain against.

    python -m scripts.compute_baseline
"""

import os
import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm


FIXATIONS_DIR = '/home/stamata/Downloads/salicon/fixations/train'
OUTPUT_PATH   = './saved_models/center_bias_baseline.pt'
HEIGHT, WIDTH = 480, 640


def parse_fixations(mat_data, height, width):
    """Parse a SALICON .mat file into a binary fixation matrix (same logic as dataset.py)."""
    fixation_matrix = np.zeros((height, width), dtype=np.float32)

    # Official SALICON struct format: a gaze array with one row per subject.
    if 'gaze' in mat_data:
        gaze_data = mat_data['gaze']
        num_subjects = gaze_data.shape[0]

        for i in range(num_subjects):
            subject_fixations = gaze_data[i, 0]['fixations']

            # Unwrap the SciPy object shell if present.
            if subject_fixations.dtype == object and subject_fixations.size == 1:
                subject_fixations = subject_fixations[0, 0]

            # Single-fixation edge case: (2,) -> (1, 2).
            if subject_fixations.ndim == 1 and subject_fixations.size >= 2:
                subject_fixations = subject_fixations.reshape(-1, 2)

            for fix in subject_fixations:
                x, y = int(fix[0]), int(fix[1])
                if 0 <= y < height and 0 <= x < width:
                    fixation_matrix[y, x] = 1.0

    # Fallback: a pre-converted binary matrix stored directly in the file.
    else:
        for key, value in mat_data.items():
            if (not key.startswith('__')
                    and isinstance(value, np.ndarray)
                    and value.shape == (height, width)):
                fixation_matrix = value.astype(np.float32)
                break

    return fixation_matrix


def main():
    mat_files = sorted([
        f for f in os.listdir(FIXATIONS_DIR) if f.endswith('.mat')
    ])

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {FIXATIONS_DIR}")

    print(f"Found {len(mat_files)} fixation files in training set.")
    print("Accumulating fixations...")

    accumulator = np.zeros((HEIGHT, WIDTH), dtype=np.float64)

    for fname in tqdm(mat_files):
        fpath = os.path.join(FIXATIONS_DIR, fname)
        mat_data = sio.loadmat(fpath)
        fixation_matrix = parse_fixations(mat_data, HEIGHT, WIDTH)
        accumulator += fixation_matrix

    # Normalize to a valid probability distribution (sums to 1.0)
    total = accumulator.sum()
    if total == 0:
        raise ValueError("Accumulator is empty — no fixations were parsed.")

    baseline = accumulator / total

    # Convert to torch tensor: [1, 1, H, W]
    baseline_tensor = torch.tensor(baseline, dtype=torch.float32).view(1, 1, HEIGHT, WIDTH)

    # Sanity check
    prob_sum = baseline_tensor.sum().item()
    print(f"\nSanity check — probability mass: {prob_sum:.6f} (should be ~1.0)")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(baseline_tensor, OUTPUT_PATH)
    print(f"Baseline saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()