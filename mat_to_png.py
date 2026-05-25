"""
Convert a SALICON .mat fixation file to a binary PNG fixation map.
Usage: python mat_to_png.py <input.mat> [output.png] [--radius N]
  --radius N  Radius of each fixation dot in pixels (default: 8)
If output path is omitted, saves alongside the input file with a .png extension.
"""

import sys
import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw


def mat_to_fixation_map(mat_path: str, height: int = 480, width: int = 640) -> np.ndarray:
    mat_data = sio.loadmat(mat_path)
    fixation_matrix = np.zeros((height, width), dtype=np.float32)

    if 'gaze' in mat_data:
        gaze_data = mat_data['gaze']
        # SALICON: gaze is a (num_subjects, 1) struct array
        num_subjects = gaze_data.shape[0]
        for i in range(num_subjects):
            subject_fixations = gaze_data[i, 0]['fixations']

            # Defensive: handle object-wrapped or 1-D edge cases
            if subject_fixations.dtype == object and subject_fixations.size == 1:
                subject_fixations = subject_fixations[0, 0]
            if subject_fixations.ndim == 1 and subject_fixations.size >= 2:
                subject_fixations = subject_fixations.reshape(-1, 2)

            for fix in subject_fixations:
                x, y = int(fix[0]), int(fix[1])
                if 0 <= y < height and 0 <= x < width:
                    fixation_matrix[y, x] = 1.0
    else:
        for key, value in mat_data.items():
            if not key.startswith('__') and isinstance(value, np.ndarray) and value.shape == (height, width):
                fixation_matrix = value.astype(np.float32)
                break

    return fixation_matrix


def draw_dots(fixation_matrix: np.ndarray, radius: int = 8) -> Image.Image:
    h, w = fixation_matrix.shape
    img = Image.new('L', (w, h), color=0)
    draw = ImageDraw.Draw(img)
    ys, xs = np.where(fixation_matrix > 0)
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)
    return img


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python mat_to_png.py <input.mat> [output.png] [--radius N]")
        sys.exit(1)

    mat_path = args.pop(0)
    png_path = None
    radius = 8

    i = 0
    while i < len(args):
        if args[i] == '--radius' and i + 1 < len(args):
            radius = int(args[i + 1])
            i += 2
        elif not args[i].startswith('--'):
            png_path = args[i]
            i += 1
        else:
            i += 1

    if png_path is None:
        png_path = mat_path.rsplit('.', 1)[0] + '.png'

    fixation_matrix = mat_to_fixation_map(mat_path)
    img = draw_dots(fixation_matrix, radius=radius)
    img.save(png_path)
    print(f"Saved fixation map ({int(fixation_matrix.sum())} fixations, radius={radius}) -> {png_path}")


if __name__ == '__main__':
    main()
