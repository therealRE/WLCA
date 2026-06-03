import argparse
import glob
import os

import cv2
import einops
import numpy as np
import torch

from ciconv2d0 import MultiInvariantPrior


PRIOR_INVARIANTS = ('W', 'H', 'C')


def prior_image(tensor):
    x = tensor[0] if tensor.ndim == 4 else tensor
    x = x.detach().cpu().clamp(0, 1).numpy()
    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)
    x = x.transpose(1, 2, 0)
    return (x * 255.0).astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='test/input', type=str)
    parser.add_argument('--output_folder', default='test/input_s', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    extraction_model = MultiInvariantPrior(invariants=PRIOR_INVARIANTS, k=3, scale=0.9).to(args.device).eval()

    img_list = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
        img_list.extend(glob.glob(os.path.join(args.input_folder, ext)))
    img_list = sorted(img_list)
    print(f'Find {len(img_list)} files in {args.input_folder}')

    os.makedirs(args.output_folder, exist_ok=True)

    for img_path in img_list:
        input_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if input_image is None:
            print(f'[skip] broken file: {img_path}')
            continue
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        input_tensor = (torch.from_numpy(input_image.copy()).to(args.device, dtype=torch.float32) / 255.0).unsqueeze(0)
        input_tensor = einops.rearrange(input_tensor, 'b h w c -> b c h w').contiguous()

        with torch.no_grad():
            features = extraction_model(input_tensor)

        save_path = os.path.join(args.output_folder, os.path.basename(img_path))
        cv2.imwrite(save_path, prior_image(features))
        print(f'saved {save_path}')
