import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte

from basicsr.models import create_model
from basicsr.utils.options import parse


def unwrap_model_output(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    if isinstance(output, dict):
        if 'result' in output:
            return output['result']
        return next(iter(output.values()))
    return output


def select_checkpoint_state(checkpoint, ema_mode='auto'):
    if ema_mode == 'on':
        if 'params_ema' not in checkpoint:
            raise KeyError("EMA weights requested but 'params_ema' not found in checkpoint.")
        print("✅ Using EMA weights: 'params_ema'.")
        return checkpoint['params_ema'], 'params_ema'
    if ema_mode == 'off':
        if 'params' in checkpoint:
            print("✅ Using standard weights: 'params'.")
            return checkpoint['params'], 'params'
        print("⚠️ 'params' not found, falling back to raw checkpoint as state_dict...")
        return checkpoint, 'raw'

    if 'params_ema' in checkpoint:
        print("✅ Found EMA weights in checkpoint, loading 'params_ema'...")
        return checkpoint['params_ema'], 'params_ema'
    if 'params' in checkpoint:
        print("⚠️ EMA weights not found, falling back to standard 'params'...")
        return checkpoint['params'], 'params'
    print("⚠️ 'params_ema' / 'params' not found, trying raw checkpoint as state_dict...")
    return checkpoint, 'raw'


def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))

        x = unwrap_model_output(model(x))

        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x

    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)


parser = argparse.ArgumentParser(description='Image Enhancement Testing for SeaClarityNetV5A')
parser.add_argument('--opt', type=str, default='Options/UIEB.yml', help='Path to option YAML file')
parser.add_argument('--weights', type=str,default='./pretrained_weights/net_g_best_psnr.pth', help='Path to weights')
parser.add_argument('--dataset', default='UIEB', type=str, help='Dataset name')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--output_dir', default='', type=str, help='Directory for output images')
parser.add_argument('--gpus', type=str, default='0', help='GPU devices')
parser.add_argument('--GT_mean', action='store_true', help='Use GT mean to rectify model output')
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble')
parser.add_argument('--ema', type=str, default='auto', choices=['auto', 'on', 'off'], help='Checkpoint weight selection: auto prefers params_ema, on requires it, off uses params')
args = parser.parse_args()

gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

opt = parse(args.opt, is_train=False)
opt['dist'] = False

print(f"dataset {args.dataset}")

model_restoration = create_model(opt).net_g

checkpoint = torch.load(args.weights, map_location='cpu')
state_dict, loaded_param_key = select_checkpoint_state(checkpoint, args.ema)
print(f"Checkpoint param_key: {loaded_param_key}")

try:
    model_restoration.load_state_dict(state_dict, strict=True)
except Exception:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    try:
        model_restoration.load_state_dict(new_state_dict, strict=True)
        print("✅ Load state_dict success (removed 'module.' prefix).")
    except Exception:
        print("⚠️ Direct load and prefix removal failed, attempting to add 'module.' prefix...")
        new_state_dict_add = {}
        for k, v in state_dict.items():
            new_state_dict_add['module.' + k] = v
        model_restoration.load_state_dict(new_state_dict_add, strict=False)
        print("✅ Load state_dict success (forced adding 'module.' prefix).")

print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

factor = int(opt.get('val', {}).get('window_size', 4))
dataset_name = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]

result_dir = os.path.join(args.result_dir, dataset_name, config, checkpoint_name)
output_dir = args.output_dir

os.makedirs(result_dir, exist_ok=True)
if output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

psnr = []
ssim = []

if dataset_name in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
    if dataset_name == 'SID':
        from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset_name == 'SMID':
        from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    else:
        from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset

    val_opt = opt['datasets']['val']
    val_opt['phase'] = 'test'
    if val_opt.get('scale') is None:
        val_opt['scale'] = 1

    if '~' in val_opt['dataroot_gt']:
        val_opt['dataroot_gt'] = os.path.expanduser('~') + val_opt['dataroot_gt'][1:]
    if '~' in val_opt['dataroot_lq']:
        val_opt['dataroot_lq'] = os.path.expanduser('~') + val_opt['dataroot_lq'][1:]

    dataset = Dataset(val_opt)
    print(f'test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            target = data_batch['gt'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = unwrap_model_output(model_restoration(input_))

            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / max(mean_restored, 1e-8)), 0, 1)

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))

            out_name = os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png'
            save_path = os.path.join(output_dir if output_dir else result_dir, out_name)
            utils.save_img(save_path, img_as_ubyte(restored))

else:
    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
    print(input_dir)
    print(target_dir)

    exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
    input_paths = []
    target_paths = []
    for ext in exts:
        input_paths.extend(glob(os.path.join(input_dir, ext)))
        target_paths.extend(glob(os.path.join(target_dir, ext)))

    input_paths = natsorted(input_paths)
    target_paths = natsorted(target_paths)

    if len(input_paths) != len(target_paths):
        raise ValueError(f'Input/GT count mismatch: {len(input_paths)} vs {len(target_paths)}')

    with torch.inference_mode():
        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255.

            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if h < 3000 and w < 3000:
                if args.self_ensemble:
                    restored = self_ensemble(input_, model_restoration)
                else:
                    restored = unwrap_model_output(model_restoration(input_))
            else:
                input_1 = input_[:, :, :, 1::2]
                input_2 = input_[:, :, :, 0::2]
                if args.self_ensemble:
                    restored_1 = self_ensemble(input_1, model_restoration)
                    restored_2 = self_ensemble(input_2, model_restoration)
                else:
                    restored_1 = unwrap_model_output(model_restoration(input_1))
                    restored_2 = unwrap_model_output(model_restoration(input_2))
                restored = torch.zeros_like(input_)
                restored[:, :, :, 1::2] = restored_1
                restored[:, :, :, 0::2] = restored_2

            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / max(mean_restored, 1e-8)), 0, 1)

            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))

            out_name = os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png'
            save_path = os.path.join(output_dir if output_dir else result_dir, out_name)
            utils.save_img(save_path, img_as_ubyte(restored))

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("Final Results for %s:" % dataset_name)
print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))
