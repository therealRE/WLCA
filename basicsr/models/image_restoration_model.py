import importlib
import os
import random
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from os import path as osp

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)
        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class ImageCleanModel(BaseModel):
    """Base restoration model for single image enhancement/restoration."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # Baseline mode: single-input restoration only.
        # Online structure priors / guidance and color-perturbation branches are disabled.
        self.use_online_prior = False
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        self.input_gate_cfg = {}
        self.use_learnable_input_gate = False
        self.input_gate_net = None

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(
                self.net_g,
                load_path,
                self.opt['path'].get('strict_load_g', True),
                param_key=self.opt['path'].get('param_key', 'auto')
            )


        self.best_metric = float('-inf')
        self.best_metric_name = None
        self.best_model_path = None

        self.current_epoch = 0
        self.total_epochs = 1
        self.current_iter = 0
        self.total_iters = int(self.opt.get('train', {}).get('total_iter', 1)) if self.is_train else 1
        self.train_stage = 'clean'
        self.guide_lq = None
        self.input_gate = None
        self.input_gate_stats = OrderedDict()
        self.rule_input_gate = None
        self.learned_input_gate = None

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.l1 = nn.L1Loss()
        self.multi_input_loss_names = {'multi_VGGPerceptualLoss'}
        self.loss_configs = OrderedDict()
        self.log_scale_losses = train_opt.get('log_scale_losses', False)

        self.ema_decay = train_opt.get('ema_decay', 0)
        self.ema_eval_start_iter = int(train_opt.get('ema_eval_start_iter', 0))
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')

            self.net_g_ema = define_network(deepcopy(self.opt['network_g'])).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt['path'].get('strict_load_g', True),
                    'params_ema'
                )
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        self._register_loss(train_opt, 'pixel_opt', 'l_pix')
        self._register_loss(train_opt, 'perceptual_opt', 'l_percep')
        self._register_loss(train_opt, 'ssim_opt', 'l_ssim')
        self._register_loss(train_opt, 'histogram_opt', 'l_hist')
        self._register_loss(train_opt, 'color_stats_opt', 'l_color')

        if len(self.loss_configs) == 0:
            raise ValueError('All losses are None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def set_training_progress(self, epoch, total_epochs, current_iter=None, total_iters=None):
        self.current_epoch = int(epoch)
        self.total_epochs = max(int(total_epochs), 1)
        if current_iter is not None:
            self.current_iter = int(current_iter)
        if total_iters is not None:
            self.total_iters = max(int(total_iters), 1)

    def _forward_generator(self, img, img_s=None, guide_x=None):
        return self.net_g(img)

    def _forward_generator_ema(self, img, img_s=None, guide_x=None):
        return self.net_g_ema(img)


    def _register_loss(self, train_opt, opt_key, log_prefix):
        if not train_opt.get(opt_key):
            return

        loss_opt = deepcopy(train_opt[opt_key])
        loss_type = loss_opt.pop('type')
        multi_scale_weights = loss_opt.pop('multi_scale_weights', None)
        cri_cls = getattr(loss_module, loss_type)
        criterion = cri_cls(**loss_opt).to(self.device)

        self.loss_configs[opt_key] = {
            'type': loss_type,
            'criterion': criterion,
            'mode': 'multi_input' if loss_type in self.multi_input_loss_names else 'single_input',
            'multi_scale_weights': multi_scale_weights,
            'log_prefix': log_prefix,
        }

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_g_opt = deepcopy(train_opt['optim_g'])
        optim_type = optim_g_opt.pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **optim_g_opt)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **optim_g_opt)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    @staticmethod
    def _match_spatial_size(src, ref):
        if src.shape[-2:] != ref.shape[-2:]:
            src = F.interpolate(src, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return src

    def _build_gt_for_pred(self, pred, gt):
        if pred.shape[-2:] == gt.shape[-2:]:
            return gt
        return F.interpolate(gt, size=pred.shape[-2:], mode='bilinear', align_corners=False)

    def _normalize_preds(self, preds):
        if isinstance(preds, (list, tuple)):
            return list(preds)
        return [preds]

    @staticmethod
    def _is_scalar_tensor(value):
        return torch.is_tensor(value) and value.ndim == 0

    def _get_scale_weights(self, loss_cfg, num_preds):
        weights = loss_cfg.get('multi_scale_weights', None)
        if weights is None:
            return [1.0] * num_preds
        weights = list(weights)
        if len(weights) < num_preds:
            weights = weights + [weights[-1]] * (num_preds - len(weights))
        return weights[:num_preds]

    def _parse_multi_input_loss_output(self, loss_out, log_prefix):
        if isinstance(loss_out, tuple):
            if len(loss_out) == 2 and isinstance(loss_out[1], dict):
                loss_val, aux_dict = loss_out
                parsed = OrderedDict()
                for key, value in aux_dict.items():
                    parsed[key] = value
                if log_prefix not in parsed:
                    parsed[log_prefix] = loss_val
                return loss_val, parsed
            raise TypeError(f'Unsupported multi-input loss output for {log_prefix}: {type(loss_out)}')

        return loss_out, OrderedDict({log_prefix: loss_out})

    def _compute_standard_loss(self, opt_key, loss_cfg, preds):
        criterion = loss_cfg['criterion']
        log_prefix = loss_cfg['log_prefix']
        scale_weights = self._get_scale_weights(loss_cfg, len(preds))

        device = preds[0].device
        total_loss = torch.zeros((), device=device)
        total_logs = OrderedDict()

        if opt_key == 'perceptual_opt':
            total_percep = torch.zeros((), device=device)
            total_style = torch.zeros((), device=device)
            for idx, (pred, weight) in enumerate(zip(preds, scale_weights)):
                gt_i = self._build_gt_for_pred(pred, self.gt)
                percep_loss, style_loss = criterion(pred, gt_i)
                if percep_loss is not None:
                    weighted_percep = float(weight) * percep_loss
                    total_percep = total_percep + weighted_percep
                    total_loss = total_loss + weighted_percep
                    if self.log_scale_losses:
                        total_logs[f'l_percep_{idx + 1}'] = percep_loss
                if style_loss is not None:
                    weighted_style = float(weight) * style_loss
                    total_style = total_style + weighted_style
                    total_loss = total_loss + weighted_style
                    if self.log_scale_losses:
                        total_logs[f'l_style_{idx + 1}'] = style_loss
            total_logs['l_percep'] = total_percep
            if torch.any(total_style != 0):
                total_logs['l_style'] = total_style
            return total_loss, total_logs

        for idx, (pred, weight) in enumerate(zip(preds, scale_weights)):
            gt_i = self._build_gt_for_pred(pred, self.gt)
            loss_out = criterion(pred, gt_i)
            if isinstance(loss_out, tuple):
                if len(loss_out) == 2 and isinstance(loss_out[1], dict):
                    loss_i = loss_out[0]
                    aux_dict = loss_out[1]
                else:
                    raise TypeError(f'Unsupported loss output for {log_prefix}: {type(loss_out)}')
            else:
                loss_i = loss_out
                aux_dict = None

            weighted_loss_i = float(weight) * loss_i
            total_loss = total_loss + weighted_loss_i
            if self.log_scale_losses:
                total_logs[f'{log_prefix}_{idx + 1}'] = loss_i
                if aux_dict is not None:
                    for key, value in aux_dict.items():
                        total_logs[f'{key}_{idx + 1}'] = value

        total_logs[log_prefix] = total_loss
        return total_loss, total_logs

    def _compute_loss(self, opt_key, loss_cfg, preds):
        if loss_cfg['mode'] == 'multi_input':
            if len(preds) < 3:
                raise ValueError(f"{loss_cfg['type']} expects at least 3 predictions, but got {len(preds)}")
            loss_out = loss_cfg['criterion'](preds[0], preds[1], preds[2], self.gt)
            return self._parse_multi_input_loss_output(loss_out, loss_cfg['log_prefix'])

        return self._compute_standard_loss(opt_key, loss_cfg, preds)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_path = data.get('lq_path', None)
        self.train_stage = 'clean'
        self.guide_lq = None
        self.lq_s = None

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if hasattr(self, 'gt_s'):
            del self.gt_s

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_path = data.get('lq_path', None)
        self.guide_lq = None
        self.train_stage = 'clean'
        self.lq_s = None

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if hasattr(self, 'gt_s'):
            del self.gt_s

    def _set_model_progress(self, current_iter):
        total_iter = max(int(self.opt.get('train', {}).get('total_iter', 1)), 1)
        progress = float(current_iter) / float(total_iter)
        for net_name in ('net_g', 'net_g_ema'):
            net = getattr(self, net_name, None)
            if net is not None:
                bare = self.get_bare_model(net)
                if hasattr(bare, 'set_training_progress'):
                    bare.set_training_progress(progress)

    def optimize_parameters(self, current_iter):
        self._set_model_progress(current_iter)
        self.optimizer_g.zero_grad()

        preds = self._forward_generator(self.lq)
        preds = self._normalize_preds(preds)

        total_loss = torch.zeros((), device=preds[0].device)
        loss_dict = OrderedDict()

        for opt_key, loss_cfg in self.loss_configs.items():
            loss_val, sub_loss_dict = self._compute_loss(opt_key, loss_cfg, preds)
            total_loss = total_loss + loss_val
            loss_dict.update(sub_loss_dict)

        total_loss.backward()
        if self.opt['train']['use_grad_clip']:
            grad_params = [p for p in self.net_g.parameters() if p.requires_grad]
            clip_val = float(self.opt['train'].get('grad_clip_val', 0.01))
            torch.nn.utils.clip_grad_norm_(grad_params, clip_val)
        self.optimizer_g.step()

        loss_dict['loss_total'] = total_loss
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size, current_iter=None):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size

        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img=img, current_iter=current_iter)

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img_s=None, img=None, current_iter=None):
        if img is None:
            img = self.lq

        if current_iter is not None:
            self._set_model_progress(current_iter)

        use_ema = hasattr(self, 'net_g_ema') and self.current_iter >= self.ema_eval_start_iter
        if use_ema:
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self._forward_generator_ema(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self._forward_generator(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        return 0.

    def _get_monitor_metric_name(self):
        val_opt = self.opt.get('val', {})
        monitor_metric = val_opt.get('monitor_metric', None)
        if monitor_metric:
            return monitor_metric
        metrics = val_opt.get('metrics', None)
        if not metrics:
            return None
        if 'psnr' in metrics:
            return 'psnr'
        return next(iter(metrics.keys()))

    def _save_best_network(self, current_iter, metric_name, metric_value):
        exp_root = self.opt['path'].get('experiments_root', None)
        if not exp_root:
            models_dir = self.opt['path']['models']
            exp_root = osp.dirname(models_dir)
        os.makedirs(exp_root, exist_ok=True)

        save_path = osp.join(exp_root, f'net_g_best_{metric_name}.pth')
        info_path = osp.join(exp_root, f'best_{metric_name}.txt')

        net = [self.net_g, self.net_g_ema] if self.ema_decay > 0 else [self.net_g]
        param_key = ['params', 'params_ema'] if self.ema_decay > 0 else ['params']

        save_dict = {
            'meta': {
                'iter': int(current_iter),
                'metric_name': metric_name,
                'metric_value': float(metric_value),
            }
        }
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            clean_state_dict = OrderedDict()
            for key, param in state_dict.items():
                if key.startswith('module.'):
                    key = key[7:]
                clean_state_dict[key] = param.cpu()
            save_dict[param_key_] = clean_state_dict

        torch.save(save_dict, save_path)
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f'best_{metric_name}: {metric_value:.4f}\n')
            f.write(f'iter: {current_iter}\n')
            f.write(f'checkpoint: {osp.basename(save_path)}\n')

        self.best_model_path = save_path

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0 for metric in self.opt['val']['metrics'].keys()
            }

        window_size = self.opt['val'].get('window_size', 0)
        test = partial(self.pad_test, window_size, current_iter=current_iter) if window_size else partial(self.nonpad_test, current_iter=current_iter)

        cnt = 0
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            if hasattr(self, 'gt_s'):
                del self.gt_s
            if hasattr(self, 'lq_s'):
                del self.lq_s

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], str(current_iter), f'{img_name}.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

            monitor_metric = self._get_monitor_metric_name()
            if monitor_metric is not None:
                current_metric = self.metric_results[monitor_metric]
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.best_metric_name = monitor_metric
                    logger = get_root_logger()
                    self._save_best_network(current_iter, monitor_metric, current_metric)
                    logger.info(
                        f'Found better {monitor_metric}: {current_metric:.4f}. '
                        f'Saved/overwritten best model at {self.best_model_path}')
            else:
                current_metric = 0.

        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            nets = [self.net_g, self.net_g_ema]
            param_keys = ['params', 'params_ema']
        else:
            nets = [self.net_g]
            param_keys = ['params']

        self.save_network(nets, 'net_g', current_iter, param_key=param_keys)
        self.save_training_state(epoch, current_iter)
