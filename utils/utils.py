import logging
import sys

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))
        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def edge_aware_smoothness(disp, image, alpha=10.0):
    """
    disp:  [B,1,H,W]
    image: [B,3,H,W] or [B,1,H,W]  (用于边缘权重)
    """
    # image 转灰度也行，这里简单用均值
    if image.shape[1] > 1:
        img = image.mean(dim=1, keepdim=True)
    else:
        img = image

    dx_disp = gradient_x(disp)
    dy_disp = gradient_y(disp)
    dx_img  = gradient_x(img).abs()
    dy_img  = gradient_y(img).abs()

    wx = torch.exp(-alpha * dx_img)
    wy = torch.exp(-alpha * dy_img)

    # charbonnier 比 L1 更平滑
    smooth = (wx * charbonnier(dx_disp)).mean() + (wy * charbonnier(dy_disp)).mean()
    return smooth

def sequence_loss_smoothness(
    flow_preds, flow_gt, valid,
    loss_gamma=0.9, max_flow=700,

    # ---- 新增：两类正则的权重 ----
    lambda_step=0.01,      # 建议 0.01 起
    lambda_osc=0.05,       # 建议 0.05 起
    lambda_smooth=0.01,    # 建议 0.001~0.01（需要 image 才能用）
    image=None,           # 传入左图 [B,3,H,W] 才启用 edge-aware smoothness
):
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    # gamma 调整保持你原逻辑
    adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))

    prev_delta = None
    for i in range(n_predictions):
        pred = flow_preds[i]
        assert not torch.isnan(pred).any() and not torch.isinf(pred).any()

        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (pred - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, pred.shape]
        flow_loss = flow_loss + i_weight * i_loss[valid.bool()].mean()

        # --- (1) 步长正则：抑制每步更新过大（不需要额外输出） ---
        if lambda_step > 0.0:
            if i == 0:
                delta = pred - pred.detach()  # 0（第一步没有上一步；你也可以跳过 i==0）
            else:
                delta = pred - flow_preds[i-1].detach()  # 用 detach 防止“互相拉扯”
            # 只在 valid 上算，避免无效区影响
            flow_loss = flow_loss + lambda_step * delta.abs()[valid.bool()].mean()

            # --- (2) 振荡正则：抑制来回震荡 Δt-Δt-1 ---
            if lambda_osc > 0.0 and prev_delta is not None:
                osc = (delta - prev_delta).abs()
                flow_loss = flow_loss + lambda_osc * osc[valid.bool()].mean()
            prev_delta = delta

        # --- (3) 边缘感知平滑：需要 image ---
        if lambda_smooth > 0.0 and image is not None:
            # pred 可能是 flow [B,2,H,W]，你若是 disparity 就是 [B,1,H,W]
            # stereo disparity 情况：只取 x 分量或直接 pred
            if pred.shape[1] == 2:
                disp = pred[:, :1]  # 只用 x 分量当作 disparity
            else:
                disp = pred
            flow_loss = flow_loss + (lambda_smooth * i_weight) * edge_aware_smoothness(disp, image)

    # metrics（保持不动）
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics

def sequence_loss_withconf(flow_preds, conf_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0
    B, _, H, W = flow_gt.shape
    #print("asd", flow_gt.shape)
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()
    adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))
    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        
        
        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (flow_preds[i] - flow_gt).abs()
        # with torch.no_grad():
        #     conf_gt = (i_loss <= 2).float()

        disp_loss = i_loss 
        #loss_conf_map = F.binary_cross_entropy_with_logits(logit_preds[i], conf_gt, reduction='none')
        w = (1 - conf_preds[i]).detach()
        w = w / (w.mean(dim=[2,3], keepdim=True) + 1e-6)
        w = w.clamp(0.25, 4.0)

        final_loss = disp_loss * w
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += (i_weight * final_loss)[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def fetch_optimizer(args, model, last_epoch=-1, checkpoint=None):
    """ Create the optimizer and learning rate scheduler """
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100, pct_start=0.01,
                                              cycle_momentum=False, anneal_strategy='linear', last_epoch=last_epoch)

    return optimizer, scheduler

def fetch_optimizer2(args, model):

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.AdamW(
        trainable_params,
        lr=1e-6,
        weight_decay=args.wdecay,
        eps=1e-8
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=150000
    )

    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, name):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.log_dir = 'runs/' + name
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar("train/" + k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for key in results:
            self.writer.add_scalar("valid/" + key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

