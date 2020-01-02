import time
import torch
import torch.nn as nn

from progress.bar import Bar
from core.metrics import mpjpe, p_mpjpe
from core.utils import AverageMeter, lr_decay


def train(data_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mloss_3d_pose = AverageMeter()
    mloss_3d_motion = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    for i, data in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        step += 1
        if step % decay == 0:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        pose_2d = data['pose_2d']
        pose_3d = data['pose_3d']
        motion_gt = data['future_pose_3d']
        batch, seq_len = pose_2d.size()[:2]
        _, future_seq_len = motion_gt.size()[:2]
        opt_dim = model_pos.encoder.opt_dim
        # reshape, for 3d poses and motion, ignore the hip joint
        pose_2d = pose_2d.to(device).view(batch, seq_len, -1)
        pose_3d = pose_3d[:, :, 1:, :].to(device).view(batch, seq_len, -1)
        motion_gt = motion_gt[:, :, 1:, :].to(device).view(batch, future_seq_len, -1)
        pred = model_pos(pose_2d, motion_gt)

        pred_pose_3d = pred['past_pose']
        pred_motion_3d = pred['future_motion']

        optimizer.zero_grad()
        loss_3d_pose = criterion(pred_pose_3d.view(batch * seq_len, opt_dim), pose_3d.view(batch * seq_len, opt_dim))
        loss_3d_motion = criterion(pred_motion_3d.view(batch * future_seq_len, opt_dim),
                                   motion_gt.view(batch * future_seq_len, opt_dim))
        loss_3d = loss_3d_pose + loss_3d_motion
        loss_3d.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        mloss_3d_pose.update(loss_3d_pose.item(), batch * seq_len)
        mloss_3d_motion.update(loss_3d_motion.item(), batch * future_seq_len)
        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = f'({i + 1}/{len(data_loader)}) Data: {data_time.val:.5f}s | Batch: {batch_time.avg:.3f}s ' \
                     f'| Total: {bar.elapsed_td:} | ETA: {bar.eta_td:}' \
                     f'| Loss_Pose: {mloss_3d_pose.avg: .6f} | Loss_Motion: {mloss_3d_motion.avg: .6f}'
        bar.next()

    bar.finish()
    return mloss_3d_pose.avg, mloss_3d_motion.avg, lr_now, step


def evaluate(data_loader, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mpjpe_pose = AverageMeter()
    mpjpe_pose_raligned = AverageMeter()
    mpjpe_motion = AverageMeter()
    mpjpe_motion_raligned = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))
    for i, data in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        pose_2d = data['pose_2d']
        pose_3d = data['pose_3d']
        motion_gt = data['future_pose_3d']
        batch, seq_len = pose_2d.size()[:2]
        _, future_seq_len = motion_gt.size()[:2]

        pose_2d = pose_2d.to(device).view(batch, seq_len, -1)
        pred = model_pos(pose_2d, motion_gt[:, :, 1:, :].view(batch, future_seq_len, -1).to(device),
                         teacher_forcing_ratio=0.)

        pred_pose_3d = pred['past_pose'].view(batch * seq_len, -1, 3).cpu()
        pose_3d = pose_3d.view(batch * seq_len, -1, 3)
        motion_gt = motion_gt.view(batch * future_seq_len, -1, 3)
        pred_motion_3d = pred['future_motion'].view(batch * future_seq_len, -1, 3).cpu()
        pred_pose_3d = torch.cat([torch.zeros(batch * seq_len, 1, pred_pose_3d.size(2)), pred_pose_3d], 1)
        pred_motion_3d = torch.cat([torch.zeros(batch * future_seq_len, 1, pred_motion_3d.size(2)), pred_motion_3d], 1)

        mpjpe_pose.update(mpjpe(pred_pose_3d, pose_3d).item() * 1000.0, batch * seq_len)
        mpjpe_pose_raligned.update(p_mpjpe(pred_pose_3d.numpy(), pose_3d.numpy()).item() * 1000.0,
                                   batch * seq_len)
        mpjpe_motion.update(mpjpe(pred_motion_3d, motion_gt).item() * 1000.0, batch * future_seq_len)
        mpjpe_motion_raligned.update(p_mpjpe(pred_motion_3d.numpy(), motion_gt.numpy()).item() * 1000.0,
                                     batch * future_seq_len)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = f'({i + 1}/{len(data_loader)}) Data: {data_time.val:.5f}s | Batch: {batch_time.avg:.3f}s ' \
                     f'| Total: {bar.elapsed_td:}' \
                     f'| P MPJPE: {mpjpe_pose.avg: .4f} | P P-MPJPE: {mpjpe_pose_raligned.avg: .4f} ' \
                     f'| M MPJPE: {mpjpe_motion.avg: .4f} | M P-MPJPE: {mpjpe_motion_raligned.avg: .4f}'
        bar.next()

    bar.finish()
    return mpjpe_pose.avg, mpjpe_pose_raligned.avg, mpjpe_motion.avg, mpjpe_motion_raligned.avg
