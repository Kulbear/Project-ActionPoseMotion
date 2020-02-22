import time
import torch
import torch.nn as nn
import numpy as np

from progress.bar import Bar
from core.metrics import mpjpe, p_mpjpe
from core.utils import AverageMeter, lr_decay


def calculate_refinement_stage_loss(refined_prediction, pose_3d, motion_gt, loss_function,
                                    batch_size, pose_seq_len, future_seq_len, opt_dim):
    refined_pose_3d = refined_prediction[:, :pose_seq_len, :]
    refined_motion_3d = refined_prediction[:, pose_seq_len:, :]
    loss_3d_pose = loss_function(refined_pose_3d.reshape(batch_size * pose_seq_len, opt_dim),
                                 pose_3d.view(batch_size * pose_seq_len, opt_dim))
    loss_3d_motion = loss_function(refined_motion_3d.reshape(batch_size * future_seq_len, opt_dim),
                                   motion_gt.view(batch_size * future_seq_len, opt_dim))
    return loss_3d_pose, loss_3d_motion


def train(data_loader, pos2mot_model, criterion, optimizer, device, lr_init, lr_now,
          step, decay, gamma, max_norm=True, refine_model=None, refine_iteration=1):
    # Whether do refinement?
    with_refinement = refine_model is not None and refine_iteration

    # Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mloss_3d_pose = AverageMeter()
    mloss_3d_motion = AverageMeter()

    # Used only when with_refinement=True
    rloss_3d_pose = AverageMeter()
    rloss_3d_motion = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    pos2mot_model.train()
    if with_refinement:
        refine_model.train()
    end = time.time()

    # Epoch starts
    bar = Bar('Train', max=len(data_loader))
    for i, data in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Adjust learning rate if decay condition is satisfied
        step += 1
        if step % decay == 0:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        # Parse data
        pose_2d = data['pose_2d']
        pose_3d = data['pose_3d']
        motion_gt = data['future_pose_3d']
        batch, seq_len = pose_2d.size()[:2]
        _, future_seq_len = motion_gt.size()[:2]
        opt_dim = pos2mot_model.encoder.opt_dim

        # Reshaping input for 3d poses and motion and ignoring the hip joint
        pose_2d = pose_2d.to(device).view(batch, seq_len, -1)
        pose_3d = pose_3d[:, :, 1:, :].to(device).view(batch, seq_len, -1)
        motion_gt = motion_gt[:, :, 1:, :].to(device).view(batch, future_seq_len, -1)

        # Forward pass
        pred = pos2mot_model(pose_2d, motion_gt)
        pred_pose_3d = pred['past_pose']
        pred_motion_3d = pred['future_motion']

        # Multi-stage refinement module
        refined_loss_3d_pose, refined_loss_3d_motion = 0, 0
        if with_refinement:
            refined_pred = torch.cat((pred_pose_3d, pred_motion_3d), 1)
            for _ in range(refine_iteration):
                refined_pred = refine_model(refined_pred)['refined_3d']
                # Multi-stage loss is calculated and summed up here
                rloss, mloss = calculate_refinement_stage_loss(refined_pred, pose_3d, motion_gt, criterion,
                                                               batch, seq_len, future_seq_len, opt_dim)
                refined_loss_3d_pose += rloss
                refined_loss_3d_motion += mloss

        # Back-propagation
        optimizer.zero_grad()
        loss_3d_pose = criterion(pred_pose_3d.view(batch * seq_len, opt_dim), pose_3d.view(batch * seq_len, opt_dim))
        loss_3d_motion = criterion(pred_motion_3d.view(batch * future_seq_len, opt_dim),
                                   motion_gt.view(batch * future_seq_len, opt_dim))
        if with_refinement:
            loss_3d = loss_3d_pose + loss_3d_motion + refined_loss_3d_pose + refined_loss_3d_motion
        else:
            loss_3d = loss_3d_pose + loss_3d_motion
        loss_3d.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(pos2mot_model.parameters(), max_norm=1)
        optimizer.step()

        # Update meters with computed losses
        mloss_3d_pose.update(loss_3d_pose.item(), batch * seq_len)
        mloss_3d_motion.update(loss_3d_motion.item(), batch * future_seq_len)
        if with_refinement:
            rloss_3d_pose.update(refined_loss_3d_pose.item(), batch * seq_len)
            rloss_3d_motion.update(refined_loss_3d_motion.item(), batch * future_seq_len)
        batch_time.update(time.time() - end)
        end = time.time()

        # Progress bar text
        bar.suffix = f'({i + 1}/{len(data_loader)}) D.: {data_time.val:.4f}s | B.: {batch_time.avg:.3f}s ' \
                     f'| Ttl: {bar.elapsed_td:} | ETA: {bar.eta_td:}' \
                     f'| LPos: {mloss_3d_pose.avg: .5f} | LMot: {mloss_3d_motion.avg: .6f}'
        if with_refinement:
            bar.suffix += f'| rLPos: {rloss_3d_pose.avg: .5f} | rLMot: {rloss_3d_motion.avg: .6f}'
        bar.next()

    bar.finish()

    if with_refinement:  # TODO: should we just return these two?
        return [rloss_3d_pose.avg, rloss_3d_motion.avg], lr_now, step
    else:
        return [mloss_3d_pose.avg, mloss_3d_motion.avg], lr_now, step


def evaluate(data_loader, pos2mot_model, device, inference_mode=False, refine_model=None, refine_iteration=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mpjpe_pose = AverageMeter()
    mpjpe_pose_raligned = AverageMeter()
    mpjpe_motion = AverageMeter()
    mpjpe_motion_raligned = AverageMeter()

    with_refinement = refine_model is not None and refine_iteration

    if inference_mode:
        keypoint_sequence = []
        pose_sequence = []
        motion_sequence = []
        pose_sequence_gt = []
        motion_sequence_gt = []

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    pos2mot_model.eval()
    if with_refinement:
        refine_model.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))
    for i, data in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Parse data
        pose_2d = data['pose_2d']
        pose_3d = data['pose_3d']
        motion_gt = data['future_pose_3d']
        batch, seq_len = pose_2d.size()[:2]
        _, future_seq_len = motion_gt.size()[:2]

        if inference_mode:
            keypoint_sequence.append(pose_2d.numpy())

        # Inference: pose to motion
        pose_2d = pose_2d.to(device).view(batch, seq_len, -1)
        pred = pos2mot_model(pose_2d, motion_gt[:, :, 1:, :].view(batch, future_seq_len, -1).to(device),
                             teacher_forcing_ratio=0.)

        # Inference: refinement
        refined_pred = torch.cat((pred['past_pose'], pred['future_motion']), 1)
        if with_refinement:
            for _ in range(refine_iteration):
                refined_pred = refine_model(refined_pred)['refined_3d']

        refined_pred_pose_3d = refined_pred[:, :seq_len, :]
        refined_pred_motion_3d = refined_pred[:, seq_len:, :]

        # Evaluation
        pred_pose_3d = refined_pred_pose_3d.reshape(batch * seq_len, -1, 3).cpu()
        pose_3d = pose_3d.view(batch * seq_len, -1, 3)

        pred_motion_3d = refined_pred_motion_3d.reshape(batch * future_seq_len, -1, 3).cpu()
        motion_gt = motion_gt.view(batch * future_seq_len, -1, 3)

        pred_pose_3d = torch.cat([torch.zeros(batch * seq_len, 1, pred_pose_3d.size(2)), pred_pose_3d], 1)
        pred_motion_3d = torch.cat([torch.zeros(batch * future_seq_len, 1, pred_motion_3d.size(2)), pred_motion_3d], 1)

        if inference_mode:
            pose_sequence.append(pred_pose_3d.cpu().numpy())
            motion_sequence.append(pred_motion_3d.cpu().numpy())
            pose_sequence_gt.append(pose_3d.cpu().numpy())
            motion_sequence_gt.append(motion_gt.cpu().numpy())

        # Update meters with computed metric values
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

    # Return predicted result or measured metric values
    if inference_mode:
        return np.vstack(np.array(pose_sequence)), np.vstack(pose_sequence_gt), \
               np.vstack(np.array(motion_sequence)), np.vstack(motion_sequence_gt), np.vstack(keypoint_sequence)
    else:
        return mpjpe_pose.avg, mpjpe_pose_raligned.avg, mpjpe_motion.avg, mpjpe_motion_raligned.avg
