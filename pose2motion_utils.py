import time
import torch
import torch.nn as nn
import numpy as np

from progress.bar import Bar
from core.metrics import mpjpe, p_mpjpe
from core.utils import AverageMeter, lr_decay


def calculate_refinement_stage_loss(refined_prediction, refined_gt, loss_function):
    batch_size, seq_len, out_dim = refined_prediction.size()
    loss = loss_function(refined_prediction.reshape(batch_size * seq_len, out_dim),
                         refined_gt.view(batch_size * seq_len, out_dim))
    return loss


def train(data_loader, pos2mot_model, criterion,
          optimizer, device, lr_init, lr_now,
          step, decay, gamma, max_norm=True,
          refine_model=None, refine_iteration=1,
          pos_loss_on=True, mot_loss_on=True, step_lr=True,
          include_lie_repr=False):
    # Whether do refinement?
    with_refinement = refine_model is not None and refine_iteration > 0

    # Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mloss_3d_pose = AverageMeter()
    mloss_3d_motion = AverageMeter()
    mloss_lie_pose = AverageMeter()
    mloss_lie_motion = AverageMeter()
    rloss_3d = AverageMeter()

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
        if step_lr and step % decay == 0:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        # Parse data
        pose_2d = data['pose_2d']
        pose_3d = data['pose_3d']
        pose_lie = data['pose_lie']
        motion_gt = data['future_pose_3d']
        motion_gt_lie = data['future_pose_lie']
        batch, seq_len = pose_2d.size()[:2]
        _, future_seq_len = motion_gt.size()[:2]
        opt_dim = pos2mot_model.encoder.opt_dim

        # Reshaping input for 3d poses and motion and ignoring the hip joint
        pose_2d = pose_2d.to(device).view(batch, seq_len, -1)
        pose_3d = pose_3d[:, :, 1:, :].to(device).view(batch, seq_len, -1)
        pose_lie = pose_lie[:, :, 1:, :].to(device).view(batch, seq_len, -1)
        motion_gt = motion_gt[:, :, 1:, :].to(device).view(batch, future_seq_len, -1)
        motion_gt_lie = motion_gt_lie[:, :, 1:, :].to(device).view(batch, future_seq_len, -1)

        # Forward pass
        if include_lie_repr:
            pred = pos2mot_model(pose_2d, motion_gt, motion_lie=motion_gt_lie,
                                 pos_loss_on=True, mot_loss_on=True)
            pred_pose_lie = pred['past_pose_lie']
            pred_motion_lie = pred['future_motion_lie']
        else:
            pred = pos2mot_model(pose_2d, motion_gt, pos_loss_on=True, mot_loss_on=True)
        pred_pose_3d = pred['past_pose']
        pred_motion_3d = pred['future_motion']

        # Multi-stage refinement module
        refined_loss = 0
        if with_refinement and include_lie_repr:
            traj = []
            traj_gt = []
            if pos_loss_on:
                traj.append(torch.cat((pred_pose_3d, pred_pose_lie), dim=2))
                traj_gt.append(torch.cat((pose_3d, pose_lie), dim=2))
            if mot_loss_on:
                traj.append(torch.cat((pred_motion_3d, pred_motion_lie), dim=2))
                traj_gt.append(torch.cat((motion_gt, motion_gt_lie), dim=2))
            if len(traj) == 1:
                refined_pred = traj[0]
                refined_gt = traj_gt[0]
            else:
                refined_pred = torch.cat(traj, 1)
                refined_gt = torch.cat(traj_gt, 1)
            for _ in range(refine_iteration):
                refined_pred = refine_model(refined_pred)['refined_3d']
                loss = calculate_refinement_stage_loss(refined_pred, refined_gt, criterion)
                refined_loss += loss

        elif with_refinement and not include_lie_repr:
            traj = []
            traj_gt = []
            if pos_loss_on:
                traj.append(pred_pose_3d)
                traj_gt.append(pose_3d)
            if mot_loss_on:
                traj.append(pred_motion_3d)
                traj_gt.append(motion_gt)
            if len(traj) == 1:
                refined_pred = traj[0]
                refined_gt = traj_gt[0]
            else:
                refined_pred = torch.cat(traj, 1)
                refined_gt = torch.cat(traj_gt, 1)
            for _ in range(refine_iteration):
                refined_pred = refine_model(refined_pred)['refined_3d']
                loss = calculate_refinement_stage_loss(refined_pred, refined_gt, criterion)
                refined_loss += loss

        # Back-propagation
        optimizer.zero_grad()

        loss_3d = 0
        total_seq_len = 0
        # Add pose loss
        if pos_loss_on:
            loss_3d_pose = criterion(pred_pose_3d.view(batch * seq_len, opt_dim),
                                     pose_3d.view(batch * seq_len, opt_dim))
            mloss_3d_pose.update(loss_3d_pose.item(), batch * seq_len)
            loss_3d += loss_3d_pose
            if include_lie_repr:
                loss_lie_pose = criterion(pred_pose_lie.view(batch * seq_len, opt_dim * 3),
                                          pose_lie.view(batch * seq_len, opt_dim * 3))
                mloss_lie_pose.update(loss_lie_pose.item(), batch * seq_len)
                loss_3d += loss_lie_pose
            total_seq_len += seq_len

        # Add motion loss
        if mot_loss_on:
            loss_3d_motion = criterion(pred_motion_3d.view(batch * future_seq_len, opt_dim),
                                       motion_gt.view(batch * future_seq_len, opt_dim))
            loss_3d += loss_3d_motion
            mloss_3d_motion.update(loss_3d_motion.item(), batch * future_seq_len)
            if include_lie_repr:
                loss_lie_motion = criterion(pred_motion_lie.view(batch * future_seq_len, opt_dim * 3),
                                          motion_gt_lie.view(batch * future_seq_len, opt_dim * 3))
                mloss_lie_motion.update(loss_lie_motion.item(), batch * future_seq_len)
                loss_3d += loss_lie_motion
            total_seq_len += future_seq_len

        # Add refined loss
        if with_refinement:
            rloss_3d.update(refined_loss.item(), batch * total_seq_len)
            loss_3d += refined_loss

        loss_3d.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(pos2mot_model.parameters(), max_norm=1)
        optimizer.step()

        # Update time meters
        batch_time.update(time.time() - end)
        end = time.time()

        # Progress bar text
        bar.suffix = f'({i + 1}/{len(data_loader)}) D.: {data_time.val:.4f}s | B.: {batch_time.avg:.3f}s ' \
                     f'| Ttl: {bar.elapsed_td:} | ETA: {bar.eta_td:}'

        if pos_loss_on:
            bar.suffix += f'| LPos: {mloss_3d_pose.avg: .5f} '
        if mot_loss_on:
            bar.suffix += f'| LMot: {mloss_3d_motion.avg: .6f}'
        if with_refinement:
            bar.suffix += f'| rLPos: {rloss_3d.avg: .5f} |'
        bar.next()

    bar.finish()

    return [mloss_3d_pose.avg, mloss_3d_motion.avg, rloss_3d.avg], lr_now, step


def evaluate(data_loader, pos2mot_model, device, inference_mode=False, refine_model=None, refine_iteration=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mpjpe_pose = AverageMeter()
    mpjpe_pose_raligned = AverageMeter()
    mpjpe_motion = AverageMeter()
    mpjpe_motion_raligned = AverageMeter()

    with_refinement = refine_model is not None and refine_iteration > 0

    if inference_mode:
        keypoint_sequence = []
        pose_sequence = []
        motion_sequence = []
        pose_sequence_gt = []
        motion_sequence_gt = []

    # Switch to evaluation mode
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
