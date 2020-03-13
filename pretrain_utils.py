from core.utils import AverageMeter

from core.metrics import mpjpe, p_mpjpe
from progress.bar import Bar

import torch
import torch.nn as nn
import time


def adjust_learning_rate(optimizer, factor=0.1, hardcode=0.):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * factor if hardcode == 0 else hardcode
    return param_group['lr']


# def pre_train(train_data_loader, valid_data_loader, loss_function, optimizer,
#               pre_refiner, pose_lifter, motion_generator, global_refiner,
#               train_pre_refiner=1, train_pose_lifter=1, train_motion_generator=1, train_global_refiner=1,
#               device='cuda', use_lie_algebra=False):

def train_pre_refiner(train_data_loader, valid_data_loader,
                      loss_function, optimizer, pre_refiner, device='cuda'):
    # Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    pre_refiner_loss = AverageMeter()
    mpjpe_pre_refiner = AverageMeter()
    mpjpe_nothing = AverageMeter()
    v_mpjpe_pre_refiner = AverageMeter()
    v_mpjpe_nothing = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    pre_refiner.train()

    end = time.time()

    # Train
    bar = Bar('Train', max=len(train_data_loader))
    pre_refiner.train()
    for i, data in enumerate(train_data_loader):
        optimizer.zero_grad()

        pose_2d = data['pose_2d']
        pose_2d_gt = data['pose_2d_gt']
        batch, seq_len = pose_2d.size()[:2]
        pose_2d = pose_2d.to(device).view(batch, seq_len, -1)
        pose_2d_gt = pose_2d_gt.to(device).view(batch, seq_len, -1)
        opt_dim = pre_refiner.opt_dim

        pose_2d_pred = pre_refiner(pose_2d)['refined_2d']
        loss = loss_function(pose_2d_pred.view(batch * seq_len, opt_dim),
                             pose_2d_gt.view(batch * seq_len, opt_dim))
        loss.backward()
        nn.utils.clip_grad_norm_(pre_refiner.parameters(), max_norm=2)
        optimizer.step()
        pre_refiner_loss.update(loss.cpu().item(), batch * seq_len)

        mpjpe_pre_refiner.update(mpjpe(pose_2d_pred, pose_2d_gt).item() * 1000.0, batch * seq_len)
        mpjpe_nothing.update(mpjpe(pose_2d, pose_2d_gt).item() * 1000.0, batch * seq_len)
        # Update time meters
        batch_time.update(time.time() - end)
        end = time.time()

        # Progress bar text
        bar.suffix = f'({i + 1}/{len(train_data_loader)}) D.: {data_time.val:.4f}s | B.: {batch_time.avg:.3f}s ' \
                     f'| Ttl: {bar.elapsed_td:} | ETA: {bar.eta_td:} | LPos: {pre_refiner_loss.avg: .5f} ' \
                     f'| Refined MPJPE: {mpjpe_pre_refiner.avg: .3f} | Trivial MPJPE: {mpjpe_nothing.avg: .3f}'
        bar.next()
    bar.finish()

    # Eval
    with torch.no_grad():
        bar = Bar('Valid', max=len(valid_data_loader))
        pre_refiner.eval()
        for i, data in enumerate(valid_data_loader):
            pose_2d = data['pose_2d']
            pose_2d_gt = data['pose_2d_gt']
            batch, seq_len = pose_2d.size()[:2]
            pose_2d = pose_2d.to(device).view(batch, seq_len, -1)
            pose_2d_gt = pose_2d_gt.to(device).view(batch, seq_len, -1)
            pose_2d_pred = pre_refiner(pose_2d)['refined_2d']

            v_mpjpe_pre_refiner.update(mpjpe(pose_2d_pred, pose_2d_gt).item() * 1000.0, batch * seq_len)
            v_mpjpe_nothing.update(mpjpe(pose_2d, pose_2d_gt).item() * 1000.0, batch * seq_len)
            # Update time meters

            # Progress bar text
            bar.suffix = f'({i + 1}/{len(valid_data_loader)}) D.: {data_time.val:.4f}s ' \
                         f'| Ttl: {bar.elapsed_td:} | ETA: {bar.eta_td:} ' \
                         f'| Refined MPJPE: {v_mpjpe_pre_refiner.avg: .3f} | Trivial MPJPE: {v_mpjpe_nothing.avg: .3f}'
            bar.next()

        bar.finish()


def train_pose_lifter(train_data_loader, valid_data_loader,
                      loss_function, optimizer, pose_lifter, device='cuda', use_lie_algebra=False):
    # Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    pose_coord_loss = AverageMeter()
    pose_lie_loss = AverageMeter()

    v_mpjpe_pose_lifter = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    pose_lifter.train()

    end = time.time()

    # Train
    bar = Bar('Train', max=len(train_data_loader))
    pose_lifter.train()
    for i, data in enumerate(train_data_loader):
        optimizer.zero_grad()

        pose_2d_gt = data['pose_2d_gt']
        pose_3d_gt = data['pose_3d']
        pose_lie_gt = data['pose_lie']
        batch, seq_len = pose_2d_gt.size()[:2]
        pose_2d_gt = pose_2d_gt.to(device).view(batch, seq_len, -1)
        # pose_3d_gt = pose_3d_gt.to(device).view(batch, seq_len, -1)
        # pose_lie_gt = pose_lie_gt.to(device).view(batch, seq_len, -1)

        pose_3d_gt = pose_3d_gt[:, :, 1:, :].to(device).view(batch, seq_len, -1)
        pose_lie_gt = pose_lie_gt[:, :, 1:, :].to(device).view(batch, seq_len, -1)

        pred = pose_lifter(pose_2d_gt)
        pose_3d_pred = pred['pose_3d']
        pose_lie_pred = pred.get('pose_lie', None)
        loss = loss_function(pose_3d_pred.view(batch * seq_len, 45),
                             pose_3d_gt.view(batch * seq_len, 45))
        pose_coord_loss.update(loss.cpu().item(), batch * seq_len)
        if use_lie_algebra:
            lie_loss = loss_function(pose_lie_pred.view(batch * seq_len, 90),
                                     pose_lie_gt.view(batch * seq_len, 90))
            pose_lie_loss.update(lie_loss.cpu().item(), batch * seq_len)
            loss += lie_loss

        loss.backward()
        nn.utils.clip_grad_norm_(pose_lifter.parameters(), max_norm=2)
        optimizer.step()

        # Update time meters
        batch_time.update(time.time() - end)
        end = time.time()
        # Progress bar text
        bar.suffix = f'({i + 1}/{len(train_data_loader)}) D.: {data_time.val:.4f}s | B.: {batch_time.avg:.3f}s ' \
                     f'| Ttl: {bar.elapsed_td:} | ETA: {bar.eta_td:} | CoordMSE: {pose_coord_loss.avg: .5f} '
        if use_lie_algebra:
            bar.suffix += f'| LieMSE: {pose_lie_loss.avg: .3f} '
        bar.next()
    bar.finish()

    # Eval
    with torch.no_grad():
        bar = Bar('Valid', max=len(valid_data_loader))
        pose_lifter.eval()
        for i, data in enumerate(valid_data_loader):
            pose_2d_gt = data['pose_2d_gt']
            pose_3d_gt = data['pose_3d']
            pose_lie_gt = data['pose_lie']
            batch, seq_len = pose_2d_gt.size()[:2]
            pose_2d_gt = pose_2d_gt.to(device).view(batch, seq_len, -1)
            opt_dim = pose_lifter.opt_dim

            pred = pose_lifter(pose_2d_gt)
            pose_3d_pred = pred['pose_3d'].cpu()
            pose_lie_pred = pred.get('pose_lie', None)

            if pose_lie_pred is not None:
                pose_lie_pred = pose_lie_pred.cpu()

            pose_3d_pred = pose_3d_pred.reshape(batch * seq_len, -1, 3).cpu()
            pose_3d_pred = torch.cat([torch.zeros(batch * seq_len, 1, pose_3d_pred.size(2)), pose_3d_pred], 1)
            pose_3d_gt = pose_3d_gt.view(batch * seq_len, -1, 3)

            v_mpjpe_pose_lifter.update(mpjpe(pose_3d_pred, pose_3d_gt).item() * 1000.0, batch * seq_len)

            # Update time meters
            batch_time.update(time.time() - end)
            end = time.time()
            # Progress bar text
            bar.suffix = f'({i + 1}/{len(valid_data_loader)}) D.: {data_time.val:.4f}s | B.: {batch_time.avg:.3f}s ' \
                         f'| Ttl: {bar.elapsed_td:} | ETA: {bar.eta_td:} ' \
                         f'| Lifter MPJPE: {v_mpjpe_pose_lifter.avg: .3f} |'
            bar.next()
        bar.finish()


def train_motion_generator(train_data_loader, valid_data_loader,
                           loss_function, optimizer, motion_generator, device='cuda', use_lie_algebra=False):
    # Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    motion_coord_loss = AverageMeter()
    motion_lie_loss = AverageMeter()

    v_mpjpe_motion_generator = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    motion_generator.train()

    end = time.time()

    # Train
    bar = Bar('Train', max=len(train_data_loader))
    motion_generator.train()
    for i, data in enumerate(train_data_loader):
        optimizer.zero_grad()

        pose_3d_gt = data['pose_3d']
        pose_lie_gt = data['pose_lie']
        motion_3d_gt = data['future_pose_3d']
        motion_lie_gt = data['future_pose_lie']
        batch, seq_len = pose_3d_gt.size()[:2]
        future_seq_len = motion_3d_gt.size()[1]

        pose_3d_gt = pose_3d_gt[:, :, 1:, :].to(device).view(batch, seq_len, -1)
        pose_lie_gt = pose_lie_gt[:, :, 1:, :].to(device).view(batch, seq_len, -1)
        motion_3d_gt = motion_3d_gt[:, :, 1:, :].to(device).view(batch, future_seq_len, -1)
        motion_lie_gt = motion_lie_gt[:, :, 1:, :].to(device).view(batch, future_seq_len, -1)

        opt_dim = 45
        outputs = torch.zeros(batch, future_seq_len, opt_dim).to(device)
        outputs_lie = torch.zeros(batch, future_seq_len, opt_dim * 2).to(device)

        output = pose_3d_gt[:, -1, :].unsqueeze(1)
        if use_lie_algebra:
            output_lie = pose_lie_gt[:, -1, :].unsqueeze(1)
            output = (output, output_lie)

        hidden = None
        for t in range(future_seq_len):
            prediction = motion_generator(output, hidden)
            outputs[:, t, :] = prediction['motion_3d'].squeeze()
            if use_lie_algebra:
                outputs_lie[:, t, :] = prediction['motion_lie'].squeeze()
            hidden = prediction['decoder_hidden']

            # no teacher forcing, only use ground truth
            # output = motion_3d_gt[:, t, :].unsqueeze(1)
            # if use_lie_algebra:
            #     output_lie = motion_lie_gt[:, t, :].unsqueeze(1)
            #     output = (output, output_lie)

            output = outputs[:, t, :].unsqueeze(1)
            if use_lie_algebra:
                output_lie = outputs_lie[:, t, :].unsqueeze(1)
                output = (output, output_lie)

        loss = loss_function(outputs.view(batch * future_seq_len, opt_dim),
                             motion_3d_gt.view(batch * future_seq_len, opt_dim))
        motion_coord_loss.update(loss.cpu().item(), batch * future_seq_len)
        if use_lie_algebra:
            lie_loss = loss_function(outputs_lie.view(batch * future_seq_len, opt_dim * 2),
                                     motion_lie_gt.view(batch * future_seq_len, opt_dim * 2))
            motion_lie_loss.update(lie_loss.cpu().item(), batch * future_seq_len)
            loss += lie_loss

        loss.backward()
        nn.utils.clip_grad_norm_(motion_generator.parameters(), max_norm=2)
        optimizer.step()

        # Update time meters
        batch_time.update(time.time() - end)
        end = time.time()
        # Progress bar text
        bar.suffix = f'({i + 1}/{len(train_data_loader)}) D.: {data_time.val:.4f}s | B.: {batch_time.avg:.3f}s ' \
                     f'| Ttl: {bar.elapsed_td:} | ETA: {bar.eta_td:} | CoordMSE: {motion_coord_loss.avg: .5f} '
        if use_lie_algebra:
            bar.suffix += f'| LieMSE: {motion_lie_loss.avg: .3f} '
        bar.next()
    bar.finish()

    # Eval
    with torch.no_grad():
        bar = Bar('Valid', max=len(valid_data_loader))
        motion_generator.eval()
        for i, data in enumerate(valid_data_loader):
            pose_3d_gt = data['pose_3d']
            pose_lie_gt = data['pose_lie']
            motion_3d_gt = data['future_pose_3d']
            motion_lie_gt = data['future_pose_lie']
            batch, seq_len = pose_3d_gt.size()[:2]
            future_seq_len = motion_3d_gt.size()[1]

            pose_3d_gt = pose_3d_gt[:, :, 1:, :].to(device).view(batch, seq_len, -1)
            pose_lie_gt = pose_lie_gt[:, :, 1:, :].to(device).view(batch, seq_len, -1)
            motion_3d_gt = motion_3d_gt[:, :, 1:, :].to(device).view(batch, future_seq_len, -1)
            motion_lie_gt = motion_lie_gt[:, :, 1:, :].to(device).view(batch, future_seq_len, -1)

            outputs = torch.zeros(batch, future_seq_len, opt_dim).to(device)
            outputs_lie = torch.zeros(batch, future_seq_len, opt_dim * 2).to(device)

            output = pose_3d_gt[:, -1, :].unsqueeze(1)
            if use_lie_algebra:
                output_lie = pose_lie_gt[:, -1, :].unsqueeze(1)
                output = (output, output_lie)

            hidden = None
            for t in range(future_seq_len):
                prediction = motion_generator(output, hidden)
                outputs[:, t, :] = prediction['motion_3d'].squeeze()
                if use_lie_algebra:
                    outputs_lie[:, t, :] = prediction['motion_lie'].squeeze()
                hidden = prediction['decoder_hidden']

                # no teacher forcing, only use ground truth
                output = motion_3d_gt[:, t, :].unsqueeze(1)
                if use_lie_algebra:
                    output_lie = motion_lie_gt[:, t, :].unsqueeze(1)
                    output = (output, output_lie)

            motion_3d_pred = outputs.reshape(batch * future_seq_len, -1, 3).cpu()
            motion_3d_pred = torch.cat([torch.zeros(batch * future_seq_len, 1, motion_3d_pred.size(2)),
                                        motion_3d_pred], 1)
            motion_3d_gt = motion_3d_gt.view(batch * future_seq_len, -1, 3).cpu()
            motion_3d_gt = torch.cat([torch.zeros(batch * future_seq_len, 1, motion_3d_gt.size(2)), motion_3d_gt], 1)

            # print(motion_3d_pred.size(), motion_3d_gt.size())
            v_mpjpe_motion_generator.update(mpjpe(motion_3d_pred, motion_3d_gt).item() * 1000.0, batch * future_seq_len)

            # Update time meters
            batch_time.update(time.time() - end)
            end = time.time()
            # Progress bar text
            bar.suffix = f'({i + 1}/{len(valid_data_loader)}) D.: {data_time.val:.4f}s | B.: {batch_time.avg:.3f}s ' \
                         f'| Ttl: {bar.elapsed_td:} | ETA: {bar.eta_td:} ' \
                         f'| Generator MPJPE: {v_mpjpe_motion_generator.avg: .3f} |'
            bar.next()
        bar.finish()
