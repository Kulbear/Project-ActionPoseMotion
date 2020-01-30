import torch
import torch.nn as nn
from core.utils import AverageMeter
from progress.bar import Bar


def evaluate(data_loader, model, device):
    torch.set_grad_enabled(False)
    model.eval()
    right = 0
    total = 0
    for i, data in enumerate(data_loader):
        pose_3d = data['pose_3d']
        motion_gt = data['future_pose_3d']
        action_num = data['action_type'].to(device)
        # third_tensor = torch.cat((pose_3d, motion_gt), dim=1).transpose(1, 3).to(device)
        third_tensor = torch.cat((pose_3d, motion_gt), dim=1).permute(0, 3, 1, 2).unsqueeze(-1).to(device)

        pred = model(third_tensor)
        pred = torch.argmax(pred, dim=1)
        a = torch.eq(pred, action_num)
        right += sum(a).item()
        total += a.size(0)
    return right / total


def train(data_loader, model, criterion, optimizer, device, max_norm=True):
    # Switch to train mode
    mloss = AverageMeter()

    torch.set_grad_enabled(True)
    model.train()
    bar = Bar('Train', max=len(data_loader))
    for i, data in enumerate(data_loader):
        pose_3d = data['pose_3d']
        motion_gt = data['future_pose_3d']
        action_num = data['action_type']
        action_vector = torch.eye(60)[action_num].to(device)
        third_tensor = torch.cat((pose_3d, motion_gt), dim=1).permute(0, 3, 1, 2).unsqueeze(-1).to(device)

        pred = model(third_tensor)
        optimizer.zero_grad()
        loss = criterion(pred, action_vector)
        mloss.update(loss.item(), pose_3d.size()[0])

        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        bar.suffix = f'({i + 1}/{len(data_loader)}) ' \
                     f'| Total: {bar.elapsed_td:} | ETA: {bar.eta_td:}'
        bar.next()

    bar.finish()
    return mloss.avg
