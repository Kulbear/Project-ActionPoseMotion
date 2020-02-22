import os
import torch
import numpy as np
from pathlib import Path

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_ckpt(state, ckpt_path, best_error=None, suffix=None):
    """
    Example usage:
    save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step,
               'pos2mot_model': pos2mot_model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'refine_model': refine_model.state_dict(),
               'error_best_pose': errors[0], error_best_motion}, ckpt_dir_path, suffix='pose_best')
    """
    if suffix is None:
        suffix = 'epoch_{:04d}'.format(state['epoch'])

    file_path = os.path.join(ckpt_path, 'ckpt_{}.pth.tar'.format(suffix))
    torch.save(state, file_path)


def load_ckpt(ckpt_dir_path, ckpt_name):
    """
    Example return value:
    {'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step,
     'pos2mot_model': pos2mot_model.state_dict(),
     'optimizer': optimizer.state_dict(),
     'refine_model': refine_model.state_dict(),
     'error_best_pose': errors[0], error_best_motion}
    """
    state = torch.load(Path(ckpt_dir_path, ckpt_name, '.pth.tar'))
    suffix = ckpt_name.split('_')[1:]
    return state, suffix


def wrap(func, unsqueeze, *args):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
