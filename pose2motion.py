import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.log import Logger, save_fig, save_config
from core.data.generators import PoseGenerator
from core.models import (
    PoseLifter, MotionGenerator, Pose2MotNet,
    REFINEMENT_ARCHS
)
from core.utils import save_ckpt, load_ckpt
from core.data.data_utils import fetch, read_3d_data, create_2d_data

from pose2motion_arguments import parse_args
from pose2motion_utils import train, evaluate


def main(config):
    device = config.device
    print('==> Using settings {}'.format(config))

    # workaround for motion evaluation calculation
    # window_stride = config.future - config.past
    # assert config.past <= config.future, \
    #     'The current evaluation scheme for motion prediction requires config.past <= config.future'
    evaluate_motion = config.final and config.past != config.future

    if evaluate_motion:
        print('==> Evaluate motion!')
    ckpt_dir_path = Path('experiments', config.exp_name)
    print('==> Created checkpoint dir: {}'.format(ckpt_dir_path))
    if ckpt_dir_path.exists():
        print('==> Found existing checkpoint dir!')
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)

    logger = Logger(Path(ckpt_dir_path, 'log.txt'), resume=config.evaluation or config.ckpt_path)
    logger.set_names(['epoch', 'iter', 'lr', 'loss_train',
                      'Pose MPJPE', 'Pose P-MPJPE', 'Motion MPJPE', 'Motion P-MPJPE'])

    print('==> Loading dataset...', config.dataset)
    if config.dataset == 'h36m':
        DATASET_NAME = config.dataset.lower()
        from core.data.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS

        dataset_path = Path('data', DATASET_NAME, f'data_3d_{DATASET_NAME}.npz')
        dataset = Human36mDataset(dataset_path)
        subjects_train = TRAIN_SUBJECTS
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError('Invalid dataset')

    # the input stored in dataset are world coordinates
    # so we need to convert world coordinates to camera coordinates
    # and also remove the global offset (It's done inside `read_3d_data`)
    # Therefore it produces 4 groups of results, which are corresponding to 4 cameras
    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    # There are 4 groups of 2D keypoints (screen coordinates) for each of the subject
    # They are stored in the format keypoints[subject][action][cam_idx]
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    # results in screen (aka frame) coordinates
    print('==> Loading 2D detections...', end='\t')
    keypoints_path = Path('data', DATASET_NAME, f'data_2d_{DATASET_NAME}_{config.keypoint_source}.npz')
    keypoints = create_2d_data(keypoints_path, dataset)
    print(keypoints.keys())

    print('==> Initializing dataloaders...')
    action_filter = None if config.actions == '*' else config.actions.split(',')
    # pose_2d_past_segments, pose_3d_past_segments, pose_3d_future_segments, pose_actions = data
    data = fetch(subjects_train, dataset, keypoints,
                 past=config.past, future=config.future, action_filter=action_filter,
                 window_stride=config.window_stride, time_stride=config.time_stride, train=True)
    train_loader = DataLoader(PoseGenerator(*data),
                              batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    # pose evaluation
    data = fetch(subjects_test, dataset, keypoints,
                 past=config.past, future=config.future, action_filter=action_filter,
                 window_stride=config.past, time_stride=config.time_stride, train=False)
    valid_loader_pose = DataLoader(PoseGenerator(*data),
                                   batch_size=config.batch_size * 4, shuffle=False, num_workers=config.num_workers)

    if evaluate_motion:
        data = fetch(subjects_test, dataset, keypoints,
                     past=config.past, future=config.future, action_filter=action_filter,
                     window_stride=config.future, time_stride=config.time_stride, train=False)
        valid_loader_motion = DataLoader(PoseGenerator(*data),
                                         batch_size=config.batch_size * 4,
                                         shuffle=False, num_workers=config.num_workers)
    print('Done!')

    # save experiment config
    save_config(config, Path(ckpt_dir_path, 'sample_config.json'))

    encoder = PoseLifter(config.encoder_ipt_dim, config.encoder_opt_dim,
                         hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                         bidirectional=config.bidirectional, dropout_ratio=config.dropout)
    decoder = MotionGenerator(config.decoder_ipt_dim, config.decoder_opt_dim,
                              hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                              bidirectional=config.bidirectional, dropout_ratio=config.dropout)
    pos2mot_model = Pose2MotNet(encoder, decoder).to(device)

    RefineNet = REFINEMENT_ARCHS.get(config.refine_version, None)
    if not RefineNet:
        raise NotImplementedError('Unknown refinement architecture!')

    refine_model = RefineNet(config.decoder_opt_dim, config.decoder_opt_dim,
                             hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                             bidirectional=config.bidirectional, dropout_ratio=config.dropout,
                             size=(config.batch_size, config.past + config.future, 45)).to(device)
    criterion = nn.MSELoss().to(device)
    if refine_model is not None:
        optimizer = torch.optim.Adam(list(pos2mot_model.parameters()) + list(refine_model.parameters()), lr=config.lr)
    else:
        optimizer = torch.optim.Adam(pos2mot_model.parameters(), lr=config.lr)

    # Resume or start from scratch
    if config.ckpt_path is not None:
        print('[DEBUG]', Path(ckpt_dir_path, config.ckpt_path))
        assert os.path.isfile(Path(ckpt_dir_path, config.ckpt_path + '.pth.tar'))
        state, suffix = load_ckpt(ckpt_dir_path, config.ckpt_path)
        pos2mot_state = state.get('pos2mot_model')
        optim_state = state.get('optimizer')
        error_best_pose = state.get('error_best_pose')
        error_best_motion = state.get('error_best_motion')
        glob_step = state.get('step')
        start_epoch = state.get('epoch')
        if refine_model is not None:
            refine_state = state.get('refine_model')
            refine_model.load_state_dict(refine_state)

        pos2mot_model.load_state_dict(pos2mot_state)
        optimizer.load_state_dict(optim_state)
        print(f'==> Restored from {Path(ckpt_dir_path, config.ckpt_path + ".pth.tar")}')

    else:
        error_best_pose = None
        error_best_motion = None
        glob_step = 0
        lr_now = config.lr
        start_epoch = 0

    if config.evaluation:
        if config.ckpt_path is not None:
            errors = evaluate(valid_loader_pose, pos2mot_model, device, inference_mode=False,
                              refine_model=refine_model, refine_iteration=config.refine_iteration)
            if evaluate_motion:
                errors = list(errors)
                errors_motion = evaluate(valid_loader_motion, pos2mot_model, device, inference_mode=False,
                                         refine_model=refine_model, refine_iteration=config.refine_iteration)
                for i in range(2, len(errors)):
                    errors[i] = errors_motion[i]

            print(f'{config.actions} ==>',
                  'MPJPE {:.3f} | P-MPJPE {:.3f} | Mot MPJPE {:.3f} | Mot P-MPJPE {:.3f} |'.format(*errors))
        else:
            raise NotImplementedError('Cannot evaluate performance without loading a trained model!')

    # Training starts here
    for epoch in range(start_epoch, config.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        [epoch_loss, epoch_loss_mot, *_], lr_now, glob_step = train(train_loader, pos2mot_model, criterion, optimizer,
                                                                    device, config.lr, lr_now, glob_step,
                                                                    config.lr_decay, config.lr_gamma,
                                                                    refine_model=refine_model,
                                                                    refine_iteration=config.refine_iteration)

        # Evaluate
        # errors = [Pose MPJPE, Pose P-MPJPE, Motion MPJPE, Motion P-MPJPE]
        errors = evaluate(valid_loader_pose, pos2mot_model, device, inference_mode=False,
                          refine_model=refine_model, refine_iteration=config.refine_iteration)
        if evaluate_motion:
            errors = list(errors)
            errors_motion = evaluate(valid_loader_motion, pos2mot_model, device, inference_mode=False,
                                     refine_model=refine_model, refine_iteration=config.refine_iteration)
            for i in range(2, len(errors)):
                errors[i] = errors_motion[i]

        # Update log file
        logger.append([epoch + 1, glob_step, lr_now, epoch_loss, *errors])

        # Save checkpoint
        state = {
            'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step,
            'pos2mot_model': pos2mot_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'refine_model': refine_model.state_dict(),
            'error_best_pose': error_best_pose,
            'error_best_motion': error_best_motion
        }
        if error_best_pose is None or error_best_pose > errors[0]:
            error_best_pose = errors[0]
            suffix = 'pose_best'
            save_ckpt(state, ckpt_dir_path, suffix=suffix)

        if error_best_motion is None or error_best_motion > errors[2]:
            error_best_motion = errors[2]
            suffix = 'motion_best'
            save_ckpt(state, ckpt_dir_path, suffix=suffix)

    logger.close()
    logger.plot(['Pose MPJPE', 'Pose P-MPJPE', 'Motion MPJPE', 'Motion P-MPJPE'])
    save_fig(Path(ckpt_dir_path, 'log.eps'))


if __name__ == '__main__':
    config = parse_args()
    # os setting
    os.environ['OMP_NUM_THREAD'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.visible_devices

    # a simple work-around for the dataloader multithread bug
    torch.set_num_threads(1)
    main(config)
