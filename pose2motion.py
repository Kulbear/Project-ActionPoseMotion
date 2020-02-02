import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.log import Logger, save_fig, save_config
from core.data.generators import PoseGenerator
from core.utils import save_ckpt
from core.data.data_utils import fetch, read_3d_data, create_2d_data

from pose2motion_arguments import parse_args
from pose2motion_utils import train, evaluate, evaluate


def main(config):
    device = config.device
    print('==> Using settings {}'.format(config))

    # workaround for motion evaluation calculation
    # window_stride = config.future - config.past
    assert config.past <= config.future, \
        'The current evaluation scheme for motion prediction requires config.past <= config.future'

    evaluate_motion = config.final and config.past != config.future
    if evaluate_motion:
        print('==> Evaluate motion!')
    ckpt_dir_path = Path('experiments', config.exp_name)
    print('==> Created checkpoint dir: {}'.format(ckpt_dir_path))
    if ckpt_dir_path.exists():
        print('==> Found existing checkpoint dir!')
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)

    logger = Logger(Path(ckpt_dir_path, 'log.txt'))
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
                 window_stride=config.window_stride, time_stride=config.time_stride)
    train_loader = DataLoader(PoseGenerator(*data),
                              batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    # pose evaluation
    data = fetch(subjects_test, dataset, keypoints,
                 past=config.past, future=config.future, action_filter=action_filter,
                 window_stride=config.past, time_stride=config.time_stride)
    valid_loader_pose = DataLoader(PoseGenerator(*data),
                                   batch_size=config.batch_size * 4, shuffle=False, num_workers=config.num_workers)

    if evaluate_motion:
        data = fetch(subjects_test, dataset, keypoints,
                     past=config.past, future=config.future, action_filter=action_filter,
                     window_stride=config.future, time_stride=config.time_stride)
        valid_loader_motion = DataLoader(PoseGenerator(*data),
                                         batch_size=config.batch_size * 4,
                                         shuffle=False, num_workers=config.num_workers)
    print('Done!')

    # save experiment config


if __name__ == '__main__':
    config = parse_args()
    # os setting
    os.environ['OMP_NUM_THREAD'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.visible_devices

    # a simple work-around for the dataloader multithread bug
    torch.set_num_threads(1)
    main(config)
