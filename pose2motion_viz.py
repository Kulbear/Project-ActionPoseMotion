import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from core.visualization import render_animation
from core.dataset.generators import PoseGenerator
from core.models import PoseLifter, MotionGenerator, Pose2MotNet
from core.dataset.data_utils import fetch_inference, read_3d_data, create_2d_data

from pose2motion_arguments import parse_args
from pose2motion_utils import evaluate
from core.transforms import camera_to_world, image_coordinates


def main(config):
    device = config.device
    print('==> Using settings {}'.format(config))

    print('==> Loading dataset...', config.dataset)
    if config.dataset == 'h36m':
        DATASET_NAME = config.dataset.lower()
        from core.dataset.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS

        dataset_path = Path('dataset', DATASET_NAME, f'data_3d_{DATASET_NAME}.npz')
        dataset = Human36mDataset(dataset_path)
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing dataset...')
    dataset = read_3d_data(dataset)
    keypoints_path = Path('dataset', DATASET_NAME, f'data_2d_{DATASET_NAME}_{config.keypoint_source}.npz')
    # keypoints[subject][action][cam_idx]
    keypoints = create_2d_data(keypoints_path, dataset)

    print('==> Initializing dataloaders...')
    # pose_2d_past_segments, pose_3d_past_segments, pose_3d_future_segments, pose_actions = dataset
    data = fetch_inference(config.viz_subject, dataset, keypoints,
                           past=config.past, future=config.future,
                           action=config.viz_action, camera_idx=config.viz_camera,
                           window_stride=config.past if config.viz_target == 'past' else config.future,
                           time_stride=config.time_stride)

    render_loader = DataLoader(PoseGenerator(*data),
                               batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    print('Done!')

    encoder = PoseLifter(config.encoder_ipt_dim, config.encoder_opt_dim,
                         hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                         bidirectional=config.bidirectional, dropout_ratio=config.dropout)
    decoder = MotionGenerator(config.decoder_ipt_dim, config.decoder_opt_dim,
                              hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                              bidirectional=config.bidirectional, dropout_ratio=config.dropout)
    model_pos = Pose2MotNet(encoder, decoder).to(device)
    model_pos.load_state_dict(torch.load(config.ckpt_path)['state_dict'])

    predicted_pose_sequence, pose_sequence_gt, predicted_motion_sequence, motion_sequence_gt, keypoint_sequence = \
        evaluate(render_loader, model_pos, device, inference_mode=True)

    cam = dataset.cameras()[config.viz_subject][config.viz_camera]

    if config.viz_target == 'past':
        prediction = predicted_pose_sequence
        ground_truth = pose_sequence_gt
    elif config.viz_target == 'future':
        prediction = predicted_motion_sequence
        ground_truth = motion_sequence_gt

    input_keypoints = keypoint_sequence
    prediction = camera_to_world(prediction, R=cam['orientation'], t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=0)
    ground_truth[:, :, 2] -= np.min(ground_truth[:, :, 2])

    anim_output = {'Regression': prediction, 'Ground truth': ground_truth}
    input_keypoints = input_keypoints.reshape(-1, 16, 2)
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

    render_animation(input_keypoints, anim_output, dataset.skeleton(), 5, config.viz_bitrate,
                     cam['azimuth'],
                     config.viz_output, limit=config.viz_limit, downsample=config.viz_downsample,
                     size=config.viz_size,
                     input_video_path=config.viz_video, viewport=(cam['res_w'], cam['res_h']),
                     input_video_skip=config.viz_skip)


if __name__ == '__main__':
    config = parse_args()
    # os setting
    os.environ['OMP_NUM_THREAD'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.visible_devices

    # a simple work-around for the dataloader multithread bug
    torch.set_num_threads(1)
    main(config)
