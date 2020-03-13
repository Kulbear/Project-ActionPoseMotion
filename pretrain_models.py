import os

import pickle
from pathlib import Path

from torch.utils.data import DataLoader

from core.dataset.generators import PoseGenerator
from core.models import (
    PoseLifter, MotionGenerator,
    KeypointRefineNet
)
from core.dataset.data_utils import fetch, read_3d_data, create_2d_data

from pretrain_utils import *
from core.utils import save_ckpt


def main(config):
    device = config.device
    print('==> Using settings {}'.format(config))

    print('==> Loading dataset...', config.dataset)
    if config.dataset == 'h36m':
        DATASET_NAME = config.dataset.lower()
        from core.dataset.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS

        dataset_path = Path('data', DATASET_NAME, f'data_3d_{DATASET_NAME}.npz')
        dataset = Human36mDataset(dataset_path)
        subjects_train = TRAIN_SUBJECTS
        subjects_test = TEST_SUBJECTS
    elif config.dataset == 'humanevaI':
        DATASET_NAME = config.dataset.lower()
        dataset_path = Path('data', DATASET_NAME, f'data_3d_{DATASET_NAME}.npz')

        raise NotImplementedError('Haven\'t done HumanEva-I!')
    else:
        raise KeyError('Invalid dataset')

    # the input stored in dataset are world coordinates
    # so we need to convert world coordinates to camera coordinates
    # and also remove the global offset (It's done inside `read_3d_data`)
    # Therefore it produces 4 groups of results, which are corresponding to 4 cameras
    print('==> Preparing dataset...')
    dataset = read_3d_data(dataset)

    # There are 4 groups of 2D keypoints (screen coordinates) for each of the subject
    # They are stored in the format keypoints[subject][action][cam_idx]
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    # results in screen (aka frame) coordinates
    print('==> Loading 2D detections...')
    keypoints_path = Path('data', DATASET_NAME, f'data_2d_{DATASET_NAME}_{config.keypoint_source}.npz')
    keypoints_dt = create_2d_data(keypoints_path, dataset)

    keypoints_path = Path('data', DATASET_NAME, f'data_2d_{DATASET_NAME}_gt.npz')
    keypoints_gt = create_2d_data(keypoints_path, dataset)

    print('==> Loading Lie Algebra dataset...')
    try:
        lie_dataset_train = pickle.load(open(Path('data/', DATASET_NAME, 'train_data_lie.pkl'), 'rb'))
        lie_dataset_test = pickle.load(open(Path('data/', DATASET_NAME, 'valid_data_lie.pkl'), 'rb'))
        print(f'==> Found existing Lie repr at {Path("data/", DATASET_NAME)}')
    except:
        print('==> No existing Lie Algebra dataset, will create it now and save it.')

    print('==> Initializing dataloaders...')
    action_filter = None if config.actions == '*' else config.actions.split(',')
    # action_filter = 'Walking 1'
    # pose_2d_past_segments, pose_2d_past_gt_segments,
    # pose_3d_past_segments, pose_3d_future_segments,
    # pose_lie_segments, pose_lie_future_segments, pose_actions
    data = fetch(subjects_train, dataset, keypoints_gt, keypoints_dt, lie_dataset=lie_dataset_train,
                 past=config.past, future=config.future, action_filter=action_filter,
                 window_stride=config.window_stride, time_stride=config.time_stride, train=True)
    train_loader = DataLoader(PoseGenerator(*data),
                              batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    # pose evaluation
    data = fetch(subjects_test, dataset, keypoints_gt, keypoints_dt, lie_dataset=lie_dataset_test,
                 past=config.past, future=config.future, action_filter=action_filter,
                 window_stride=config.past, time_stride=config.time_stride, train=False)
    valid_loader_pose = DataLoader(PoseGenerator(*data),
                                   batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    print('==> Finished creating dataloaders!')

    # pre-refinement model
    pre_refiner = KeypointRefineNet(config.encoder_ipt_dim, config.encoder_ipt_dim,
                                    hid_dim=config.hid_dim, n_layers=1,
                                    bidirectional=config.bidirectional).to(device)

    # # global refinement model
    # try:
    #     RefineNet = REFINEMENT_ARCHS[config.refine_version]
    # except:
    #     RefineNet = REFINEMENT_ARCHS.get(1)
    #     config.refine_iteration = 0
    #
    # if config.use_lie_algebra:
    #     assert config.refine_version in (1, 2)
    #     global_refiner = RefineNet(config.decoder_opt_dim * 3, config.decoder_opt_dim * 3,
    #                                hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
    #                                bidirectional=config.bidirectional, dropout_ratio=config.dropout,
    #                                size=(config.batch_size, config.past + config.future, config.decoder_opt_dim * 3),
    #                                use_lie_algebra=config.use_lie_algebra).to(device)
    #
    #
    # else:
    #     global_refiner = RefineNet(config.decoder_opt_dim, config.decoder_opt_dim,
    #                                hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
    #                                bidirectional=config.bidirectional, dropout_ratio=config.dropout,
    #                                size=(config.batch_size,
    #                                      config.past + config.future,
    #                                      config.decoder_opt_dim)).to(device)

    encoder = PoseLifter(config.encoder_ipt_dim, config.encoder_opt_dim,
                         hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                         bidirectional=config.bidirectional, dropout_ratio=config.dropout,
                         use_lie_algebra=config.use_lie_algebra).to(device)

    decoder = MotionGenerator(config.decoder_ipt_dim, config.decoder_opt_dim,
                              hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                              bidirectional=config.bidirectional, dropout_ratio=config.dropout,
                              use_lie_algebra=config.use_lie_algebra).to(device)

    # models = [pre_refiner, encoder, decoder, global_refiner]
    models = [pre_refiner, encoder, decoder]
    torch.set_grad_enabled(True)
    parameter_to_optim = []
    for model in models:
        model.train()
        parameter_to_optim += list(model.parameters())

    lr_now = 1e-3
    optimizer = torch.optim.Adam(parameter_to_optim, lr=lr_now)
    loss_function = nn.MSELoss().to(device)

    # Training starts here
    lr_now = adjust_learning_rate(optimizer, hardcode=1e-3)
    if config.pretrain_on['train_pre_refiner']:
        print('Train Pre Refiner')
        for epoch in range(config.pretrain_on['train_pre_refiner']):
            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))
            train_pre_refiner(train_loader, valid_loader_pose, loss_function, optimizer, pre_refiner)
            # lr_now = adjust_learning_rate(optimizer, factor=0.9)

    lr_now = adjust_learning_rate(optimizer, hardcode=1e-3)
    if config.pretrain_on['train_pose_lifter']:
        print('Train Pose Lifter')
        for epoch in range(config.pretrain_on['train_pose_lifter']):
            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))
            train_pose_lifter(train_loader, valid_loader_pose,
                              loss_function, optimizer, encoder,
                              use_lie_algebra=config.use_lie_algebra)
            lr_now = adjust_learning_rate(optimizer, factor=0.9)

    lr_now = adjust_learning_rate(optimizer, hardcode=1e-3)
    if config.pretrain_on['train_motion_generator']:
        print('Train Motion Generator')
        for epoch in range(config.pretrain_on['train_motion_generator']):
            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))
            train_motion_generator(train_loader, valid_loader_pose,
                                   loss_function, optimizer, decoder,
                                   use_lie_algebra=config.use_lie_algebra)
            lr_now = adjust_learning_rate(optimizer, factor=0.9)

    state = {
        'pre_refiner': pre_refiner,
        'encoder': encoder,
        'decoder': decoder,
        'optimizer': optimizer.state_dict(),
    }
    save_ckpt(state, '.', suffix=f'test_pretrain_lie{config.use_lie_algebra}')


if __name__ == '__main__':
    import json
    import argparse

    config = json.load(open('pretrain_config.json'))
    config = argparse.Namespace(**config)

    os.environ['OMP_NUM_THREAD'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.visible_devices

    # a simple work-around for the dataloader multithread bug
    torch.set_num_threads(1)
    main(config)
