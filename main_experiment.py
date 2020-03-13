import os
import pickle
from pathlib import Path
from datetime import date

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.log import Logger, save_fig, save_config
from core.dataset.generators import PoseGenerator
from core.models import (
    PoseLifter, MotionGenerator,
    Pose2MotNet, REFINEMENT_ARCHS
)
from core.utils import save_ckpt, load_ckpt
from core.dataset.data_utils import fetch, read_3d_data, create_2d_data

from pose2motion_arguments import parse_args
from pose2motion_utils import train, evaluate


def main(config):
    device = config.device
    print('==> Using settings {}'.format(config))

    exp_name = f'{config.dataset}-{config.keypoint_source}-p{config.past}-f{config.future}-' \
               f'h{config.hid_dim}-wMotion_{config.train_motion_model}-rfv_{config.refine_version}-' \
               f'lie_{config.use_lie_algebra}-liew_{config.lie_weight}'

    exp_name = exp_name + (config.exp_postfix if config.exp_postfix else date.today().strftime("%b-%d-%Y"))

    ckpt_dir_path = Path('experiments', exp_name)
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

    # if evaluate_motion:
    #     data = fetch(subjects_test, dataset, keypoints,
    #                  past=config.past, future=config.future, action_filter=action_filter,
    #                  window_stride=config.future, time_stride=config.time_stride, train=False)
    #     valid_loader_motion = DataLoader(PoseGenerator(*data),
    #                                      batch_size=config.batch_size * 4,
    #                                      shuffle=False, num_workers=config.num_workers)
    print('Done!')

    # save experiment config
    save_config(config, Path(ckpt_dir_path, 'sample_config.json'))

    encoder = PoseLifter(config.encoder_ipt_dim, config.encoder_opt_dim,
                         hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                         bidirectional=config.bidirectional, dropout_ratio=config.dropout,
                         use_lie_algebra=config.use_lie_algebra)
    decoder = MotionGenerator(config.decoder_ipt_dim, config.decoder_opt_dim,
                              hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                              bidirectional=config.bidirectional, dropout_ratio=config.dropout,
                              use_lie_algebra=config.use_lie_algebra)

    pos2mot_model = Pose2MotNet(encoder, decoder, use_lie_algebra=config.use_lie_algebra).to(device)
    total_params = sum(p.numel() for p in pos2mot_model.parameters() if p.requires_grad)
    print('Pose model # params:', total_params)

    try:
        RefineNet = REFINEMENT_ARCHS[config.refine_version]
    except:
        print('No Refinement!!')
        RefineNet = REFINEMENT_ARCHS.get(1)
        config.refine_iteration = 0

    if config.use_lie_algebra:
        assert config.refine_version in (1, 2)
        refine_model = RefineNet(config.decoder_opt_dim * 3, config.decoder_opt_dim * 3,
                                 hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                                 bidirectional=config.bidirectional, dropout_ratio=config.dropout,
                                 size=(config.batch_size, config.past + config.future, config.decoder_opt_dim * 3),
                                 use_lie_algebra=config.use_lie_algebra).to(device)
        total_params = sum(p.numel() for p in refine_model.parameters() if p.requires_grad)
        print('Refine model # params:', total_params)

    else:
        refine_model = RefineNet(config.decoder_opt_dim, config.decoder_opt_dim,
                                 hid_dim=config.hid_dim, n_layers=config.num_recurrent_layers,
                                 bidirectional=config.bidirectional, dropout_ratio=config.dropout,
                                 size=(config.batch_size,
                                       config.past + config.future,
                                       config.decoder_opt_dim)).to(device)

        total_params = sum(p.numel() for p in refine_model.parameters() if p.requires_grad)
        print('Refine model # params:', total_params)

    criterion = nn.MSELoss().to(device)
    parameter_to_optim = []
    if refine_model is not None and config.refine_iteration > 0:
        parameter_to_optim += list(refine_model.parameters())
    if config.pos_loss_on:
        parameter_to_optim += list(encoder.parameters())
    if config.train_motion_model:
        parameter_to_optim += list(decoder.parameters())
    optimizer = torch.optim.Adam(parameter_to_optim, lr=config.lr)
    lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3,
                                                        verbose=True, cooldown=0, min_lr=0.00005, eps=1e-08)

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

    # Training starts here
    for epoch in range(start_epoch, config.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        [epoch_loss, epoch_loss_mot, *_], lr_now, glob_step = train(train_loader, pos2mot_model, criterion,
                                                                    optimizer, device, config.lr, lr_now, glob_step,
                                                                    config.lr_decay, config.lr_gamma,
                                                                    refine_model=refine_model,
                                                                    refine_iteration=config.refine_iteration,
                                                                    pos_loss_on=config.pos_loss_on,
                                                                    train_motion_model=config.train_motion_model,
                                                                    step_lr=config.lr_schedule == 'step',
                                                                    use_lie_algebra=config.use_lie_algebra,
                                                                    lie_weight=config.lie_weight)

        # Evaluate
        # errors = [Pose MPJPE, Pose P-MPJPE, Motion MPJPE, Motion P-MPJPE]
        errors = evaluate(valid_loader_pose, pos2mot_model, device, inference_mode=False,
                          refine_model=refine_model, refine_iteration=config.refine_iteration,
                          use_lie_algebra=config.use_lie_algebra)

        if evaluate_motion:
            errors = list(errors)
            errors_motion = evaluate(valid_loader_motion, pos2mot_model, device, inference_mode=False,
                                     refine_model=refine_model, refine_iteration=config.refine_iteration,
                                     use_lie_algebra=config.use_lie_algebra)
            for i in range(2, len(errors)):
                errors[i] = errors_motion[i]

        if config.lr_schedule == 'reduce':
            if config.primary_target == 'pose':
                lr_sch.step(errors[0])
            elif config.primary_target == 'motion':
                lr_sch.step(errors[2])
            else:
                raise NotImplementedError('Primary target has to be set as "pose" or "motion"!')

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

    # logger.plot(['Pose MPJPE', 'Pose P-MPJPE', 'Motion MPJPE', 'Motion P-MPJPE'])
    # save_fig(Path(ckpt_dir_path, 'log.eps'))
    logger.close()


if __name__ == '__main__':
    config = parse_args()
    # os setting
    os.environ['OMP_NUM_THREAD'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.visible_devices

    # a simple work-around for the dataloader multithread bug
    torch.set_num_threads(1)
    main(config)
