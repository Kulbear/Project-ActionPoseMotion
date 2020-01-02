import numpy as np

from core.transforms import world_to_camera, normalize_screen_coordinates


def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset


def fetch(subjects, dataset, keypoints,
          past=8, future=16, window_stride=None,
          action_filter=None, time_stride=1, parse_3d_poses=True):
    # If not specified, use past size as the sliding window strides
    if window_stride is None:
        window_stride = past

    out_poses_3d = []
    out_poses_2d = []
    out_actions = []

    for subject in subjects:
        print('==> Fetching subject:', subject)
        for action in keypoints[subject].keys():
            # print('==> Fetching vide:', action)
            action_type = action.split(' ')[0]
            # Example: subject => S1, action => Walking 1
            # Note the action is actually the video name
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action_type == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action_type] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if time_stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::time_stride]
            out_actions[i] = out_actions[i][::time_stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::time_stride]

    # out_poses_3d, out_poses_2d, out_actions
    pose_2d_past_segments = []
    pose_3d_past_segments = []
    pose_3d_future_segments = []
    pose_actions = []

    for cam_idx in range(len(out_poses_2d)):
        for i in range(past, len(out_poses_2d[cam_idx]) - future, window_stride):
            pose_2d_past_segments.append(out_poses_2d[cam_idx][i - past:i])
            pose_actions.append(out_actions[cam_idx][i])
            if out_poses_3d is not None:
                pose_3d_past_segments.append(out_poses_3d[cam_idx][i - past:i])
                pose_3d_future_segments.append(out_poses_3d[cam_idx][i:i + future])

    return pose_2d_past_segments, pose_3d_past_segments, pose_3d_future_segments, pose_actions
