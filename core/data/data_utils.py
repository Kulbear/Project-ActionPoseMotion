import numpy as np
from pathlib import Path
from core.transforms import world_to_camera, normalize_screen_coordinates


def ntu_fetch(subjects, dataset, past=8, future=16, window_stride=8):
    pose_2d_past_segments = []
    pose_3d_past_segments = []
    pose_3d_future_segments = []
    pose_actions = []
    for subject in subjects:
        print('==> Fetching subject:', subject)
        num_data = 0
        for filename in dataset[subject]:
            with open(Path(dataset.get_path(), filename), 'r') as fr:
                str_data = fr.readlines()

            num_frames = int(str_data[0].strip('\r\n'))
            bodies_data = dict()
            valid_frames = -1  # 0-based index
            current_line = 1

            for f in range(num_frames):
                num_bodies = int(str_data[current_line].strip('\r\n'))
                current_line += 1

                if num_bodies != 1:
                    break

                valid_frames += 1
                joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
                colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)

                for b in range(num_bodies):
                    bodyID = str_data[current_line].strip('\r\n').split()[0]
                    current_line += 1
                    num_joints = int(str_data[current_line].strip('\r\n'))  # 25 joints
                    current_line += 1

                    for j in range(num_joints):
                        temp_str = str_data[current_line].strip('\r\n').split()
                        joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                        colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                        current_line += 1

                    if bodyID not in bodies_data:  # Add a new body's data
                        body_data = dict()
                        body_data['joints'] = joints[b, np.newaxis]  # ndarray: (25, 3)
                        body_data['colors'] = colors[b, np.newaxis]  # ndarray: (1, 25, 2)
                        body_data['interval'] = [valid_frames]  # the index of the first frame
                    else:  # Update an already existed body's data
                        body_data = bodies_data[bodyID]
                        # Stack each body's data of each frame along the frame order
                        body_data['joints'] = np.vstack((body_data['joints'], joints[b, np.newaxis]))
                        body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                        pre_frame_idx = body_data['interval'][-1]
                        body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index
                    bodies_data[bodyID] = body_data  # Update bodies_data
            for i in range(0, len(body_data['colors']) - past - future, window_stride):
                num_data += 1
                pose_2d_past_segments.append(body_data['colors'][i: i + past, :, :])
                pose_3d_past_segments.append(body_data['joints'][i: i + past, :, :])
                pose_3d_future_segments.append(body_data['joints'][i + past: i + past + future, :, :])
                pose_actions.append(int(filename[17:20]) - 1)
        print('==> Finish Fetching subject:', subject)
        print('Total %d records' % num_data)
    print(len(pose_actions))
    return pose_2d_past_segments, pose_3d_past_segments, pose_3d_future_segments, pose_actions


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


def fetch_inference(subject, dataset, keypoints,
                    past=8, future=16, window_stride=None,
                    action='Walking 1', camera_idx=0, time_stride=1, parse_3d_poses=True):
    # If not specified, use past size as the sliding window strides
    if window_stride is None:
        window_stride = past

    out_poses_3d = []
    out_poses_2d = []
    out_actions = []
    # Example: subject => S1, action => Walking 1
    print('==> Fetching subject:', subject)
    print('==> Fetching action:', action)
    action_type = action.split(' ')[0]
    poses_2d = keypoints[subject][action]
    for i in range(len(poses_2d)):  # Iterate across cameras
        if i == camera_idx:
            out_poses_2d.append(poses_2d[i])
            out_actions.append([action_type] * poses_2d[i].shape[0])

    if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)):  # Iterate across cameras
            if i == camera_idx:
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
        if cam_idx == camera_idx:
            for i in range(past, len(out_poses_2d[cam_idx]) - future, window_stride):
                pose_2d_past_segments.append(out_poses_2d[cam_idx][i - past:i])
                pose_actions.append(out_actions[cam_idx][i])
                if out_poses_3d is not None:
                    pose_3d_past_segments.append(out_poses_3d[cam_idx][i - past:i])
                    pose_3d_future_segments.append(out_poses_3d[cam_idx][i:i + future])
    return pose_2d_past_segments, pose_3d_past_segments, pose_3d_future_segments, pose_actions


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
