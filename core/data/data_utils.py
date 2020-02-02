import numpy as np
import torch
import general_utils as gu
from core.transforms import world_to_camera, normalize_screen_coordinates

def create_lie_data(data_path):
    liepoints = np.load(data_path, allow_pickle=True)['positions_2d'].item()
    return liepoints


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
            #anim['positions_lie'] = positions_3d
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
    out_poses_lie = []
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
                    out_poses_lie.append(convert_to_lie(poses_3d[i]))

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if time_stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::time_stride]
            out_actions[i] = out_actions[i][::time_stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::time_stride]
            if out_poses_lie is not None:
                out_poses_lie[i] = out_poses_lie[i][::time_stride]

    # out_poses_3d, out_poses_2d, out_actions, out_poses_lie
    pose_2d_past_segments = []
    pose_3d_past_segments = []
    pose_3d_future_segments = []
    pose_lie_segments = []
    pose_actions = []

    for cam_idx in range(len(out_poses_2d)):
        for i in range(past, len(out_poses_2d[cam_idx]) - future, window_stride):
            pose_2d_past_segments.append(out_poses_2d[cam_idx][i - past:i])
            pose_actions.append(out_actions[cam_idx][i])
            if out_poses_3d is not None:
                pose_3d_past_segments.append(out_poses_3d[cam_idx][i - past:i])
                pose_3d_future_segments.append(out_poses_3d[cam_idx][i:i + future])
                pose_lie_segments.append(out_poses_lie[cam_idx][i - past:i])

    return pose_2d_past_segments, pose_3d_past_segments, pose_3d_future_segments, pose_lie_segments, pose_actions


def lie_to_euler(lie_parameters):
    num_joint = lie_parameters.shape[0]
    se = torch.zeros((num_joint, 4, 4))
    for j in range(num_joint):
        if j == 0:
            se[j, :, :] = lietomatrix(lie_parameters[j + 1, 0:3], lie_parameters[j, 3:6])
        elif j == num_joint - 1:
            se[j, :, :] = torch.matmul(torch.squeeze(se[j - 1, :, :]),
                                       lietomatrix(torch.tensor([0, 0.0, 0]), lie_parameters[j, 3:6]))
        else:
            se[j, :, :] = torch.matmul(torch.squeeze(se[j - 1, :, :]),
                                       lietomatrix(lie_parameters[j + 1, 0:3], lie_parameters[j, 3:6]))

    joint_xyz = torch.zeros((num_joint, 3))

    for j in range(num_joint):
        coor = torch.tensor([0, 0, 0, 1.0]).reshape((4, 1))
        xyz = torch.matmul(torch.squeeze(se[j, :, :]), coor)
        joint_xyz[j, :] = xyz[0:3, 0]
    return joint_xyz


def xyz_to_lie_parameters(joint_xyz):
    num_joint = joint_xyz.shape[0]
    joint_xyz = joint_xyz - joint_xyz[0]
    lie_parameters = np.zeros([num_joint - 1, 6])
    # Location of joint 1 in chain
    for j in range(num_joint - 1):
        lie_parameters[j, 3] = np.linalg.norm(
            (joint_xyz[j, :] - joint_xyz[j + 1, :]))
    # Axis angle parameters of rotation
    for j in range(num_joint - 2, -1, -1):
        v = np.squeeze(joint_xyz[j + 1, :] - joint_xyz[[j], :])
        vhat = v / np.linalg.norm(v)
        if j == 0:
            uhat = [1, 0, 0]
        else:
            u = np.squeeze(joint_xyz[j, :] - joint_xyz[j - 1, :])
            uhat = u / np.linalg.norm(u)
        a = np.transpose(gu.rotmat(gu.findrot([1, 0, 0], uhat)))
        b = gu.rotmat(gu.findrot([1, 0, 0], vhat))
        c = gu.axis_angle(np.dot(a, b))
        lie_parameters[j, 0: 3] = c
    return lie_parameters


def convert_to_lie(joint_xyz):
    index = []
    index.append([0, 1, 2, 3])  # right leg
    index.append([0, 4, 5, 6])  # left leg
    index.append([0, 7, 8, 9])  # head
    index.append([8, 10, 11, 12])  # left arm
    index.append([8, 13, 14, 15])  # right arm
    num_frame = joint_xyz.shape[0]
    lie_parameters = np.zeros([joint_xyz.shape[0], 16, 6])
    for i in range(num_frame):
        joint_xyz[i, :, :] = joint_xyz[i, :, :] - joint_xyz[i, 0, :]
        for k in range(len(index)):
            lie_parameters[i, 3 * k + 1: 3 * k + 4, :] = xyz_to_lie_parameters(joint_xyz[i][index[k]])
    return lie_parameters


def lietomatrix(angle, trans):
    R = expmap2rotmat(angle)
    T = trans
    SEmatrix = torch.cat((torch.cat((R, T.reshape(3, 1)), 1), torch.tensor([[0, 0, 0, 1.0]])))

    return SEmatrix


def expmap2rotmat(A):
    theta = torch.norm(A)
    if theta == 0:
        R = torch.eye(3)
    else:
        A = A / theta
        cross_matrix = torch.tensor([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])
        R = torch.eye(3) + torch.sin(theta) * cross_matrix + (torch.ones(3) - torch.cos(theta)) * torch.matmul(cross_matrix,
                                                                                                             cross_matrix)
    return R
