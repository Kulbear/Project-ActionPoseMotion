import numpy as np
import torch
import general_utils as gu
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


# input example
'''
Input:
tensor([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00, -1.42574529e+00, -2.78577072e+00,
          1.31984758e+00,  0.00000000e+00,  0.00000000e+00],
        [ 4.36305484e-01, -1.34098570e+00, -2.61432970e+00,
          1.29885161e+00,  0.00000000e+00,  0.00000000e+00],
        [-1.35005908e-02,  1.16907110e+00,  1.77697921e-01,
          4.54206467e-01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  5.51702618e-02,  1.07797614e-01,
          1.32948831e-01,  0.00000000e+00,  0.00000000e+00],
        [ 8.56086900e-02,  1.57634546e+00, -1.96087659e-02,
          4.42894399e-01,  0.00000000e+00,  0.00000000e+00],
        [-1.05487734e-01, -3.06242266e-02,  9.77824857e-02,
          4.54206526e-01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00, -1.63173045e+00,  1.88910186e-01,
          2.33470604e-01,  0.00000000e+00,  0.00000000e+00],
        [-8.76679676e-02,  9.02161573e-02, -9.25758583e-02,
          2.57077694e-01,  0.00000000e+00,  0.00000000e+00],
        [ 1.27809636e-01, -1.97655879e-01,  1.36915351e-01,
          1.82721660e-01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  5.31881840e-01,  1.23771897e-03,
          1.51034251e-01,  0.00000000e+00,  0.00000000e+00],
        [-1.11793993e-02,  7.91712186e-01,  4.28837829e-02,
          2.78882682e-01,  0.00000000e+00,  0.00000000e+00],
        [ 1.13125074e-01,  4.91560706e-02, -1.43428442e-01,
          2.51733512e-01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  2.62645850e+00,  1.62948430e-01,
          1.51031420e-01,  0.00000000e+00,  0.00000000e+00],
        [-3.97588275e-02, -8.24841055e-01, -4.07882212e-02,
          2.78892934e-01,  0.00000000e+00,  0.00000000e+00],
        [ 3.45872343e-01, -1.67232858e-01, -2.88497875e-01,
          2.51728743e-01,  0.00000000e+00,  0.00000000e+00]])

Output:
tensor([[ 0.0000,  0.0000,  0.0000],
        [-0.1320, -0.0143,  0.0073],
        [-0.0983,  0.0235, -0.4327],
        [-0.0792,  0.0863, -0.8821],
        [ 0.1320,  0.0143, -0.0073],
        [ 0.1034,  0.0313, -0.4490],
        [ 0.0840,  0.0940, -0.8984],
        [-0.0168,  0.0268,  0.2313],
        [-0.0104,  0.0344,  0.4882],
        [-0.0438,  0.0609,  0.6659],
        [ 0.1198,  0.0345,  0.4116],
        [ 0.1878,  0.0443,  0.1413],
        [ 0.2359,  0.0174, -0.1043],
        [-0.1422,  0.0389,  0.4146],
        [-0.2071,  0.0626,  0.1444],
        [-0.2288,  0.0095, -0.1007]])
'''
# input tensor size (num_joint = 16, 6) output tensor size (num_joint = 16, 3)
def lie_to_euler_h36m_hard_code(lie_parameters):
    indices = []
    indices.append([0, 1, 2, 3])  # right leg
    indices.append([0, 4, 5, 6])  # left leg
    indices.append([0, 7, 8, 9])  # head
    indices.append([8, 10, 11, 12])  # left arm
    indices.append([8, 13, 14, 15])  # right arm
    output = torch.zeros(size=(16, 3))
    for i in range(len(indices)):
        euler = lie_to_euler(lie_parameters[indices[i]])
        if indices[i][0] == 0:
            output[indices[i]] = euler
        else:
            print(euler)
            euler = euler - euler[0] + output[indices[i][0]]
            print(euler)
            output[indices[i]] = euler
            output[indices[i]] = euler
    return output


# input tensor size (num_joint, 6) output tensor size (num_joint, 3)
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

# input example
'''
array([[[ 0.0000,  0.0000,  0.0000],
        [-0.1320, -0.0143,  0.0073],
        [-0.0983,  0.0235, -0.4327],
        [-0.0792,  0.0863, -0.8821],
        [ 0.1320,  0.0143, -0.0073],
        [ 0.1034,  0.0313, -0.4490],
        [ 0.0840,  0.0940, -0.8984],
        [-0.0168,  0.0268,  0.2313],
        [-0.0104,  0.0344,  0.4882],
        [-0.0438,  0.0609,  0.6659],
        [ 0.1198,  0.0345,  0.4116],
        [ 0.1878,  0.0443,  0.1413],
        [ 0.2359,  0.0174, -0.1043],
        [-0.1422,  0.0389,  0.4146],
        [-0.2071,  0.0626,  0.1444],
        [-0.2288,  0.0095, -0.1007]]])
array([[[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00, -1.42574529e+00, -2.78577072e+00,
          1.31984758e+00,  0.00000000e+00,  0.00000000e+00],
        [ 4.36305484e-01, -1.34098570e+00, -2.61432970e+00,
          1.29885161e+00,  0.00000000e+00,  0.00000000e+00],
        [-1.35005908e-02,  1.16907110e+00,  1.77697921e-01,
          4.54206467e-01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  5.51702618e-02,  1.07797614e-01,
          1.32948831e-01,  0.00000000e+00,  0.00000000e+00],
        [ 8.56086900e-02,  1.57634546e+00, -1.96087659e-02,
          4.42894399e-01,  0.00000000e+00,  0.00000000e+00],
        [-1.05487734e-01, -3.06242266e-02,  9.77824857e-02,
          4.54206526e-01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00, -1.63173045e+00,  1.88910186e-01,
          2.33470604e-01,  0.00000000e+00,  0.00000000e+00],
        [-8.76679676e-02,  9.02161573e-02, -9.25758583e-02,
          2.57077694e-01,  0.00000000e+00,  0.00000000e+00],
        [ 1.27809636e-01, -1.97655879e-01,  1.36915351e-01,
          1.82721660e-01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  5.31881840e-01,  1.23771897e-03,
          1.51034251e-01,  0.00000000e+00,  0.00000000e+00],
        [-1.11793993e-02,  7.91712186e-01,  4.28837829e-02,
          2.78882682e-01,  0.00000000e+00,  0.00000000e+00],
        [ 1.13125074e-01,  4.91560706e-02, -1.43428442e-01,
          2.51733512e-01,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  2.62645850e+00,  1.62948430e-01,
          1.51031420e-01,  0.00000000e+00,  0.00000000e+00],
        [-3.97588275e-02, -8.24841055e-01, -4.07882212e-02,
          2.78892934e-01,  0.00000000e+00,  0.00000000e+00],
        [ 3.45872343e-01, -1.67232858e-01, -2.88497875e-01,
          2.51728743e-01,  0.00000000e+00,  0.00000000e+00]]])
'''

# input numpy array size (time, num_joint, 3) output numpy array size (time, num_joint, 6)
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
        R = torch.eye(3) + torch.sin(theta) * cross_matrix + (torch.ones(3) - torch.cos(theta)) * torch.matmul(
            cross_matrix,
            cross_matrix)
    return R
