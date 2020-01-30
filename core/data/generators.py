import numpy as np
import torch
from torch.utils.data import Dataset

ACTION_INDEX_MAP = {
    'h36m': {
        'Directions': 0,
        'Discussion': 1,
        'Eating': 2,
        'Greeting': 3,
        'Phoning': 4,
        'Photo': 5,
        'Posing': 6,
        'Purchases': 7,
        'Sitting': 8,
        'SittingDown': 9,
        'Smoking': 10,
        'Waiting': 11,
        'WalkDog': 12,
        'Walking': 13,
        'WalkTogether': 14
    },
    'ntu': {  # TODO: complete action list for NTU dataset
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        18: 18,
        19: 19,
        20: 20,
        21: 21,
        22: 22,
        23: 23,
        24: 24,
        25: 25,
        26: 26,
        27: 27,
        28: 28,
        29: 29,
        30: 30,
        31: 31,
        32: 32,
        33: 33,
        34: 34,
        35: 35,
        36: 36,
        37: 37,
        38: 38,
        39: 39,
        40: 40,
        41: 41,
        42: 42,
        43: 43,
        44: 44,
        45: 45,
        46: 46,
        47: 47,
        48: 48,
        49: 49,
        50: 50,
        51: 51,
        52: 52,
        53: 53,
        54: 54,
        55: 55,
        56: 56,
        57: 57,
        58: 58,
        59: 59
    }
}


class PoseGenerator(Dataset):
    def __init__(self, pose_2d_past, pose_3d_past, pose_3d_future, actions, dataset_name='h36m'):
        self._pose_2d_past = np.array(pose_2d_past)
        self._pose_3d_past = np.array(pose_3d_past)
        self._pose_3d_future = np.array(pose_3d_future)
        self._actions = actions
        self._dataset_name = dataset_name

        assert len(self._actions) == len(self._pose_3d_past)
        assert len(self._pose_3d_past) == len(self._pose_3d_future)
        assert len(self._pose_2d_past) == len(self._actions)
        print('Generating {} pose sequences...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_2d_past = torch.from_numpy(self._pose_2d_past[index]).float()
        out_3d_past = torch.from_numpy(self._pose_3d_past[index]).float()
        out_3d_future = torch.from_numpy(self._pose_3d_future[index]).float()

        return {
            'pose_2d': out_2d_past,
            'pose_3d': out_3d_past,
            'future_pose_3d': out_3d_future,
            'action_type': ACTION_INDEX_MAP[self._dataset_name][self._actions[index]]
        }

    def __len__(self):
        return len(self._actions)
