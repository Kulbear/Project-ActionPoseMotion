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
        'drink water': 0,
        'eat meal/snack': 1
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
