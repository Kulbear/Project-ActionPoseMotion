import os
from pathlib import Path

from core.data.mocap_dataset import MocapDataset


class NTUDataset(MocapDataset):
    def __init__(self, path, include_image: bool = False, *args, **kwargs):
        files = [f for f in os.listdir(path) if os.path.isfile(Path(path, f))]
        self._path = path
        self._data = {}
        for filename in files:
            subject = filename[8:12]
            if subject not in self._data:
                self._data[subject] = []
            self._data[subject].append(filename)
        print(len(files))
        print(len(self._data))

    def get_path(self):
        return self._path
