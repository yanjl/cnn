"""
Version:v2
Author: yan.jl
        sinberyh@aliyun.com
        9/29/2017
"""

from pathlib import Path

import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self,
                 img_dir: str = './data',
                 img_width: int = 224,
                 img_height: int = 224,
                 img_channels: int = 3) -> None:
        self._img_width = img_width
        self._img_height = img_height
        self._img_channels = img_channels
        self._img_dir = img_dir
        self._number_to_label_dict = {}
        self._saved_data = False
        self._num_classes = 0

    def _collect_dirs(self):
        return (dir for dir in Path(self._img_dir).iterdir()
                if dir.is_dir() and not dir.name.startswith('.'))

    def _scan_dir(self):
        """
        return [(label1, [file1,file2,...,fileN]),
             (label2, [file1,file2,...,fileN]),
              ...,
               (labelN, [file1,file2,...,fileN])]
        """

        def _collect_files(path):  # type: Path
            return (file.resolve() for file in path.iterdir()
                    if file.is_file() and not file.name.startswith('.'))

        return ((label.name, list(_collect_files(label)))
                for label in self._collect_dirs())

    def _im2np(self):
        x = []
        y = []
        label_code = 0
        for label, files in self._scan_dir():
            for file in files:
                im = Image.open(file)
                # channels convert
                if self._img_channels == 3:
                    im = im.convert("RGB")
                else:
                    im = im.convert("L")
                # resize  (widht, height)
                im = im.resize((self._img_width, self._img_height))
                x.append(np.array(im))
                y.append(label_code)
            self._number_to_label_dict[label_code] = label
            label_code += 1
        x = np.array(x).astype('float32')
        x /= 255  # normalize

        y = np.array(y).astype('int64')

        return (x, y)

    def _save_npz(self, dest_path: str = './data',
                  filename: str = 'test_data') -> None:
        (x, y) = self._im2np()
        np.savez(dest_path / filename, x=x, y=y)
        self._saved_data = True

    def load_data(self, npz_path: str, test_size: float):
        if not self._saved_data:
            print('starting save npz data.')
            self._save_npz(Path(npz_path).parent, Path(npz_path).name)
            print('saved npz data to {}.'.format(npz_path))

        print("starting load data to '(x_train, y_train), (x_test, y_test)'.")
        with np.load(npz_path) as f:
            x = f['x']
            y = f['y']

        self._num_classes = len(set(y))
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=2)

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], self._img_channels,
                                      self._img_height, self._img_width)
            x_test = x_test.reshape(x_test.shape[0], self._img_channels,
                                    self._img_height, self._img_width)
        else:  # channels_last
            x_train = x_train.reshape(x_train.shape[0], self._img_height,
                                      self._img_width, self._img_channels)
            x_test = x_test.reshape(x_test.shape[0], self._img_height,
                                    self._img_width, self._img_channels)

        y_train = to_categorical(y_train, self._num_classes)
        y_test = to_categorical(y_test, self._num_classes)

        print('end load data.')
        return (x_train, y_train), (x_test, y_test)

    def query_class(self, label: int) -> str:
        return self._number_to_label_dict.get(label, "invalid label")
