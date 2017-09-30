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
        self._num_classes = 0
        self._input_shape = ()
        self._samples = 0

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def img_width(self):
        return self._img_width

    @property
    def img_height(self):
        return self._img_height

    @property
    def img_channels(self):
        return self._img_channels

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def samples(self):
        return self._samples

    def load_data(self, npz_path: str, test_size: float = 0.2):
        if Path(npz_path).exists():
            # next step, load and parse npz file.
            print("npz data file '{}' already exist.".format(npz_path))
        else:
            # next step, read image files and save to npz file.
            print("npz data file not exist.")
            self._save_npz(npz_path)

        print(
            "Starting load npz file and parse data to '(x_train, y_train), (x_test, y_test)'..."
        )

        # load npz file and parse data for npz file
        with np.load(npz_path) as f:
            x = f['x']
            y = f['y']
            self._number_to_label_dict = dict(f['label'].tolist())
        self._num_classes = y.shape[1]
        self._samples = x.shape[0]
        self._input_shape = x.shape[1:]
        if x.shape[1] == 1:
            self._img_width = x.shape[3]
            self._img_height = x.shape[2]
            self._img_channels = x.shape[1]
        else:
            self._img_width = x.shape[2]
            self._img_height = x.shape[1]
            self._img_channels = x.shape[3]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=2)

        print('Load npz file and parse data samples ({}) completed.'.format(
            self._samples))

        return (x_train, y_train), (x_test, y_test)

    def query_label(self, number: int) -> str:
        return self._number_to_label_dict.get(number, "Invalid number")

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

        print('Starting read and process image files...')
        for label, files in self._scan_dir():
            for file in files:
                im = Image.open(file)
                self._samples += 1
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

        self._num_classes = len(self._number_to_label_dict)
        print('Total read and process ({}) image files.'.format(self._samples))
        x = np.array(x).astype('float32')
        x /= 255  # normalize

        y = np.array(y).astype('int64')

        if K.image_data_format() == 'channels_first':
            x = x.reshape(x.shape[0], self._img_channels, self._img_height,
                          self._img_width)
        else:  # channels_last
            x = x.reshape(x.shape[0], self._img_height, self._img_width,
                          self._img_channels)

        y = to_categorical(y, self._num_classes)

        return (x, y)

    def _save_npz(self, npz_path: str) -> None:
        (x, y) = self._im2np()

        print('Starting save npz data...')
        np.savez(
            npz_path, x=x, y=y, label=np.array(self._number_to_label_dict))
        print("Saved npz data to '{}'".format(Path(npz_path).resolve()))