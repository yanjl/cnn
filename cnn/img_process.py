"""
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

img_rows = 320  # image height
img_cols = 320  # image width
num_channels = 3  # color channels  1 or 3
class_dict = {}


def _scan_dir(src_img_dir: str):
    """
    return [(label1, [file1,file2,...,fileN]),
             (label2, [file1,file2,...,fileN]),
              ...,
               (labelN, [file1,file2,...,fileN])]
    """

    def _collect_dirs(path: str):
        return (dir for dir in Path(path).iterdir()
                if dir.is_dir() and not dir.name.startswith('.'))

    def _collect_files(path):  # path:Path
        return (file.resolve() for file in path.iterdir()
                if file.is_file() and not file.name.startswith('.'))

    return ((label.name, list(_collect_files(label)))
            for label in list(_collect_dirs(src_img_dir)))


def _im2np(path: str):
    x = []
    y = []
    label_code = 0
    for label, files in _scan_dir(path):
        for file in files:
            im = Image.open(file)
            # channels convert
            if num_channels == 3:
                im = im.convert("RGB")
            else:
                im = im.convert("L")
            # resize  (widht, height)
            im = im.resize((img_cols, img_rows))
            x.append(np.array(im))
            y.append(label_code)
        class_dict[label_code] = label
        label_code += 1
    x = np.array(x).astype('float32')
    x /= 255  # normalize

    return (x, y)


def save_npz(src_img_path: str, dest_path: str, filename: str):
    (x, y) = _im2np(src_img_path)
    np.savez(dest_path + '/' + filename, x=x, y=y)


def load_data(npz_path: str, test_size: float):
    with np.load(npz_path) as f:
        x = f['x']
        y = f['y']

    num_classes = len(set(y))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=2)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], num_channels, img_rows,
                                  img_cols)
        x_test = x_test.reshape(x_test.shape[0], num_channels, img_rows,
                                img_cols)
    else:  # channels_last
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,
                                  num_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,
                                num_channels)

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


"""
image dir structure
cnn/
    dog/ 1.jpg
         2.jpg
    cat/ 1.jpg
         2.jpg
         3.jpg
    hosr/a.jpg
         b.jpg
         c.jpg
    ....
src_img_path = ./cnn
Example:

>>>import img_process
>>>img_process.save_npz('../cnn', '../cnn', 'test111')
>>>(x_train, y_train), (x_test, y_test) = img_process.load_data(
    '../cnn/test111.npz', 0.2)
"""