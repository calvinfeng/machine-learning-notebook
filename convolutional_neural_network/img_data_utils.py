from PIL import Image
import numpy as np
import scipy
import scipy.ndimage as ndimage
from sys import getsizeof
import time
from matplotlib import pyplot as plt


def load_jpg_from_dir(dir, resize_px=32, num_images_per_class=12500, start_idx=1):
    img_set, labels = [], []

    i = start_idx
    while i < start_idx + num_images_per_class:
        try:
            image_name = "cat.%d.jpg" % i
            image = np.array(ndimage.imread(dir + image_name, flatten=False))
            resized_image = scipy.misc.imresize(image, size=(resize_px, resize_px))
            img_set.append(resized_image)
            labels.append(1)
            i += 1
        except IOError as e:
            break

    i = start_idx
    while i < start_idx + num_images_per_class:
        try:
            image_name = "dog.%d.jpg" % i
            image = np.array(ndimage.imread(dir + image_name, flatten=False))
            resized_image = scipy.misc.imresize(image, size=(resize_px, resize_px))
            img_set.append(resized_image)
            labels.append(0)
            i += 1
        except IOError as e:
            break

    return {
        'X': np.stack(img_set, axis=0),
        'y': np.array(labels)
    }
