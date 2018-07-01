from scipy.misc import imread, imresize
import numpy as np


def load_image(filename, size=None):
    """Load and resize an image from disk

    Args:
        filename (str): Path to file
        size (int): Size of the shortest dimension after rescaling
    """
    img = imread(filename)
    if size is not None:
        org_shape = np.array(img.shape[:2])
        min_idx = np.argmin(org_shape)
        scale_factor = float(size) / org_shape[min_idx]
        img = imresize(img, scale_factor)
    
    return img


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img):
    """Preprocess an image for SqueezeNet.
    
    Subtracts the pixel mean and divides by the standard deviation
    
    Args:
        img (np.array): Input image for preprocessing
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD
