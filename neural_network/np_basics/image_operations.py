"""SciPy Image Operations

SciPy provides some basic functions to work with images. For example, it has functions to read images from disk into
numpy arrays, to write numpy arrays to disk as images, and to resize images. Here is a simple example
"""
from scipy.misc import imread, imsave, imresize

img = imread('assets/cat.jpg')
print(img.dtype, img.shape)

# We can tint the image by scaling each of the color channels by a different scale constant
img_tinted = img * [1, 0.50, 0.9] # Using broadcasting

img_tinted = imresize(img_tinted, (300, 300)) # Resize the tinted image to be 300 by 300 pixels

imsave('assets/cat_tinted.jpg', img_tinted)
