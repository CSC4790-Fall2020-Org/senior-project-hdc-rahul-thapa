#!/usr/bin/env python
# coding: utf-8

# In[131]:


from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import interpolation
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageEnhance
from mpl_toolkits.axes_grid1 import AxesGrid
from mnist import MNIST

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math


# In[132]:


def skew(image, loc=12, scale=1):
    image = image.reshape(28, 28)
    h, l = image.shape
    distortion = np.random.normal(loc=12, scale=1)

    def mapping(point):
        x, y = point
        dec = (distortion*(x-h))/h
        return x, y+dec+5
    return interpolation.geometric_transform(
        image, mapping, (h, l), order=5, mode='nearest')

def rotate(image, d):
    """Rotate the image by d/180 degrees."""
    center = 0.5*np.array(image.shape)
    rot = np.array([[np.cos(d), np.sin(d)],[-np.sin(d), np.cos(d)]])
    offset = (center-center.dot(rot)).dot(np.linalg.inv(rot))
    return interpolation.affine_transform(
        image,
        rot,
        order=2,
        offset=-offset,
        cval=0.0,
        output=np.float32)

def noise(image, n=100):
    """Add noise by randomly changing n pixels"""
    indices = np.random.random(size=(n, 2))*28
    image = image.copy()
    for x, y in indices:
        x, y = int(x), int(y)
        image[x][y] = 0
    return image

def brightness(image, factor=0.8):
    return image * factor

def elastic_transform(image, alpha=36, sigma=5, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    
    :param image: a 28x28 image
    :param alpha: scale for filter
    :param sigma: the standard deviation for the gaussian
    :return: distorted 28x28 image
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)


def draw_examples_with_perturbation(examples, f):
    """Draw examples with provided perturbation f
    
    :param examples: list of examples
    :param f: transformation function with takes a 28x28 image
    and returns a 28x28 image
    """
    examples = [(e, n) for n, e in enumerate(examples)]
    grid = AxesGrid(plt.figure(figsize=(8,15)), 141,  # similar to subplot(141)
                        nrows_ncols=(len(examples), 2),
                        axes_pad=0.05,
                        label_mode="1",
                        )

    for examplenum,num in examples:
        image = X_train[examplenum].reshape(28,28)
        im = grid[2*num].imshow(image)
        im2 = grid[2*num+1].imshow(f(image))
        
def load_dataset(s="data"):
    mndata = MNIST('../%s/'%s)
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


# In[ ]:




