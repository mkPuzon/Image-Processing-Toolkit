'''img_loader.py

Loads in image data--either batch or individual images.

Maddie Puzon
Created: 03/20/2025 
Last updated: 03/20/2025

TODO: Add different standardization methods
'''
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

def load_img_as_np(path_to_img, dtype=np.float64):
    '''
    Loads in an image from a path and returns it as a numpy array.
    '''
    return np.array(Image.open(path_to_img), dtype=dtype)


def standardize(img):
    '''
    Standardizes the feature values to between 0.0 and 1.0 for a given image. Works for both single images and batches.
    
    Parameters:
        img: The image to standardize.
    '''
    # if single image
    if len(img.shape) == 3:
        Iy, Ix, D = img.shape
        # flatten to shape=(h*w*rgb)
        flat_imgs = np.reshape(img, [np.prod(img.shape)])
        # standardize pixels across axis 0
        print(f"Initial data range: ({flat_imgs.min():.2f}, {flat_imgs.max():.2f})")
        norm_imgs = (flat_imgs / 255)
        print(f"Normalized data range: ({norm_imgs.min():.2f}, {norm_imgs.max():.2f})")
        # reshape into img dims
        reshaped_img = np.reshape(norm_imgs, [Iy, Ix, D])
        return reshaped_img
    # if batch of images
    elif len(img.shape) == 4: 
        B, Iy, Ix, D = img.shape
        # flatten to shape=(num_imgs, h*w*rgb)
        flat_imgs = np.reshape(img, [B, Iy * Ix * D])
        # standardize pixels across axis 0
        print(f"Initial data range: ({flat_imgs.min():.2f}, {flat_imgs.max():.2f})")
        norm_imgs = (flat_imgs / 255)
        print(f"Normalized data range: ({norm_imgs.min():.2f}, {norm_imgs.max():.2f})")
        # reshape into img dims
        reshaped_imgs = np.reshape(norm_imgs, [B, Iy, Ix, D])
        return reshaped_imgs
    else:
        raise ValueError(f"Cannot standardize img with dims {img.shape}. standardize(img) expects img to be of shape (B, Iy, Ix, D) or (Iy, Ix, D).")
    
def show_img(img, title):
    '''
    Displays a single image with the given title.
    
    Parameters:
        img: tf.tensor, the image to display; expects shape=(Iy, Ix, n_color_chans).
        title: (str) text to caption image.
    '''
    plt.imshow(img)
    # turn off gridlines
    plt.grid(False)
    # turn of ticks on axes
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    pass