import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image

from tensorflow.python.keras.preprocessing import image as kp_image


# module to work with images

def load_img(path_to_img):
    max_dim, img = 512, PIL.Image.open(path_to_img)
    scale = max_dim / max(img.size)
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), PIL.Image.ANTIALIAS)
    return np.expand_dims(kp_image.img_to_array(img), axis=0)


def imshow(img, title=None):  # to show an image
    out = np.squeeze(img, axis=0).astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)


def load_and_process_img(path_to_img):  # preprocessing for VGG training process
    return tf.keras.applications.vgg19.preprocess_input(load_img(path_to_img))


def deprocess_img(processed_img):  # inverse preprocessing in order to view the outputs
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, "input to deprocess image must be an image of correct dimension"
    if len(x.shape) != 3:
        raise ValueError("invalid input")
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')
