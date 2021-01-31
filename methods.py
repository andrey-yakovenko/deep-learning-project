import tensorflow as tf
from tensorflow.python.keras import models

from loss import compute_loss
from img import load_and_process_img


# module with main methods

content_layers = ['block5_conv2']  # content layer where will pull our feature maps
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']  # style layers
num_content_layers, num_style_layers = len(content_layers), len(style_layers)


def get_model():  # loading the VGG19 model
    # take input image and return the outputs from intermediate layers from the VGG model
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')  # loading pretrained VGG (imagenet)
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)


def get_feature_representations(model, content_path, style_path):
    # to compute content and style feature representations
    content_image, style_image = load_and_process_img(content_path), load_and_process_img(style_path)
    style_outputs, content_outputs = model(style_image), model(content_image)
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def compute_grads(cfg):  # computing the gradients
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    return tape.gradient(all_loss[0], cfg['init_image']), all_loss
