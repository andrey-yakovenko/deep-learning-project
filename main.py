import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display
import PIL.Image

# module with main methods
from methods import get_model, get_feature_representations, compute_grads

# module with methods to work with losses
from loss import gram_matrix

# module with methods to work with images
from img import load_img, imshow, load_and_process_img, deprocess_img


# module to run the program

# to change the images to work with you need:
# 1. to add style and content images to /img
# 2. change the names below
# 3. select the number of iterations below
content_path, style_path, num_iterations = 'img/content2.jpg', 'img/style2.jpg', 1000

content_weight, style_weight = 1e3, 1e-2
model = get_model()  # getting pretrained model
for layer in model.layers:
    layer.trainable = False
# getting the style and content feature representations
style_features, content_features = get_feature_representations(model, content_path, style_path)
init_image = tf.Variable(load_and_process_img(content_path), dtype=tf.float32)  # setting initial image
opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
best_loss, best_img = float('inf'), None  # storing best result
cfg = {'model': model, 'loss_weights': (style_weight, content_weight), 'init_image': init_image,
       'gram_style_features': [gram_matrix(style_feature) for style_feature in style_features],
       'content_features': content_features}  # configs for grads

num_rows, num_cols = 2, 5  # for displaying later
global_start, display_interval = time.time(), num_iterations / (num_rows * num_cols)
norm_means = np.array([103.939, 116.779, 123.68])
min_vals, max_vals = - norm_means, 255 - norm_means
imgs = []
for i in range(num_iterations):
    print("Iteration:", i)
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time()
    if loss < best_loss:
        best_loss = loss  # updating best loss
        best_img = deprocess_img(init_image.numpy())  # updating best image
    if i % display_interval == 0:  # getting numpy array for images
        plot_img = deprocess_img(init_image.numpy())
        imgs.append(plot_img)
        IPython.display.clear_output(wait=True)
        IPython.display.display_png(PIL.Image.fromarray(plot_img))
print('\nTotal time: {:.4f}s'.format(time.time() - global_start))

# displaying results
IPython.display.clear_output(wait=True)
plt.figure(figsize=(14, 4))
for i, img in enumerate(imgs):
    plt.subplot(num_rows, num_cols, i+1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
PIL.Image.fromarray(best_img)
plt.figure(figsize=(10, 5))
content, style = load_img(content_path), load_img(style_path)
plt.subplot(1, 2, 1)
imshow(content, 'Content Image')
plt.subplot(1, 2, 2)
imshow(style, 'Style Image')
plt.figure(figsize=(10, 10))
plt.imshow(best_img)
plt.title('Output Image')
plt.show()
