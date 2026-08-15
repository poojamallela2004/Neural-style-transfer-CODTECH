import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array


from google.colab import drive
drive.mount('/content/drive')

import os
print(os.listdir())   # lists files in the current folder


def preprocess_image(image_path, target_size=(400, 400)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

content_image = preprocess_image("/content/drive/MyDrive/Colab Notebooks/CodtechAIProject/content.jpg")
style_image = preprocess_image("/content/drive/MyDrive/Colab Notebooks/CodtechAIProject/style.jpg")


content_display = load_img("/content/drive/MyDrive/Colab Notebooks/CodtechAIProject/content.jpg", target_size=(400, 400))
style_display = load_img("/content/drive/MyDrive/Colab Notebooks/CodtechAIProject/style.jpg", target_size=(400, 400))

# Plot side by side
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(content_display)
plt.title("Content Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(style_display)
plt.title("Style Image")
plt.axis("off")

plt.show()

# Load VGG19 pretrained on ImageNet
vgg_model = vgg19.VGG19(weights="imagenet", include_top=False)

# Select layers for content and style extraction
content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1', 'block5_conv1']


outputs = [vgg_model.get_layer(content_layer).output] + [vgg_model.get_layer(layer).output for layer in style_layers]
feature_extractor = tf.keras.Model([vgg_model.input], outputs)

def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations


def compute_loss(content_features, style_features, combination_features):
    # Content Loss
    content_loss = tf.reduce_mean((combination_features[0] - content_features[0]) ** 2)

    # Style Loss
    style_loss = 0
    for cf, sf in zip(combination_features[1:], style_features[1:]):
        gram_c = gram_matrix(cf)
        gram_s = gram_matrix(sf)
        style_loss += tf.reduce_mean((gram_c - gram_s) ** 2)
    style_loss /= len(style_features[1:])

    # Total Loss (weighted sum)
    total_loss = 0.7 * content_loss + 0.3 * style_loss
    return total_loss


# Initialize generated image as a copy of content image
generated_image = tf.Variable(content_image, dtype=tf.float32)

optimizer = tf.optimizers.Adam(learning_rate=5.0)

@tf.function
def train_step(content_features, style_features):
    with tf.GradientTape() as tape:
        combination_features = feature_extractor(generated_image)
        loss = compute_loss(content_features, style_features, combination_features)
    grads = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(grads, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, -128.0, 128.0))
    return loss


content_features = feature_extractor(content_image)
style_features = feature_extractor(style_image)

epochs = 500
for i in range(epochs):
    loss = train_step(content_features, style_features)
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.numpy()}")


def deprocess_image(x):
    x = x.reshape((400, 400, 3))
    x = x - x.mean()
    x = x / (x.std() + 1e-5)
    x = x * 64 + 128
    x = np.clip(x, 0, 255).astype("uint8")
    return x

final_img = deprocess_image(generated_image.numpy())
plt.imshow(final_img)
plt.axis("off")
plt.show()
