import numpy as np
from keras.datasets import mnist


def randomize_positions(images, image_shape):
    # Downscale by factor of 2.
    images = images[:,::2,::2]

    # Pad to image_shape with zeros.
    padded = np.zeros((images.shape[0], image_shape[0], image_shape[1]))
    w, h = images.shape[1:]
    for i, image in enumerate(images):
        x = np.random.randint(0, padded.shape[1] - w)
        y = np.random.randint(0, padded.shape[2] - h)

        padded[i, x : x+w, y : y+h] = image

    padded = padded.astype(float) / 255.0
    return padded


def weighted_average(data, weights):
    return np.sum(data[None,:,:] * weights, axis=(1, 2)) / np.sum(weights, axis=(1, 2))


def get_positions(images):
    # Position is average of all pixel positions weighted by pixel intensity.
    x, y = np.meshgrid(np.arange(images.shape[2]), np.arange(images.shape[1]))
    pos_x = weighted_average(x, images)
    pos_y = weighted_average(y, images)
    return np.stack([pos_x, pos_y], axis=-1)


def process_images(images, image_shape):
    padded = randomize_positions(images, image_shape)
    return padded, get_positions(padded)


def generate_dataset(image_shape):
    (train_images, _), (test_images, _) = mnist.load_data()

    np.random.shuffle(train_images)
    np.random.shuffle(test_images)

    return process_images(train_images, image_shape), process_images(test_images, image_shape)
