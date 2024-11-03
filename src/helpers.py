# Copyright 2024 Nikolai Körber. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Some helper functionality."""

import tensorflow as tf
import numpy as np
import collections
from coco_utils import create_coco_label_colormap

Nodes = collections.namedtuple(
    "Nodes",  # Expected ranges for RGB:
    ["input_image",  # [0, 255]
     "input_image_scaled",  # [0, 1]
     "reconstruction",  # [0, 255]
     "reconstruction_scaled",  # [0, 1]
     "latent_quantized"])  # Latent post-quantization.


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
    """Saves an image to a PNG file."""
    string = tf.image.encode_png(image)
    tf.io.write_file(filename, string)


def perturb_color(color, noise, used_colors, max_trials=50, random_state=None):
    """
    Pertrubs the color with some noise.
    
    If `used_colors` is not None, we will return the color that has
    not appeared before in it.
    
    :param color: A numpy array with three elements [R, G, B].
    :param noise: Integer, specifying the amount of perturbing noise (in uint8 range).
    :param used_colors: A set, used to keep track of used colors.
    :param max_trials: An integer, maximum trials to generate random color.
    :param random_state: An optional np.random.RandomState. If passed, will be used to
        generate random numbers.
    :return: A perturbed color that has not appeared in used_colors.
    """

    if random_state is None:
        random_state = np.random

    for _ in range(max_trials):
        random_color = color + random_state.randint(
            low=-noise, high=noise + 1, size=3)
        random_color = np.clip(random_color, 0, 255)

        if tuple(random_color) not in used_colors:
            used_colors.add(tuple(random_color))
            return random_color

    print('Max trial reached and duplicate color will be used. Please consider '
          'increase noise in `perturb_color()`.')
    return random_color


def colorize_segments(segments):
    """Colorize segments to coco color theme"""
    color_map = create_coco_label_colormap()
    segments = segments.numpy()
    return color_map[segments[0, :, :]]


def pad(input_image, factor=256):
    """Pad `input_image such that H and W are divisible by `factor`."""
    with tf.name_scope("pad"):
        height, width = tf.shape(input_image)[0], tf.shape(input_image)[1]
        pad_height = (factor - (height % factor)) % factor
        pad_width = (factor - (width % factor)) % factor
        return tf.pad(input_image,
                      [[0, pad_height], [0, pad_width], [0, 0]], "REFLECT")


def learning_rate_decay(epoch, learning_rate):
    # hardcoded for now -> shift to config
    if epoch >= 180:
        new_learning_rate = 1e-5
        return new_learning_rate
    return learning_rate


def learning_rate_identity(epoch, learning_rate):
    return learning_rate
