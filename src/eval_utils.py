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

"""Evaluate dataset (bpp, PSNR)."""

import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
import collections
import os
import glob

from helpers import pad, write_png

# How many dataset preprocessing processes to use.
DATASET_NUM_PARALLEL = 8

# How many batches to prefetch.
DATASET_PREFETCH_BUFFER = 20


def evaluate_ds(args, orp=False):
    """Evaluate dataset."""
    out_dir = args.out_dir
    images_glob = args.images_glob
    os.makedirs(out_dir, exist_ok=True)

    accumulated_metrics = collections.defaultdict(list)

    # Load model and use it to compress the image.
    model = tf.keras.models.load_model(args.model_path)
    if orp:
        model.alpha.assign(args.alpha)

    dataset = build_input(
        batch_size=1,
        images_glob=images_glob)

    image_names = get_image_names(images_glob)

    for idx, elem in enumerate(dataset):
        x = tf.squeeze(elem['input_image'], axis=0)
        x_padded = pad(x, factor=256)

        print('processing {}...'.format(image_names[idx] + '.png'))

        # compressing...
        tensors = model.compress(x_padded)
        # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        packed.pack(tensors)

        # decompressing
        x_hat = model.decompress(*tensors)

        # undo padding
        height, width = tf.shape(x)[0], tf.shape(x)[1]
        x_hat = x_hat[:height, :width, :]

        # Cast to float in order to compute metrics.
        x = tf.cast(x, tf.float32)
        x_hat = tf.cast(x_hat, tf.float32)
        psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))

        # The actual bits per pixel including entropy coding overhead.
        num_pixels = tf.reduce_prod(tf.shape(x_padded)[:-1])
        bpp = len(packed.string) * 8 / num_pixels

        metrics = {'psnr': psnr,
                   'bpp_real': bpp}

        for metric, value in metrics.items():
            accumulated_metrics[metric].append(value)

        write_png(out_dir + image_names[idx] + '_inp.png', tf.cast(x, tf.uint8))
        write_png(out_dir + image_names[idx] + '_otp_{}.png'.format(bpp), tf.cast(x_hat, tf.uint8))

    with open(os.path.join(out_dir, "results.txt"), "w") as text_file:
        text_file.write('\n'.join(f'{metric}: {np.mean(values)}'
                                  for metric, values in accumulated_metrics.items()))
    print('Done!')


def get_image_names(images_glob):
    if not images_glob:
        return {}
    return {i: os.path.splitext(os.path.basename(p))[0]
            for i, p in enumerate(sorted(glob.glob(images_glob)))}


def build_input(batch_size, images_glob):
    """Build input dataset."""
    if not images_glob:
        raise ValueError("Need images_glob")

    def batch_to_dict(batch):
        return dict(input_image=batch)

    dataset = get_dataset(batch_size, images_glob)
    return dataset.map(batch_to_dict)


def get_dataset(batch_size, images_glob):
    """Build TFDS dataset."""
    with tf.name_scope("tfds"):
        images = sorted(glob.glob(images_glob))
        filenames = tf.data.Dataset.from_tensor_slices(images)
        dataset = filenames.map(lambda x: tf.image.decode_png(tf.io.read_file(x)))

        def preprocess(features):
            image = features
            return image

        dataset = dataset.map(
            preprocess, num_parallel_calls=DATASET_NUM_PARALLEL)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=DATASET_PREFETCH_BUFFER)

        return dataset
