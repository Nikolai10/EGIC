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

"""Loss implementations."""

import tensorflow as tf
import os
import urllib.request
import numpy as np

from deeplab2 import common

_LPIPS_URL = "http://rail.eecs.berkeley.edu/models/lpips/net-lin_alex_v0.1.pb"


def ensure_lpips_weights_exist(weight_path_out):
    if os.path.isfile(weight_path_out):
        return
    print("Downloading LPIPS weights:", _LPIPS_URL, "->", weight_path_out)
    urllib.request.urlretrieve(_LPIPS_URL, weight_path_out)
    if not os.path.isfile(weight_path_out):
        raise ValueError(f"Failed to download LPIPS weights from {_LPIPS_URL} "
                         f"to {weight_path_out}. Please manually download!")


# Credits: https://github.com/tensorflow/compression/blob/master/models/hific/model.py
def compute_perceptual_loss(x, x_hat, lpips_path):
    # [0, 255] -> [-1, 1]
    x = (x - 127.5) / 127.5
    x_hat = (x_hat - 127.5) / 127.5

    # First the fake images, then the real! Otherwise no gradients.
    return LPIPSLoss(lpips_path)(x_hat, x)


# Credits: https://github.com/tensorflow/compression/blob/master/models/hific/model.py
class LPIPSLoss(object):
    """Calcualte LPIPS loss."""

    def __init__(self, weight_path):
        ensure_lpips_weights_exist(weight_path)

        def wrap_frozen_graph(graph_def, inputs, outputs):
            def _imports_graph_def():
                tf.graph_util.import_graph_def(graph_def, name="")

            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
            import_graph = wrapped_import.graph
            return wrapped_import.prune(
                tf.nest.map_structure(import_graph.as_graph_element, inputs),
                tf.nest.map_structure(import_graph.as_graph_element, outputs))

        # Pack LPIPS network into a tf function
        graph_def = tf.compat.v1.GraphDef()
        with open(weight_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        self._lpips_func = tf.function(
            wrap_frozen_graph(
                graph_def, inputs=("0:0", "1:0"), outputs="Reshape_10:0"))

    def __call__(self, fake_image, real_image):
        """Assuming inputs are in [-1, 1]."""

        # Move inputs to NCHW format.
        def _transpose_to_nchw(x):
            return tf.transpose(x, (0, 3, 1, 2))

        fake_image = _transpose_to_nchw(fake_image)
        real_image = _transpose_to_nchw(real_image)
        loss = self._lpips_func(fake_image, real_image)
        return tf.reduce_mean(loss)  # Loss is N111, take mean to get scalar.


def seg_loss(inputs, outputs, deeplab_loss):
    """
    Get the average per-batch segmentation loss see
    deeplab2.model.loss.loss_builder.py -> DeepLabFamilyLoss for more information

    :param inputs: GT
    :param outputs: predicted segments
    :param deeplab_loss: loss_builder.DeepLabFamilyLoss
    :return: weighted softmax cross entropy (segmentation) loss
    """

    loss_dict = deeplab_loss(inputs, outputs)
    # Average over the batch.
    average_loss_dict = {}
    for name, loss in loss_dict.items():
        averaged_loss = tf.reduce_mean(loss)
        average_loss_dict[name] = tf.where(tf.math.is_nan(averaged_loss), 0.0, averaged_loss)

    return average_loss_dict[common.TOTAL_LOSS]


def generate_labelmix(segments, fake_image, real_image):
    """See https://arxiv.org/pdf/2012.04781.pdf for more information."""

    segments = segments.numpy()
    all_classes, _ = tf.unique(tf.reshape(segments, [-1]))

    # generate binary mask M to mix a pair (x, x_hat) or real and fake images conditioned on the same label map
    for c in all_classes:
        segments[segments == c.numpy()] = np.random.randint(0, 2, 1)[0]  # torch.randint(0, 2, (1,))

    segments = segments.astype(np.float32)
    target_map = tf.expand_dims(segments, axis=-1)
    # LabelMix(x, x_hat, M) = M * x + (1-M) * x_hat
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map


def loss_labelmix(mask, output_D_mixed, output_D_fake, output_D_real):
    """See https://arxiv.org/pdf/2012.04781.pdf for more information."""

    mse = tf.keras.losses.MeanSquaredError()
    mixed_D_output = mask * output_D_real[common.PRED_SEMANTIC_LOGITS_KEY] + (1 - mask) * output_D_fake[
        common.PRED_SEMANTIC_LOGITS_KEY]
    return mse(output_D_mixed[common.PRED_SEMANTIC_LOGITS_KEY], mixed_D_output), mixed_D_output, output_D_mixed[
        common.PRED_SEMANTIC_LOGITS_KEY]
