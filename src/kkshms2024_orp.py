# Copyright 2024 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/tensorflow/compression/blob/master/models/ms2020.py,
# Copyright 2020 Google LLC.
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

"""Nonlinear transform coder with hyperprior for RGB images.

This is the official implementaiton of EGIC published in:
N. Körber and E. Kromer and A. Siebert and S. Hauke and D. Mueller-Gritschneder and B. Schuller:
"EGIC: Enhanced Low-Bit-Rate Generative Image Compression Guided by Semantic Segmentation"
European Conference on Computer Vision (ECCV), 2024
https://arxiv.org/abs/2309.03244

This work is based on the image compression model published in:
D. Minnen and S. Singh:
"Channel-wise autoregressive entropy models for learned image compression"
Int. Conf. on Image Compression (ICIP), 2020
https://arxiv.org/abs/2007.08739

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.

This script requires TFC v2 (`pip install tensorflow-compression==2.*`).

This script provides training/ compression functionality (ORP).
"""

import sys
import os

from config import ConfigEGIC as cfg_egic

sys.path.append(cfg_egic.global_path)
sys.path.append(cfg_egic.global_path + '/swin-transformers-tf')

from deeplab2 import config_pb2
from deeplab2.trainer import runner_utils
from deeplab2 import common

from google.protobuf import text_format

import argparse
import sys
from absl import app
from absl.flags import argparse_flags
import tensorflow as tf
import tensorflow_compression as tfc

from config import ConfigGa as cfg_ga
from config import ConfigGs as cfg_gs
from config import ConfigHa as cfg_ha
from config import ConfigHs as cfg_hs
from config import ConfigChARM as cfg_charm

from helpers import Nodes, pad, read_png, write_png
from loss import compute_perceptual_loss

from archs import AnalysisTransform, SynthesisTransformORP, HyperAnalysisTransform, HyperSynthesisTransform, \
    SliceTransform

from eval_utils import evaluate_ds

tf.config.experimental.enable_tensor_float_32_execution(False)


class SetWeightsCallback(tf.keras.callbacks.Callback):

    def __init__(self, model):
        super(SetWeightsCallback, self).__init__()
        self.model = model

    def on_train_begin(self, model, logs=None):
        print("Callback is called at the beginning of training.")
        weights_B = self.model.synthesis_transform.get_layer(name='basic_layer_7').get_weights()
        self.model.synthesis_transform.cond_model.get_layer(name='basic_layer_8').set_weights(weights_B)


class KKSHMS2024Model(tf.keras.Model):
    """Main model class."""

    def __init__(self, lmbda, sum_wr, latent_depth, hyperprior_depth,
                 num_slices, max_support_slices, num_scales, scale_min, scale_max):

        super().__init__()
        self.sum_wr = sum_wr
        self.lmbda = lmbda
        self.num_scales = num_scales
        self.num_slices = num_slices
        self.slice_size = latent_depth // self.num_slices
        self.max_support_slices = max_support_slices
        offset = tf.math.log(scale_min)
        factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
                num_scales - 1.)
        self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
        self.analysis_transform = AnalysisTransform(cfg_ga)
        self.synthesis_transform = SynthesisTransformORP(cfg_gs)
        self.hyper_analysis_transform = HyperAnalysisTransform(cfg_ha)
        self.hyper_synthesis_mean_scale_transform = HyperSynthesisTransform(cfg_hs)
        self.cc_mean_transforms = [
            SliceTransform(cfg_charm, latent_depth, num_slices, idx) for idx in range(num_slices)]
        self.cc_scale_transforms = [
            SliceTransform(cfg_charm, latent_depth, num_slices, idx) for idx in range(num_slices)]
        self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=[hyperprior_depth])
        self.alpha = tf.Variable(0.0)  # default enabled
        self.build((None, None, None, 3))
        # The call signature of decompress() depends on the number of slices, so we
        # need to compile the function dynamically.
        self.decompress = tf.function(
            input_signature=3 * [tf.TensorSpec(shape=(2,), dtype=tf.int32)] +
                            (num_slices + 1) * [tf.TensorSpec(shape=(1,), dtype=tf.string)]
        )(self.decompress)

    def call(self, x, training):
        """Computes rate and distortion losses."""

        x = tf.cast(x, self.compute_dtype)  # TODO(jonycgn): Why is this necessary?
        h, w, c = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)
        y_shape = (h // 16, w // 16, cfg_ga.embed_dim[-1])

        z = self.hyper_analysis_transform(y)
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[1:-1]), tf.float32)

        # Build the entropy model for the hyperprior (z).
        em_z = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=False,
            offset_heuristic=False)

        # When training, z_bpp is based on the noisy version of z (z_tilde).
        _, z_bits = em_z(z, training=training)
        z_bpp = tf.reduce_mean(z_bits) / num_pixels
        z_bits_mean = tf.reduce_mean(z_bits)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_hat = em_z.quantize(z)

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        latent_means_scales = self.hyper_synthesis_mean_scale_transform(z_hat)
        latent_means, latent_scales = tf.split(latent_means_scales, 2, axis=-1)

        # Build a conditional entropy model for the slices.
        em_y = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
            coding_rank=3, compression=False)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_slices = tf.split(y, self.num_slices, axis=-1)
        y_hat_slices = []
        y_bpps = []
        y_bits_arr = []

        for slice_index, y_slice in enumerate(y_slices):
            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                              y_hat_slices[:self.max_support_slices])

            start_index = slice_index * self.slice_size
            end_index = slice_index * self.slice_size + self.slice_size
            latent_means_slice = latent_means[:, :, :, start_index:end_index]
            latent_scales_slice = latent_scales[:, :, :, start_index:end_index]

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means_slice] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat([latent_scales_slice] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            _, slice_bits = em_y(y_slice, sigma, loc=mu, training=training)
            slice_bpp = tf.reduce_mean(slice_bits) / num_pixels
            y_bpps.append(slice_bpp)

            y_bits_mean = tf.reduce_mean(slice_bits)
            y_bits_arr.append(y_bits_mean)

            # For the synthesis transform, use rounding. Note that quantize()
            # overrides the gradient to create a straight-through estimator.
            y_hat_slice = em_y.quantize(y_slice, loc=mu)
            y_hat_slices.append(y_hat_slice)

        # Merge slices and generate the image reconstruction.
        y_hat = tf.concat(y_hat_slices, axis=-1)
        _, x_hat = self.synthesis_transform([y_hat, 0.0])

        # Total bpp is sum of bpp from hyperprior and all slices.
        total_bpp = tf.add_n(y_bpps + [z_bpp])
        total_bits = tf.add_n(y_bits_arr + [z_bits_mean])

        # Mean squared error across pixels.
        # Don't clip or round pixel values while training.
        # inspiration from: https://github.com/tensorflow/compression/blob/d672ad355f15a4e33064bb112cf903398f245024/models/hific/model.py#L75
        distortion_loss = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        weighted_distortion_loss = 0.1 * 2. ** -5 * 0.75 * distortion_loss
        weighted_distortion_loss = tf.cast(weighted_distortion_loss, total_bpp.dtype)

        # LPIPS _lpips_loss_weight -> hardcoded here
        perceptual_loss = compute_perceptual_loss(x, x_hat, cfg_egic.lpips_path)
        weighted_perceptual_loss = 1 * perceptual_loss
        weighted_perceptual_loss = tf.cast(weighted_perceptual_loss, total_bpp.dtype)

        # Calculate and return the rate-distortion loss: R + lambda * D.
        # loss = total_bpp + self.lmbda * mse
        # we follow the logic of Zhu et al.
        # loss = weighted_distortion_loss + weighted_perceptual_loss + self.lmbda * total_bits
        loss = weighted_distortion_loss + weighted_perceptual_loss + self.lmbda * total_bpp

        input_image = x
        input_image_scaled = x / 255.
        reconstruction = x_hat
        reconstruction_scaled = reconstruction / 255.
        latent_quantized = y_hat

        nodes = Nodes(input_image, input_image_scaled, reconstruction, reconstruction_scaled, latent_quantized)
        return nodes, loss, total_bpp, distortion_loss, weighted_distortion_loss, weighted_perceptual_loss

    def train_step(self, x):

        x = x[common.IMAGE]
        with tf.GradientTape() as tape:
            _, loss, bpp, mse, weighted_mse, weighted_lpips = self(x, training=True)

        # only finetune ORP model
        variables = self.synthesis_transform.cond_model.trainable_variables
        gradients = tape.gradient(mse, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        self.weighted_mse.update_state(weighted_mse)
        self.weighted_lpips.update_state(weighted_lpips)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.weighted_mse, self.weighted_lpips]}

    def test_step(self, x):

        x = x[common.IMAGE]
        x = tf.image.random_crop(value=x, size=(tf.shape(x)[0], 256, 256, 3))

        _, loss, bpp, mse, weighted_mse, weighted_lpips = self(x, training=False)
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        self.weighted_mse.update_state(weighted_mse)
        self.weighted_lpips.update_state(weighted_lpips)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.weighted_mse, self.weighted_lpips]}

    def predict_step(self, x):
        raise NotImplementedError("Prediction API is not supported.")

    def compile(self, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.bpp = tf.keras.metrics.Mean(name="bpp")
        self.mse = tf.keras.metrics.Mean(name="mse")
        self.weighted_mse = tf.keras.metrics.Mean(name="weighted_mse")
        self.weighted_lpips = tf.keras.metrics.Mean(name="weighted_lpips")

    def fit(self, *args, **kwargs):
        retval = super().fit(*args, **kwargs)
        # After training, fix range coding tables.
        self.em_z = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=True,
            offset_heuristic=False)
        self.em_y = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
            coding_rank=3, compression=True)
        return retval

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
    ])
    def compress(self, x):
        """Compresses an image."""
        # Add batch dimension and cast to float.
        x = tf.expand_dims(x, 0)
        x = tf.cast(x, dtype=self.compute_dtype)

        y_strings = []

        x_shape = tf.shape(x)[1:-1]

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)
        y_shape = tf.shape(y)[1:-1]

        z = self.hyper_analysis_transform(y)
        z_shape = tf.shape(z)[1:-1]

        z_string = self.em_z.compress(z)
        z_hat = self.em_z.decompress(z_string, z_shape)

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        latent_means_scales = self.hyper_synthesis_mean_scale_transform(z_hat)
        latent_means, latent_scales = tf.split(latent_means_scales, 2, axis=-1)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_slices = tf.split(y, self.num_slices, axis=-1)
        y_hat_slices = []
        for slice_index, y_slice in enumerate(y_slices):
            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                              y_hat_slices[:self.max_support_slices])

            start_index = slice_index * self.slice_size
            end_index = slice_index * self.slice_size + self.slice_size
            latent_means_slice = latent_means[:, :, :, start_index:end_index]
            latent_scales_slice = latent_scales[:, :, :, start_index:end_index]

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means_slice] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat([latent_scales_slice] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            slice_string = self.em_y.compress(y_slice, sigma, mu)
            y_strings.append(slice_string)
            y_hat_slice = self.em_y.decompress(slice_string, sigma, mu)

            y_hat_slices.append(y_hat_slice)

        return (x_shape, y_shape, z_shape, z_string) + tuple(y_strings)

    def decompress(self, x_shape, y_shape, z_shape, z_string, *y_strings):
        """Decompresses an image."""
        assert len(y_strings) == self.num_slices

        z_hat = self.em_z.decompress(z_string, z_shape)
        _, h, w, c = z_hat.shape

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        latent_means_scales = self.hyper_synthesis_mean_scale_transform(z_hat)
        latent_means, latent_scales = tf.split(latent_means_scales, 2, axis=-1)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_hat_slices = []
        for slice_index, y_string in enumerate(y_strings):
            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                              y_hat_slices[:self.max_support_slices])

            start_index = slice_index * self.slice_size
            end_index = slice_index * self.slice_size + self.slice_size
            latent_means_slice = latent_means[:, :, :, start_index:end_index]
            latent_scales_slice = latent_scales[:, :, :, start_index:end_index]

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means_slice] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat([latent_scales_slice] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            y_hat_slice = self.em_y.decompress(y_string, sigma, loc=mu)

            y_hat_slices.append(y_hat_slice)

        # Merge slices and generate the image reconstruction.
        y_hat = tf.concat(y_hat_slices, axis=-1)

        x_hat, _ = self.synthesis_transform([y_hat, self.alpha])

        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
        # Then cast back to 8-bit integer.
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)


def learning_rate_decay(epoch, learning_rate):
    if epoch >= 480:
        new_learning_rate = 1e-5
        return new_learning_rate
    return learning_rate


def train(args):
    """Instantiates and trains the model."""
    if args.precision_policy:
        tf.keras.mixed_precision.set_global_policy(args.precision_policy)
    if args.check_numerics:
        tf.debugging.enable_check_numerics()

    # logging.info('Reading the config file.')
    with tf.io.gfile.GFile(cfg_egic.proto_config, 'r') as proto_file:
        config = text_format.ParseLines(proto_file, config_pb2.ExperimentOptions())
    print(config)

    train_dataset = runner_utils.create_dataset(
        config.train_dataset_options,
        is_training=True,
        only_semantic_annotations=False)

    validation_dataset = runner_utils.create_dataset(
        config.eval_dataset_options,
        is_training=False,
        only_semantic_annotations=False)

    sum_wr = tf.summary.create_file_writer(args.train_path)

    model = KKSHMS2024Model(
        args.lmbda, sum_wr, args.latent_depth, args.hyperprior_depth, args.num_slices,
        args.max_support_slices, args.num_scales, args.scale_min, args.scale_max)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr)
    )

    validation_dataset = validation_dataset.take(args.max_validation_steps)

    model.fit(
        train_dataset.prefetch(16),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=validation_dataset.cache(),
        validation_freq=1,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(
                log_dir=args.train_path,
                histogram_freq=1, update_freq="epoch"),
            tf.keras.callbacks.BackupAndRestore(args.train_path, delete_checkpoint=False),
            tf.keras.callbacks.LearningRateScheduler(learning_rate_decay),
            SetWeightsCallback(model),
        ],
        verbose=int(args.verbose),
    )
    model.save(args.model_path)


def compress(args):
    """Compresses an image."""
    # Load model and use it to compress the image.
    model = tf.keras.models.load_model(args.model_path)
    model.alpha.assign(args.alpha)

    x = read_png(args.input_file)
    factor = 256
    print("Padding to {}".format(factor))
    x_padded = pad(x, factor=factor)

    tensors = model.compress(x_padded)

    # Write a binary file with the shape information and the compressed string.
    packed = tfc.PackedTensors()
    packed.pack(tensors)
    with open(args.output_file, "wb") as f:
        f.write(packed.string)

    # If requested, decompress the image and measure performance.
    if args.verbose:
        x_hat = model.decompress(*tensors)

        # undo padding
        height, width = tf.shape(x)[0], tf.shape(x)[1]
        x_hat = x_hat[:height, :width, :]

        # Cast to float in order to compute metrics.
        x = tf.cast(x, tf.float32)
        x_hat = tf.cast(x_hat, tf.float32)
        mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
        msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
        msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

        # The actual bits per pixel including entropy coding overhead.
        num_pixels = tf.reduce_prod(tf.shape(x_padded)[:-1])
        bpp = len(packed.string) * 8 / num_pixels

        print(f"Mean squared error: {mse:0.4f}")
        print(f"PSNR (dB): {psnr:0.2f}")
        print(f"Multiscale SSIM: {msssim:0.4f}")
        print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
        print(f"Bits per pixel: {bpp:0.4f}")

        pathname, _ = os.path.splitext(args.output_file)
        write_png(pathname + '.png', tf.cast(x, tf.uint8))
        write_png(pathname + '_hat.png', tf.cast(x_hat, tf.uint8))


def decompress(args):
    """Decompresses an image."""
    # Load the model and determine the dtypes of tensors required to decompress.
    model = tf.keras.models.load_model(args.model_path)
    model.alpha.assign(args.alpha)
    dtypes = [t.dtype for t in model.decompress.input_signature]

    # Read the shape information and compressed string from the binary file,
    # and decompress the image using the model.
    with open(args.input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    tensors = packed.unpack(dtypes)
    x_hat = model.decompress(*tensors)

    # Write reconstructed image out as a PNG file.
    write_png(args.output_file, x_hat)


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report progress and metrics when training or compressing.")
    parser.add_argument(
        "--model_path", default="res/kkshms2024",
        help="Path where to save/load the trained model.")
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="alpha conditioning as described in the paper")
    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
             "to train) a new model. 'compress' reads an image file (lossless "
             "PNG format) and writes a compressed binary file. 'decompress' "
             "reads a binary file and reconstructs the image (in PNG format). "
             "input and output filenames need to be provided for the latter "
             "two options. Invoke '<command> -h' for more information.")

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model. Note that this "
                    "model trains on a continuous stream of patches drawn from "
                    "the training image dataset. An epoch is always defined as "
                    "the same number of batches given by --steps_per_epoch. "
                    "The purpose of validation is mostly to evaluate the "
                    "rate-distortion performance of the model using actual "
                    "quantization rather than the differentiable proxy loss. "
                    "Note that when using custom training images, the validation "
                    "set is simply a random sampling of patches from the "
                    "training set.")
    train_cmd.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument(
        "--lr", type=float, default=1e-4, dest="lr",
        help="Learning rate.")
    train_cmd.add_argument(
        "--train_glob", type=str, default=None,
        help="Glob pattern identifying custom training data. This pattern must "
             "expand to a list of RGB images in PNG format. If unspecified, the "
             "CLIC dataset from TensorFlow Datasets is used.")
    train_cmd.add_argument(
        "--num_filters", type=int, default=192,
        help="Number of filters per layer.")
    train_cmd.add_argument(
        "--latent_depth", type=int, default=320,
        help="Number of filters of the last layer of the analysis transform.")
    train_cmd.add_argument(
        "--hyperprior_depth", type=int, default=192,
        help="Number of filters of the last layer of the hyper-analysis "
             "transform.")
    train_cmd.add_argument(
        "--num_slices", type=int, default=10,
        help="Number of channel slices for conditional entropy modeling.")
    train_cmd.add_argument(
        "--max_support_slices", type=int, default=5,
        help="Maximum number of preceding slices to condition the current slice "
             "on. See Appendix C.1 of the paper for details.")
    train_cmd.add_argument(
        "--num_scales", type=int, default=64,
        help="Number of Gaussian scales to prepare range coding tables for.")
    train_cmd.add_argument(
        "--scale_min", type=float, default=.11,
        help="Minimum value of standard deviation of Gaussians.")
    train_cmd.add_argument(
        "--scale_max", type=float, default=256.,
        help="Maximum value of standard deviation of Gaussians.")
    train_cmd.add_argument(
        "--train_path", default="res/train_kkshms2024",
        help="Path where to log training metrics for TensorBoard and back up "
             "intermediate model checkpoints.")
    train_cmd.add_argument(
        "--batchsize", type=int, default=16,
        help="Batch size for training and validation.")
    train_cmd.add_argument(
        "--epochs", type=int, default=1000,
        help="Train up to this number of epochs. (One epoch is here defined as "
             "the number of steps given by --steps_per_epoch, not iterations "
             "over the full training dataset.)")
    train_cmd.add_argument(
        "--steps_per_epoch", type=int, default=1000,
        help="Perform validation and produce logs after this many batches.")
    train_cmd.add_argument(
        "--max_validation_steps", type=int, default=16,
        help="Maximum number of batches to use for validation. If -1, use one "
             "patch from each image in the training set.")
    train_cmd.add_argument(
        "--preprocess_threads", type=int, default=16,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    train_cmd.add_argument(
        "--precision_policy", type=str, default=None,
        help="Policy for `tf.keras.mixed_precision` training.")
    train_cmd.add_argument(
        "--check_numerics", action="store_true",
        help="Enable TF support for catching NaN and Inf in tensors.")

    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.")

    # 'decompress' subcommand.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file, reconstructs the image, and writes back "
                    "a PNG file.")

    # 'evaluate_ds' subcommand.
    evaluate_ds_cmd = subparsers.add_parser(
        "evaluate_ds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="evaluates the compression performance on a whole dataset")
    evaluate_ds_cmd.add_argument(
        "--images_glob", help="If given, use TODO")
    evaluate_ds_cmd.add_argument('--out_dir', required=True, help='Where to save outputs.')

    # Arguments for both 'compress' and 'decompress'.
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
        cmd.add_argument(
            "input_file",
            help="Input filename.")
        cmd.add_argument(
            "output_file", nargs="?",
            help=f"Output filename (optional). If not provided, appends '{ext}' to "
                 f"the input filename.")

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    if args.command == "train":
        train(args)
    elif args.command == "compress":
        if not args.output_file:
            args.output_file = args.input_file + ".tfci"
        compress(args)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file = args.input_file + ".png"
        decompress(args)
    elif args.command == "evaluate_ds":
        evaluate_ds(args, orp=True)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
