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

This script provides training/ compression functionality.
"""

import sys
import os

from config import ConfigEGIC as cfg_egic

sys.path.append(cfg_egic.global_path)
sys.path.append(cfg_egic.global_path + '/swin-transformers-tf')

from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.trainer import runner_utils
from deeplab2 import common
from deeplab2.model.loss import loss_builder

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

from helpers import Nodes, pad, colorize_segments, read_png, write_png, learning_rate_decay, learning_rate_identity
from loss import compute_perceptual_loss, seg_loss, generate_labelmix, loss_labelmix

from archs import AnalysisTransform, SynthesisTransform, HyperAnalysisTransform, HyperSynthesisTransform, SliceTransform
from oasis_c import make_oasis_disc

from eval_utils import evaluate_ds

tf.config.experimental.enable_tensor_float_32_execution(False)


class KKSHMS2024Model(tf.keras.Model):
    """Main model class."""

    def __init__(self, warm_up, lmbda, disc, sum_wr, deeplab_loss,
                 latent_depth, hyperprior_depth,
                 num_slices, max_support_slices,
                 num_scales, scale_min, scale_max):

        super().__init__()
        self.warm_up = warm_up
        self.sum_wr = sum_wr
        self.lmbda = lmbda
        self.deeplab_loss = deeplab_loss
        self.num_scales = num_scales
        self.num_slices = num_slices
        self.slice_size = latent_depth // self.num_slices
        self.max_support_slices = max_support_slices
        offset = tf.math.log(scale_min)
        factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
                num_scales - 1.)
        self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
        self.analysis_transform = AnalysisTransform(cfg_ga)
        self.synthesis_transform = SynthesisTransform(cfg_gs)
        self.hyper_analysis_transform = HyperAnalysisTransform(cfg_ha)
        self.hyper_synthesis_mean_scale_transform = HyperSynthesisTransform(cfg_hs)
        self.cc_mean_transforms = [
            SliceTransform(cfg_charm, latent_depth, num_slices, idx) for idx in range(num_slices)]
        self.cc_scale_transforms = [
            SliceTransform(cfg_charm, latent_depth, num_slices, idx) for idx in range(num_slices)]
        self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=[hyperprior_depth])
        self.disc = disc

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
        x_hat = self.synthesis_transform(y_hat)

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

    def prepare_data(self, x):
        image, semantic_gt, semantic_loss_weight = x[common.IMAGE], x[common.GT_SEMANTIC_KEY], x[
            common.SEMANTIC_LOSS_WEIGHT_KEY]

        x1, x2 = tf.split(image, 2)
        semantic_gt, semantic_gt_d_steps = tf.split(semantic_gt, 2)
        semantic_loss_weight, semantic_loss_weight_d_steps = tf.split(semantic_loss_weight, 2)

        inputs = dict(image=x1, semantic_gt=semantic_gt, semantic_loss_weight=semantic_loss_weight)
        inputs_d_steps = dict(image=x2, semantic_gt=semantic_gt_d_steps,
                              semantic_loss_weight=semantic_loss_weight_d_steps)

        return inputs, inputs_d_steps

    def prepare_fake_labels(self, inputs_d_steps):

        inputs_fake = inputs_d_steps.copy()
        # label 134 corresponds to N+1 class (fake label, coco2017)
        segments_fake = tf.fill(tf.shape(inputs_d_steps[common.GT_SEMANTIC_KEY]), self.deeplab_loss._num_classes - 1)
        weights_fake = tf.ones(tf.shape(inputs_d_steps[common.SEMANTIC_LOSS_WEIGHT_KEY]))
        inputs_fake[common.GT_SEMANTIC_KEY] = segments_fake
        inputs_fake[common.SEMANTIC_LOSS_WEIGHT_KEY] = weights_fake

        return inputs_fake

    def compute_discriminator_out_seq(self, nodes: Nodes):
        """Get discriminator outputs."""
        input_image = nodes.input_image
        reconstruction = nodes.reconstruction

        # gradients_to_generator=False
        reconstruction = tf.stop_gradient(reconstruction)

        discriminator_in_real = input_image
        discriminator_in_fake = reconstruction

        # Condition D.
        latent = tf.stop_gradient(nodes.latent_quantized)

        discriminator_in_real = (discriminator_in_real, latent)
        discriminator_in_fake = (discriminator_in_fake, latent)

        disc_out_all_real = self.disc(discriminator_in_real, training=True)
        disc_out_all_fake = self.disc(discriminator_in_fake, training=True)

        return disc_out_all_real, disc_out_all_fake

    def compute_discriminator_out(self, nodes: Nodes, gradients_to_generator=True):
        """Get discriminator outputs."""

        input_image = nodes.input_image
        reconstruction = nodes.reconstruction

        if not gradients_to_generator:
            reconstruction = tf.stop_gradient(reconstruction)

        discriminator_in = reconstruction

        # Condition D.
        latent = tf.stop_gradient(nodes.latent_quantized)

        discriminator_in = (discriminator_in, latent)
        disc_out_all = self.disc(discriminator_in, training=True)

        return disc_out_all

    def train_step(self, x):

        inputs, inputs_d_steps = self.prepare_data(x)
        inputs_fake = self.prepare_fake_labels(inputs_d_steps)

        # stage 2
        if not self.warm_up:
            # first disc update
            with tf.GradientTape() as disc_tape:
                nodes_disc, _, _, _, _, _ = self(inputs_d_steps[common.IMAGE], training=True)
                d_out_real, d_out_fake = self.compute_discriminator_out_seq(nodes_disc)

                d_loss_real = seg_loss(inputs_d_steps, d_out_real, self.deeplab_loss)
                d_loss_fake = seg_loss(inputs_fake, d_out_fake, self.deeplab_loss)
                d_loss = d_loss_real + d_loss_fake

                # labelmix enabled by default
                segments_gt = inputs_d_steps[common.GT_SEMANTIC_KEY]
                segments_gt = tf.cast(segments_gt, tf.float32)
                mixed_inp, mask = tf.py_function(func=generate_labelmix,
                                                 inp=[segments_gt, nodes_disc.reconstruction, nodes_disc.input_image],
                                                 Tout=[tf.float32, tf.float32])

                mixed_inp.set_shape(nodes_disc.input_image.get_shape())
                nodes_mixed = Nodes(None, None, mixed_inp, None, latent_quantized=nodes_disc.latent_quantized)
                d_out_mixed = self.compute_discriminator_out(nodes_mixed, gradients_to_generator=False)
                d_loss_lm, mixed_D_output, output_D_mixed = loss_labelmix(mask, d_out_mixed, d_out_fake, d_out_real)
                d_loss_lm_scaled = tf.constant(10.0) * d_loss_lm
                d_loss += d_loss_lm_scaled

            variables = self.disc.trainable_variables
            gradients = disc_tape.gradient(d_loss, variables)
            self.d_optimizer.apply_gradients(zip(gradients, variables))

        # then generator update
        with tf.GradientTape() as gen_tape:
            nodes, loss, bpp, mse, weighted_mse, weighted_lpips = self(inputs[common.IMAGE], training=True)

            # stage 2
            if not self.warm_up:
                d_out = self.compute_discriminator_out(nodes, gradients_to_generator=True)
                # regular segmentation loss -> generator aims to generate realistic reconstructions
                g_loss = seg_loss(inputs, d_out, self.deeplab_loss)
                loss += 0.15 * g_loss

                # fix E, P -> only finetune G
        variables = self.synthesis_transform.trainable_variables
        gradients = gen_tape.gradient(loss, variables)
        self.g_optimizer.apply_gradients(zip(gradients, variables))

        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        self.weighted_mse.update_state(weighted_mse)
        self.weighted_lpips.update_state(weighted_lpips)

        # stage 1
        if self.warm_up:
            return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.weighted_mse, self.weighted_lpips]}

        # stage 2
        self.d_loss.update_state(d_loss)
        self.d_loss_real.update_state(d_loss_real)
        self.d_loss_fake.update_state(d_loss_fake)
        self.d_loss_lm.update_state(d_loss_lm)
        self.g_loss.update_state(g_loss)

        '''
        # very memory consuming -> just for debugging purposes.
        label_map = tf.py_function(func=colorize_segments, inp=[inputs_d_steps[common.GT_SEMANTIC_KEY]], Tout=[tf.uint8])
        
        d_out_real_temp = tf.argmax(d_out_real[common.PRED_SEMANTIC_LOGITS_KEY], axis=-1)
        d_out_fake_temp = tf.argmax(d_out_fake[common.PRED_SEMANTIC_LOGITS_KEY], axis=-1)
        seg_real = tf.py_function(func=colorize_segments, inp=[d_out_real_temp], Tout=[tf.uint8])
        seg_fake = tf.py_function(func=colorize_segments, inp=[d_out_fake_temp], Tout=[tf.uint8])
        mixed_d_out = tf.argmax(mixed_D_output, axis=-1)
        d_out_mixed = tf.argmax(output_D_mixed, axis=-1)
        
        seg_mixed_d_out = tf.py_function(func=colorize_segments, inp=[mixed_d_out], Tout=[tf.uint8])
        seg_d_out_mixed = tf.py_function(func=colorize_segments, inp=[d_out_mixed], Tout=[tf.uint8])
        
        sem_loss_weights = (inputs_d_steps['semantic_loss_weight'] - 1) * 128
        sem_loss_weights = tf.expand_dims(sem_loss_weights, axis=-1)
        sem_loss_weights = tf.saturate_cast(sem_loss_weights, tf.uint8)
        
        step = self.d_optimizer.iterations
        with self.sum_wr.as_default():
            tf.summary.image("image_real", tf.saturate_cast(nodes_disc.input_image, tf.uint8), max_outputs=1, step=step)
            tf.summary.image("image_fake", tf.saturate_cast(nodes_disc.reconstruction, tf.uint8), max_outputs=1, step=step)
            tf.summary.image("label_map", label_map, max_outputs=1, step=step)
            tf.summary.image("mask_M", tf.saturate_cast(mask * 255., tf.uint8), max_outputs=1, step=step)
            tf.summary.image("seg_real", seg_real, max_outputs=1, step=step)
            tf.summary.image("seg_fake", seg_fake, max_outputs=1, step=step)
            tf.summary.image("seg_mixed_d_out", seg_mixed_d_out, max_outputs=1, step=step)
            tf.summary.image("seg_d_out_mixed", seg_d_out_mixed, max_outputs=1, step=step)
            tf.summary.image("semantic_loss_weights", sem_loss_weights, max_outputs=1, step=step)
        '''

        return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.weighted_mse, self.weighted_lpips,
                                             self.d_loss, self.d_loss_real, self.d_loss_fake, self.d_loss_lm,
                                             self.g_loss]}

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

    def compile(self, d_optimizer, g_optimizer, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.bpp = tf.keras.metrics.Mean(name="bpp")
        self.mse = tf.keras.metrics.Mean(name="mse")
        self.weighted_mse = tf.keras.metrics.Mean(name="weighted_mse")
        self.weighted_lpips = tf.keras.metrics.Mean(name="weighted_lpips")
        self.d_loss = tf.keras.metrics.Mean(name="d_loss")
        self.d_loss_real = tf.keras.metrics.Mean(name="d_loss_real")
        self.d_loss_fake = tf.keras.metrics.Mean(name="d_loss_fake")
        self.d_loss_lm = tf.keras.metrics.Mean(name="d_loss_lm")
        self.g_loss = tf.keras.metrics.Mean(name="g_loss")

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

        x_hat = self.synthesis_transform(y_hat)

        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
        # Then cast back to 8-bit integer.
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)


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

    dataset_name = config.train_dataset_options.dataset

    num_classes = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].num_classes
    ignore_label = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].ignore_label
    ignore_depth = dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].ignore_depth
    class_has_instances_list = (dataset.MAP_NAME_TO_DATASET_INFO[dataset_name].class_has_instances_list)

    train_dataset = runner_utils.create_dataset(
        config.train_dataset_options,
        is_training=True,
        only_semantic_annotations=False)

    validation_dataset = runner_utils.create_dataset(
        config.eval_dataset_options,
        is_training=False,
        only_semantic_annotations=False)

    disc = None
    if not args.warm_up:
        disc = make_oasis_disc(input_dim_img=(256, 256, 3), input_dim_latent=(256 // 16, 256 // 16, 320))
        if args.init_disc == 1:
            print("loading disc checkpoint...")
            checkpoint = tf.train.Checkpoint(model=disc)
            checkpoint.restore(cfg_egic.disc_ckp_path)

    deeplab_loss = loss_builder.DeepLabFamilyLoss(
        loss_options=config.trainer_options.loss_options,
        deeplab_options=None,
        num_classes=num_classes + 1,
        ignore_label=ignore_label,
        ignore_depth=ignore_depth,
        thing_class_ids=class_has_instances_list)

    sum_wr = tf.summary.create_file_writer(args.train_path)

    model = KKSHMS2024Model(
        args.warm_up, args.lmbda, disc, sum_wr, deeplab_loss, args.latent_depth,
        args.hyperprior_depth, args.num_slices, args.max_support_slices,
        args.num_scales, args.scale_min, args.scale_max)
    model.compile(
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
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
            tf.keras.callbacks.LearningRateScheduler(
                learning_rate_decay) if args.warm_up else tf.keras.callbacks.LearningRateScheduler(
                learning_rate_identity)
        ],
        verbose=int(args.verbose),
    )
    model.save(args.model_path)


def compress(args):
    """Compresses an image."""
    # Load model and use it to compress the image.
    model = tf.keras.models.load_model(args.model_path)

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
        "--warm_up", type=int, default=0,
        help="{1: stage 1 (full learning objective w/o GAN), 0: stage 2 (full learning objective)}")
    train_cmd.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument(
        "--lr", type=float, default=1e-4, dest="lr",
        help="Learning rate.")
    train_cmd.add_argument(
        "--init_disc", type=int, default=1, dest="init_disc",
        help="load pre-trained disc weights")
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
        evaluate_ds(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
