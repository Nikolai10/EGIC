# Copyright 2024 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/Nikolai10/SwinT-ChARM/blob/master/zyc2022.py
# Copyright 2022 Nikolai Körber.
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

"""Core neural network blocks."""

import functools
import tensorflow as tf
import tensorflow_compression as tfc

from swins.blocks import BasicLayer
from swins.layers import PatchMerging, PatchSplitting, PatchUnpack


class AnalysisTransform(tf.keras.Sequential):
    """The analysis transform."""

    def __init__(self, cfg_ga):
        super().__init__()

        layers = [BasicLayer(
            dim=cfg_ga.embed_dim[i],
            out_dim=cfg_ga.embed_out_dim[i],
            depth=cfg_ga.depths[i],
            num_heads=cfg_ga.embed_dim[i] // cfg_ga.head_dim[i],
            head_dim=cfg_ga.head_dim[i],
            window_size=cfg_ga.window_size[i],
            mlp_ratio=4.0,
            qkv_bias=True,
            downsample=PatchMerging if (i < cfg_ga.num_layers - 1) else None,
            name=f"basic_layer_ga_{i}",
        ) for i in range(cfg_ga.num_layers)]

        self.add(tf.keras.layers.Lambda(lambda x: x / 255.))
        # self.add(PatchMerging((h*2, w*2), dim=3, out_dim=cfg_ga.embed_dim[0]))
        # PatchEmbed
        self.add(
            tf.keras.layers.Conv2D(filters=cfg_ga.embed_dim[0], kernel_size=(2, 2), strides=(2, 2), padding='same'))
        for layer in layers:
            self.add(layer)


class SynthesisTransform(tf.keras.Sequential):
    """The synthesis transform."""

    def __init__(self, cfg_gs):
        super().__init__()

        layers = [BasicLayer(
            dim=cfg_gs.embed_dim[i],
            out_dim=cfg_gs.embed_out_dim[i],
            depth=cfg_gs.depths[i],
            num_heads=cfg_gs.embed_dim[i] // cfg_gs.head_dim[i],
            head_dim=cfg_gs.head_dim[i],
            window_size=cfg_gs.window_size[i],
            mlp_ratio=4.0,
            qkv_bias=True,
            upsample=PatchSplitting if i < cfg_gs.num_layers - 1 else PatchUnpack,
            name=f"basic_layer_gs_{i}",
        ) for i in range(cfg_gs.num_layers)]

        for layer in layers:
            self.add(layer)
        self.add(tf.keras.layers.Lambda(lambda x: x * 255.))


class ORP(tf.keras.Sequential):
    """ORP head."""

    def __init__(self, cfg_gs):
        super().__init__()

        layers = [BasicLayer(
            dim=cfg_gs.embed_dim[-1],
            out_dim=cfg_gs.embed_out_dim[-1],
            depth=cfg_gs.depths[-1],
            num_heads=cfg_gs.embed_dim[-1] // cfg_gs.head_dim[-1],
            head_dim=cfg_gs.head_dim[-1],
            window_size=cfg_gs.window_size[-1],
            mlp_ratio=4.0,
            qkv_bias=True,
            upsample=PatchUnpack,
            name=f"final_basic_layer_gs",
        )]

        for layer in layers:
            self.add(layer)


class SynthesisTransformORP(tf.keras.Sequential):
    """The synthesis transform + ORP head."""

    def __init__(self, cfg_gs):
        super().__init__()

        layers = [BasicLayer(
            dim=cfg_gs.embed_dim[i],
            out_dim=cfg_gs.embed_out_dim[i],
            depth=cfg_gs.depths[i],
            num_heads=cfg_gs.embed_dim[i] // cfg_gs.head_dim[i],
            head_dim=cfg_gs.head_dim[i],
            window_size=cfg_gs.window_size[i],
            mlp_ratio=4.0,
            qkv_bias=True,
            upsample=PatchSplitting if i < cfg_gs.num_layers - 1 else PatchUnpack,
            name=f"basic_layer_gs_{i}",
        ) for i in range(cfg_gs.num_layers)]

        for layer in layers:
            self.add(layer)

        self.cond_model = ORP(cfg_gs)
        self.multiply = tf.keras.layers.Lambda(lambda x: x * 255.)

    def call(self, inputs):
        x, alpha = inputs
        features = []

        for layer in self.layers:
            features.append(tf.stop_gradient(x))
            x = layer(x)

        x = tf.stop_gradient(x)

        # features = features[::-1]
        pred_mse = self.cond_model(features[-1])

        residual = pred_mse - x

        x += (1 - alpha) * residual
        x = self.multiply(x)
        pred_mse = self.multiply(pred_mse)
        return x, pred_mse


class HyperAnalysisTransform(tf.keras.Sequential):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, cfg_ha):
        super().__init__()

        layers = [BasicLayer(
            dim=cfg_ha.embed_dim[i],
            out_dim=cfg_ha.embed_out_dim[i],
            depth=cfg_ha.depths[i],
            num_heads=cfg_ha.embed_dim[i] // cfg_ha.head_dim[i],
            head_dim=cfg_ha.head_dim[i],
            window_size=cfg_ha.window_size[i],
            mlp_ratio=4.0,
            qkv_bias=True,
            downsample=PatchMerging if (i < cfg_ha.num_layers - 1) else None,
            name=f"basic_layer_ha_{i}",
        ) for i in range(cfg_ha.num_layers)]

        # self.add(PatchMerging((h*2, w*2), dim=cfg_ga.embed_dim[-1], out_dim=cfg_ha.embed_dim[0]))
        # PatchEmbed
        self.add(
            tf.keras.layers.Conv2D(filters=cfg_ha.embed_dim[0], kernel_size=(2, 2), strides=(2, 2), padding='same'))
        for layer in layers:
            self.add(layer)


class HyperSynthesisTransform(tf.keras.Sequential):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, cfg_hs):
        super().__init__()

        layers = [BasicLayer(
            dim=cfg_hs.embed_dim[i],
            out_dim=cfg_hs.embed_out_dim[i],
            depth=cfg_hs.depths[i],
            num_heads=cfg_hs.embed_dim[i] // cfg_hs.head_dim[i],
            head_dim=cfg_hs.head_dim[i],
            window_size=cfg_hs.window_size[i],
            mlp_ratio=4.0,
            qkv_bias=True,
            upsample=PatchSplitting if i < cfg_hs.num_layers - 1 else PatchUnpack,
            name=f"basic_layer_hs_{i}",
        ) for i in range(cfg_hs.num_layers)]

        for layer in layers:
            self.add(layer)


# see Minnen et al., Appendix A./
# Zhu et al. Appendix A, Figure 12 for more information
class SliceTransform(tf.keras.layers.Layer):
    """Transform for channel-conditional params and latent residual prediction."""

    def __init__(self, cfg_charm, latent_depth, num_slices, index):
        super().__init__()
        conv = functools.partial(
            tfc.SignalConv2D, corr=False, strides_up=1, padding="same_zeros",
            use_bias=True, kernel_parameter="variable")

        # Note that the number of channels in the output tensor must match the
        # size of the corresponding slice. If we have 10 slices and a bottleneck
        # with 320 channels, the output is 320 / 10 = 32 channels.
        slice_depth = latent_depth // num_slices
        if slice_depth * num_slices != latent_depth:
            raise ValueError("Slices do not evenly divide latent depth (%d / %d)" % (
                latent_depth, num_slices))

        depth_conv0 = cfg_charm.depths_conv0[index]
        depth_conv1 = cfg_charm.depths_conv1[index]

        self.transform = tf.keras.Sequential([
            conv(depth_conv0, (3, 3), name="layer_0", activation=tf.nn.relu),
            conv(depth_conv1, (3, 3), name="layer_1", activation=tf.nn.relu),
            conv(slice_depth, (3, 3), name="layer_2", activation=None),
        ])

    def call(self, tensor):
        return self.transform(tensor)
