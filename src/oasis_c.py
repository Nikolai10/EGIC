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

"""OASIS-C implementation."""

import tensorflow as tf
import logging

from deeplab2 import common


def make_oasis_disc(input_dim_img=(256, 256, 3), input_dim_latent=(256 // 16, 256 // 16, 320), output_channel=135,
                    num_res_blocks=6):
    """
    OASIS-C discriminator based on:
    https://github.com/boschresearch/OASIS/blob/master/models/discriminator.py

    :param input_dim_img: image dimension
    :param input_dim_latent: latent dimension
    :param output_channel: number of output channels
    :param num_res_blocks: number of residual blocks
    :return:
    """
    channels = [3, 128, 128, 256, 256, 512, 512]
    c = 64
    body_up = []
    body_down = []

    # input: x,y
    h_x, w_x, c_x = input_dim_img
    h_y, w_y, c_y = input_dim_latent
    tar = tf.keras.layers.Input(shape=[h_x, w_x, c_x], name='x')
    y = tf.keras.layers.Input(shape=[h_y, w_y, c_y], name='y')

    # Upscale and fuse latent
    prepare_latent = tf.keras.Sequential()
    prepare_latent.add(WeightNormalization(tf.keras.layers.Conv2D(c, 3, strides=1, padding='same')))
    prepare_latent.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    latent = prepare_latent(y)
    latent = tf.image.resize(latent, [h_x, w_x], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # encoder part
    for i in range(num_res_blocks):
        body_down.append(OASISResBlock(channels[i], channels[i + 1], -1, first=(i == 0)))
    # decoder part
    body_up.append(OASISResBlock(channels[-1], channels[-2], 1))
    for i in range(1, num_res_blocks - 1):
        body_up.append(OASISResBlock(2 * channels[-1 - i], channels[-2 - i], 1))
    body_up.append(OASISResBlock(2 * channels[1], 64, 1))
    # nn.Conv2d(64, output_channel, 1, 1, 0)
    layer_up_last = tf.keras.Sequential()
    layer_up_last.add(tf.keras.layers.Conv2D(output_channel, kernel_size=1, strides=1, padding='same'))

    x = tar

    # encoder
    encoder_res = []
    for i in range(len(body_down)):
        x = body_down[i](x)
        encoder_res.append(x)
    # decoder
    x = body_up[0](x)
    for i in range(1, len(body_down)):
        x = body_up[i](tf.concat((encoder_res[-i - 1], x), axis=-1))

    # conditioning
    proj = tf.math.reduce_sum(latent * x, axis=-1, keepdims=True)  # (n, 256, 256, 1)
    proj = tf.repeat(proj, repeats=output_channel, axis=-1)

    # (n,h,w,output_channel)
    ans = layer_up_last(x)
    ans = ans + proj

    model = tf.keras.Model(inputs=[tar, y], outputs={common.PRED_SEMANTIC_LOGITS_KEY: ans})
    model.summary()
    return model


class OASISResBlock(tf.keras.layers.Layer):
    def __init__(self, fin, fout, up_or_down, first=False):
        super(OASISResBlock, self).__init__()
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        self.conv1 = tf.keras.Sequential()
        self.conv2 = tf.keras.Sequential()
        self.conv_s = None
        self.sampling = None
        self.fout = fout
        self.fmiddle = fout

    def build(self, input_shape):
        if self.first:
            # self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            self.conv1.add(
                WeightNormalization(tf.keras.layers.Conv2D(self.fmiddle, 3, strides=1, padding='same')))
        else:
            if self.up_or_down > 0:
                # self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), nn.Upsample(scale_factor=2), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
                self.conv1.add(tf.keras.layers.LeakyReLU(alpha=0.2))
                self.conv1.add(tf.keras.layers.UpSampling2D())
                self.conv1.add(
                    WeightNormalization(
                        tf.keras.layers.Conv2D(self.fmiddle, 3, strides=1, padding='same')))
            else:
                # self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
                self.conv1.add(tf.keras.layers.LeakyReLU(alpha=0.2))
                self.conv1.add(
                    WeightNormalization(
                        tf.keras.layers.Conv2D(self.fmiddle, 3, strides=1, padding='same')))

        # self.conv2 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))
        self.conv2.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.conv2.add(
            WeightNormalization(tf.keras.layers.Conv2D(self.fout, 3, strides=1, padding='same')))

        if self.learned_shortcut:
            # self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
            self.conv_s = tf.keras.Sequential()
            self.conv_s.add(
                WeightNormalization(tf.keras.layers.Conv2D(self.fout, 1, strides=1, padding='same')))
        if self.up_or_down > 0:
            # self.sampling = nn.Upsample(scale_factor=2)
            self.sampling = tf.keras.Sequential()
            self.sampling.add(tf.keras.layers.UpSampling2D())
        elif self.up_or_down < 0:
            # self.sampling = nn.AvgPool2d(2)
            self.sampling = tf.keras.Sequential()
            self.sampling.add(tf.keras.layers.AveragePooling2D())
        else:
            # self.sampling = nn.Sequential()
            pass

    def call(self, inputs):
        x = inputs
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    # ------------------------------------------------------------------------------
    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s

    # ------------------------------------------------------------------------------


# Credit: https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/layers/wrappers.py#L22
class WeightNormalization(tf.keras.layers.Wrapper):
    """Performs weight normalization.
    This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    This speeds up convergence by improving the
    conditioning of the optimization problem.
    See [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868).
    Wrap `tf.keras.layers.Conv2D`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = WeightNormalization(tf.keras.layers.Conv2D(2, 2), data_init=False)
    >>> y = conv2d(x)
    >>> y.shape
    TensorShape([1, 9, 9, 2])
    Wrap `tf.keras.layers.Dense`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = WeightNormalization(tf.keras.layers.Dense(10), data_init=False)
    >>> y = dense(x)
    >>> y.shape
    TensorShape([1, 10, 10, 10])
    Args:
      layer: A `tf.keras.layers.Layer` instance.
      data_init: If `True` use data dependent variable initialization.
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights.
      NotImplementedError: If `data_init` is True and running graph execution.
    """

    # @typechecked
    def __init__(self, layer: tf.keras.layers, data_init: bool = True, **kwargs):
        super().__init__(layer, **kwargs)
        self.data_init = data_init
        self._track_trackable(layer, name="layer")
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

        if self.data_init and self.is_rnn:
            logging.warning(
                "WeightNormalization: Using `data_init=True` with RNNs "
                "is advised against by the paper. Use `data_init=False`."
            )

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, "kernel"):
            raise ValueError(
                "`WeightNormalization` must wrap a layer that"
                " contains a `kernel` for weights"
            )

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(kernel.shape[-1])
        self.kernel_norm_axes = list(range(kernel.shape.rank - 1))

        self.g = self.add_weight(
            name="g",
            shape=(self.layer_depth,),
            initializer="ones",
            dtype=kernel.dtype,
            trainable=True,
        )
        self.v = kernel

        self._initialized = self.add_weight(
            name="initialized",
            shape=None,
            initializer="zeros",
            dtype=tf.dtypes.bool,
            trainable=False,
        )

        if self.data_init:
            # Used for data initialization in self._data_dep_init.
            with tf.name_scope("data_dep_init"):
                layer_config = tf.keras.layers.serialize(self.layer)
                layer_config["config"]["trainable"] = False
                self._naked_clone_layer = tf.keras.layers.deserialize(layer_config)
                self._naked_clone_layer.build(input_shape)
                self._naked_clone_layer.set_weights(self.layer.get_weights())
                if not self.is_rnn:
                    self._naked_clone_layer.activation = None

        self.built = True

    def call(self, inputs):
        """Call `Layer`"""

        def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            # Ensure we read `self.g` after _update_weights.
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = tf.cond(self._initialized, _do_nothing, _update_weights)

        with tf.name_scope("compute_weights"):
            # Replace kernel by normalized weight variable.
            kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * g

            if self.is_rnn:
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def _initialize_weights(self, inputs):
        """Initialize weight g.
        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        with tf.control_dependencies(
                [
                    tf.debugging.assert_equal(  # pylint: disable=bad-continuation
                        self._initialized, False, message="The layer has been initialized."
                    )
                ]
        ):
            if self.data_init:
                assign_tensors = self._data_dep_init(inputs)
            else:
                assign_tensors = self._init_norm()
            assign_tensors.append(self._initialized.assign(True))
            return assign_tensors

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope("init_norm"):
            v_flat = tf.reshape(self.v, [-1, self.layer_depth])
            v_norm = tf.linalg.norm(v_flat, axis=0)
            g_tensor = self.g.assign(tf.reshape(v_norm, (self.layer_depth,)))
            return [g_tensor]

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""
        with tf.name_scope("data_dep_init"):
            # Generate data dependent init values
            x_init = self._naked_clone_layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1.0 / tf.math.sqrt(v_init + 1e-10)

            # RNNs have fused kernels that are tiled
            # Repeat scale_init to match the shape of fused kernel
            # Note: This is only to support the operation,
            # the paper advises against RNN+data_dep_init
            if scale_init.shape[0] != self.g.shape[0]:
                rep = int(self.g.shape[0] / scale_init.shape[0])
                scale_init = tf.tile(scale_init, [rep])

            # Assign data dependent init values
            g_tensor = self.g.assign(self.g * scale_init)
            if hasattr(self.layer, "bias") and self.layer.bias is not None:
                bias_tensor = self.layer.bias.assign(-m_init * scale_init)
                return [g_tensor, bias_tensor]
            else:
                return [g_tensor]

    def get_config(self):
        config = {"data_init": self.data_init}
        base_config = super().get_config()
        return {**base_config, **config}

    def remove(self):
        kernel = tf.Variable(
            tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g,
            name="recurrent_kernel" if self.is_rnn else "kernel",
        )

        if self.is_rnn:
            self.layer.cell.recurrent_kernel = kernel
        else:
            self.layer.kernel = kernel

        return self.layer
