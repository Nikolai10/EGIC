# coding=utf-8
# Copyright 2022 The Deeplab2 Authors.
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

"""This file contains functions to post-process Panoptic-DeepLab results."""

import functools
from typing import Tuple, Dict, Text

import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import utils


def _get_semantic_predictions(semantic_logits: tf.Tensor) -> tf.Tensor:
  """Computes the semantic classes from the predictions.

  Args:
    semantic_logits: A tf.tensor of shape [batch, height, width, classes].

  Returns:
    A tf.Tensor containing the semantic class prediction of shape
      [batch, height, width].
  """
  return tf.argmax(semantic_logits, axis=-1, output_type=tf.int32)


def _get_instance_centers_from_heatmap(
    center_heatmap: tf.Tensor, center_threshold: float, nms_kernel_size: int,
    keep_k_centers: int) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes a list of instance centers.

  Args:
    center_heatmap: A tf.Tensor of shape [height, width, 1].
    center_threshold: A float setting the threshold for the center heatmap.
    nms_kernel_size: An integer specifying the nms kernel size.
    keep_k_centers: An integer specifying the number of centers to keep (K).
      Non-positive values will keep all centers.

  Returns:
    A tuple of
    - tf.Tensor of shape [N, 2] containing N center coordinates (after
      non-maximum suppression) in (y, x) order.
    - tf.Tensor of shape [height, width] containing the center heatmap after
      non-maximum suppression.
  """
  # Threshold center map.
  center_heatmap = tf.where(
      tf.greater(center_heatmap, center_threshold), center_heatmap, 0.0)

  # Non-maximum suppression.
  padded_map = utils.add_zero_padding(center_heatmap, nms_kernel_size, rank=3)
  pooled_center_heatmap = tf.keras.backend.pool2d(
      tf.expand_dims(padded_map, 0),
      pool_size=(nms_kernel_size, nms_kernel_size),
      strides=(1, 1),
      padding='valid',
      pool_mode='max')
  pooled_center_heatmap = tf.squeeze(pooled_center_heatmap, axis=0)

  center_heatmap = tf.where(
      tf.equal(pooled_center_heatmap, center_heatmap), center_heatmap, 0.0)
  center_heatmap = tf.squeeze(center_heatmap, axis=2)

  # `centers` is of shape (N, 2) with (y, x) order of the second dimension.
  centers = tf.where(tf.greater(center_heatmap, 0.0))

  if keep_k_centers > 0 and tf.shape(centers)[0] > keep_k_centers:
    topk_scores, _ = tf.math.top_k(
        tf.reshape(center_heatmap, [-1]), keep_k_centers, sorted=False)
    centers = tf.where(tf.greater(center_heatmap, topk_scores[-1]))

  return centers, center_heatmap


def _find_closest_center_per_pixel(centers: tf.Tensor,
                                   center_offsets: tf.Tensor) -> tf.Tensor:
  """Assigns all pixels to their closest center.

  Args:
    centers: A tf.Tensor of shape [N, 2] containing N centers with coordinate
      order (y, x).
    center_offsets: A tf.Tensor of shape [height, width, 2].

  Returns:
    A tf.Tensor of shape [height, width] containing the index of the closest
      center, per pixel.
  """
  height = tf.shape(center_offsets)[0]
  width = tf.shape(center_offsets)[1]

  x_coord, y_coord = tf.meshgrid(tf.range(width), tf.range(height))
  coord = tf.stack([y_coord, x_coord], axis=-1)

  center_per_pixel = tf.cast(coord, tf.float32) + center_offsets

  # centers: [N, 2] -> [N, 1, 2].
  # center_per_pixel: [H, W, 2] -> [1, H*W, 2].
  centers = tf.cast(tf.expand_dims(centers, 1), tf.float32)
  center_per_pixel = tf.reshape(center_per_pixel, [height*width, 2])
  center_per_pixel = tf.expand_dims(center_per_pixel, 0)

  # distances: [N, H*W].
  distances = tf.norm(centers - center_per_pixel, axis=-1)

  return tf.reshape(tf.argmin(distances, axis=0), [height, width])


def _get_instances_from_heatmap_and_offset(
    semantic_segmentation: tf.Tensor, center_heatmap: tf.Tensor,
    center_offsets: tf.Tensor, center_threshold: float,
    thing_class_ids: tf.Tensor, nms_kernel_size: int,
    keep_k_centers: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the instance assignment per pixel.

  Args:
    semantic_segmentation: A tf.Tensor containing the semantic labels of shape
      [height, width].
    center_heatmap: A tf.Tensor of shape [height, width, 1].
    center_offsets: A tf.Tensor of shape [height, width, 2].
    center_threshold: A float setting the threshold for the center heatmap.
    thing_class_ids: A tf.Tensor of shape [N] containing N thing indices.
    nms_kernel_size: An integer specifying the nms kernel size.
    keep_k_centers: An integer specifying the number of centers to keep.
      Negative values will keep all centers.

  Returns:
    A tuple of:
    - tf.Tensor containing the instance segmentation (filtered with the `thing`
      segmentation from the semantic segmentation output) with shape
      [height, width].
    - tf.Tensor containing the processed centermap with shape [height, width].
    - tf.Tensor containing instance scores (where higher "score" is a reasonable
      signal of a higher confidence detection.) Will be of shape [height, width]
      with the score for a pixel being the score of the instance it belongs to.
      The scores will be zero for pixels in background/"stuff" regions.
  """
  thing_segmentation = tf.zeros_like(semantic_segmentation)
  for thing_id in thing_class_ids:
    thing_segmentation = tf.where(tf.equal(semantic_segmentation, thing_id),
                                  1,
                                  thing_segmentation)

  centers, processed_center_heatmap = _get_instance_centers_from_heatmap(
      center_heatmap, center_threshold, nms_kernel_size, keep_k_centers)
  if tf.shape(centers)[0] == 0:
    return (tf.zeros_like(semantic_segmentation), processed_center_heatmap,
            tf.zeros_like(processed_center_heatmap))

  instance_center_index = _find_closest_center_per_pixel(
      centers, center_offsets)
  # Instance IDs should start with 1. So we use the index into the centers, but
  # shifted by 1.
  instance_segmentation = tf.cast(instance_center_index, tf.int32) + 1

  # The value of the heatmap at an instance's center is used as the score
  # for that instance.
  instance_scores = tf.gather_nd(processed_center_heatmap, centers)
  tf.debugging.assert_shapes([
      (centers, ('N', 2)),
      (instance_scores, ('N',)),
  ])
  # This will map the instance scores back to the image space: where each pixel
  # has a value equal to the score of its instance.
  flat_center_index = tf.reshape(instance_center_index, [-1])
  instance_score_map = tf.gather(instance_scores, flat_center_index)
  instance_score_map = tf.reshape(instance_score_map,
                                  tf.shape(instance_segmentation))
  instance_score_map *= tf.cast(thing_segmentation, tf.float32)

  return (thing_segmentation * instance_segmentation, processed_center_heatmap,
          instance_score_map)


@tf.function
def _get_panoptic_predictions(
    semantic_logits: tf.Tensor, center_heatmap: tf.Tensor,
    center_offsets: tf.Tensor, center_threshold: float,
    thing_class_ids: tf.Tensor, label_divisor: int, stuff_area_limit: int,
    void_label: int, nms_kernel_size: int, keep_k_centers: int,
    merge_semantic_and_instance_with_tf_op: bool
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the semantic class and instance ID per pixel.

  Args:
    semantic_logits: A tf.Tensor of shape [batch, height, width, classes].
    center_heatmap: A tf.Tensor of shape [batch, height, width, 1].
    center_offsets: A tf.Tensor of shape [batch, height, width, 2].
    center_threshold: A float setting the threshold for the center heatmap.
    thing_class_ids: A tf.Tensor of shape [N] containing N thing indices.
    label_divisor: An integer specifying the label divisor of the dataset.
    stuff_area_limit: An integer specifying the number of pixels that stuff
      regions need to have at least. The stuff region will be included in the
      panoptic prediction, only if its area is larger than the limit; otherwise,
      it will be re-assigned as void_label.
    void_label: An integer specifying the void label.
    nms_kernel_size: An integer specifying the nms kernel size.
    keep_k_centers: An integer specifying the number of centers to keep.
      Negative values will keep all centers.
    merge_semantic_and_instance_with_tf_op: Boolean, specifying the merging
      operation uses TensorFlow (CUDA kernel) implementation (True) or
      tf.py_function implementation (False). Note the tf.py_function
      implementation is simply used as a backup solution when you could not
      successfully compile the provided TensorFlow implementation. To reproduce
      our results, please use the provided TensorFlow implementation `merge_ops`
      (i.e., set to True).

  Returns:
    A tuple of:
    - the panoptic prediction as tf.Tensor with shape [batch, height, width].
    - the semantic prediction as tf.Tensor with shape [batch, height, width].
    - the instance prediction as tf.Tensor with shape [batch, height, width].
    - the centermap prediction as tf.Tensor with shape [batch, height, width].
    - the instance score maps as tf.Tensor with shape [batch, height, width].
  """
  semantic_prediction = _get_semantic_predictions(semantic_logits)
  batch_size = tf.shape(semantic_logits)[0]

  instance_map_lists = tf.TensorArray(
      tf.int32, size=batch_size, dynamic_size=False)
  center_map_lists = tf.TensorArray(
      tf.float32, size=batch_size, dynamic_size=False)
  instance_score_map_lists = tf.TensorArray(
      tf.float32, size=batch_size, dynamic_size=False)

  for i in tf.range(batch_size):
    (instance_map, center_map,
     instance_score_map) = _get_instances_from_heatmap_and_offset(
         semantic_prediction[i, ...], center_heatmap[i, ...],
         center_offsets[i, ...], center_threshold, thing_class_ids,
         nms_kernel_size, keep_k_centers)
    instance_map_lists = instance_map_lists.write(i, instance_map)
    center_map_lists = center_map_lists.write(i, center_map)
    instance_score_map_lists = instance_score_map_lists.write(
        i, instance_score_map)

  # This does not work with unknown shapes.
  instance_maps = instance_map_lists.stack()
  center_maps = center_map_lists.stack()
  instance_score_maps = instance_score_map_lists.stack()

  if merge_semantic_and_instance_with_tf_op:
    # pylint: disable=g-import-not-at-top
    from deeplab2.tensorflow_ops.python.ops import merge_semantic_and_instance_maps_op as merge_ops
    # pylint: enable=g-import-not-at-top
    panoptic_prediction = merge_ops.merge_semantic_and_instance_maps(
        semantic_prediction, instance_maps, thing_class_ids, label_divisor,
        stuff_area_limit, void_label)
  else:
    panoptic_prediction = _merge_semantic_and_instance_maps(
        semantic_prediction, instance_maps, thing_class_ids, label_divisor,
        stuff_area_limit, void_label)
  return (panoptic_prediction, semantic_prediction, instance_maps, center_maps,
          instance_score_maps)


@tf.function
def _merge_semantic_and_instance_maps(
    semantic_prediction: tf.Tensor,
    instance_maps: tf.Tensor,
    thing_class_ids: tf.Tensor,
    label_divisor: int,
    stuff_area_limit: int,
    void_label: int) -> tf.Tensor:
  """Merges semantic and instance maps to obtain panoptic segmentation.

  This function merges the semantic segmentation and class-agnostic
  instance segmentation to form the panoptic segmentation. In particular,
  the class label of each instance mask is inferred from the majority
  votes from the corresponding pixels in the semantic segmentation. This
  operation is first poposed in the DeeperLab paper and adopted by the
  Panoptic-DeepLab.

  - DeeperLab: Single-Shot Image Parser, T-J Yang, et al. arXiv:1902.05093.
  - Panoptic-DeepLab, B. Cheng, et al. In CVPR, 2020.

  Note that this function only supports batch = 1 for simplicity. Additionally,
  this function has a slightly different implementation from the provided
  TensorFlow implementation `merge_ops` but with a similar performance. This
  function is mainly used as a backup solution when you could not successfully
  compile the provided TensorFlow implementation. To reproduce our results,
  please use the provided TensorFlow implementation (i.e., not use this
  function, but the `merge_ops.merge_semantic_and_instance_maps`).

  Args:
    semantic_prediction: A tf.Tensor of shape [batch, height, width].
    instance_maps: A tf.Tensor of shape [batch, height, width].
    thing_class_ids: A tf.Tensor of shape [N] containing N thing indices.
    label_divisor: An integer specifying the label divisor of the dataset.
    stuff_area_limit: An integer specifying the number of pixels that stuff
      regions need to have at least. The stuff region will be included in the
      panoptic prediction, only if its area is larger than the limit; otherwise,
      it will be re-assigned as void_label.
    void_label: An integer specifying the void label.

  Returns:
    panoptic_prediction: A tf.Tensor with shape [batch, height, width].
  """
  prediction_shape = semantic_prediction.get_shape().as_list()
  # This implementation only supports batch size of 1. Since model construction
  # might lose batch size information (and leave it to None), override it here.
  prediction_shape[0] = 1
  semantic_prediction = tf.ensure_shape(semantic_prediction, prediction_shape)
  instance_maps = tf.ensure_shape(instance_maps, prediction_shape)

  # Default panoptic_prediction to have semantic label = void_label.
  panoptic_prediction = tf.ones_like(
      semantic_prediction) * void_label * label_divisor

  # Start to paste predicted `thing` regions to panoptic_prediction.
  # Infer `thing` segmentation regions from semantic prediction.
  semantic_thing_segmentation = tf.zeros_like(semantic_prediction,
                                              dtype=tf.bool)
  for thing_class in thing_class_ids:
    semantic_thing_segmentation = tf.math.logical_or(
        semantic_thing_segmentation,
        semantic_prediction == thing_class)
  # Keep track of how many instances for each semantic label.
  num_instance_per_semantic_label = tf.TensorArray(
      tf.int32, size=0, dynamic_size=True, clear_after_read=False)
  instance_ids, _ = tf.unique(tf.reshape(instance_maps, [-1]))
  for instance_id in instance_ids:
    # Instance ID 0 is reserved for crowd region.
    if instance_id == 0:
      continue
    thing_mask = tf.math.logical_and(instance_maps == instance_id,
                                     semantic_thing_segmentation)
    if tf.reduce_sum(tf.cast(thing_mask, tf.int32)) == 0:
      continue
    semantic_bin_counts = tf.math.bincount(
        tf.boolean_mask(semantic_prediction, thing_mask))
    semantic_majority = tf.cast(
        tf.math.argmax(semantic_bin_counts), tf.int32)

    while num_instance_per_semantic_label.size() <= semantic_majority:
      num_instance_per_semantic_label = num_instance_per_semantic_label.write(
          num_instance_per_semantic_label.size(), 0)

    new_instance_id = (
        num_instance_per_semantic_label.read(semantic_majority) + 1)
    num_instance_per_semantic_label = num_instance_per_semantic_label.write(
        semantic_majority, new_instance_id)
    panoptic_prediction = tf.where(
        thing_mask,
        tf.ones_like(panoptic_prediction) * semantic_majority * label_divisor
        + new_instance_id,
        panoptic_prediction)

  # Done with `num_instance_per_semantic_label` tensor array.
  num_instance_per_semantic_label.close()

  # Start to paste predicted `stuff` regions to panoptic prediction.
  instance_stuff_regions = instance_maps == 0
  semantic_ids, _ = tf.unique(tf.reshape(semantic_prediction, [-1]))
  for semantic_id in semantic_ids:
    if tf.reduce_sum(tf.cast(thing_class_ids == semantic_id, tf.int32)) > 0:
      continue
    # Check stuff area.
    stuff_mask = tf.math.logical_and(semantic_prediction == semantic_id,
                                     instance_stuff_regions)
    stuff_area = tf.reduce_sum(tf.cast(stuff_mask, tf.int32))
    if stuff_area >= stuff_area_limit:
      panoptic_prediction = tf.where(
          stuff_mask,
          tf.ones_like(panoptic_prediction) * semantic_id * label_divisor,
          panoptic_prediction)

  return panoptic_prediction


class SemanticOnlyPostProcessor(tf.keras.layers.Layer):
  """This class contains code of a semantic only post-processor."""

  def __init__(self):
    """Initializes a semantic only post-processor."""
    super(SemanticOnlyPostProcessor, self).__init__(
        name='SemanticOnlyPostProcessor')

  def call(self, result_dict: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    """Performs the post-processing given model predicted results.

    Args:
      result_dict: A dictionary of tf.Tensor containing model results. The dict
      has to contain
       - common.PRED_SEMANTIC_PROBS_KEY,

    Returns:
      The post-processed dict of tf.Tensor, containing the following:
       - common.PRED_SEMANTIC_KEY,
    """
    processed_dict = {}
    processed_dict[common.PRED_SEMANTIC_KEY] = _get_semantic_predictions(
        result_dict[common.PRED_SEMANTIC_PROBS_KEY])
    return processed_dict


class PostProcessor(tf.keras.layers.Layer):
  """This class contains code of a Panoptic-Deeplab post-processor."""

  def __init__(
      self,
      config: config_pb2.ExperimentOptions,
      dataset_descriptor: dataset.DatasetDescriptor):
    """Initializes a Panoptic-Deeplab post-processor.

    Args:
      config: A config_pb2.ExperimentOptions configuration.
      dataset_descriptor: A dataset.DatasetDescriptor.
    """
    super(PostProcessor, self).__init__(name='PostProcessor')
    self._post_processor = functools.partial(
        _get_panoptic_predictions,
        center_threshold=config.evaluator_options.center_score_threshold,
        thing_class_ids=tf.convert_to_tensor(
            dataset_descriptor.class_has_instances_list),
        label_divisor=dataset_descriptor.panoptic_label_divisor,
        stuff_area_limit=config.evaluator_options.stuff_area_limit,
        void_label=dataset_descriptor.ignore_label,
        nms_kernel_size=config.evaluator_options.nms_kernel,
        keep_k_centers=config.evaluator_options.keep_k_centers,
        merge_semantic_and_instance_with_tf_op=(
            config.evaluator_options.merge_semantic_and_instance_with_tf_op),
        )

  def call(self, result_dict: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
    """Performs the post-processing given model predicted results.

    Args:
      result_dict: A dictionary of tf.Tensor containing model results. The dict
      has to contain
        - common.PRED_SEMANTIC_PROBS_KEY,
        - common.PRED_CENTER_HEATMAP_KEY,
        - common.PRED_OFFSET_MAP_KEY,

    Returns:
      The post-processed dict of tf.Tensor, containing the following:
        - common.PRED_SEMANTIC_KEY,
        - common.PRED_INSTANCE_KEY,
        - common.PRED_PANOPTIC_KEY,
        - common.PRED_INSTANCE_CENTER_KEY,
        - common.PRED_INSTANCE_SCORES_KEY,
    """
    processed_dict = {}
    (processed_dict[common.PRED_PANOPTIC_KEY],
     processed_dict[common.PRED_SEMANTIC_KEY],
     processed_dict[common.PRED_INSTANCE_KEY],
     processed_dict[common.PRED_INSTANCE_CENTER_KEY],
     processed_dict[common.PRED_INSTANCE_SCORES_KEY]
    ) = self._post_processor(
        result_dict[common.PRED_SEMANTIC_PROBS_KEY],
        result_dict[common.PRED_CENTER_HEATMAP_KEY],
        result_dict[common.PRED_OFFSET_MAP_KEY])
    return processed_dict
