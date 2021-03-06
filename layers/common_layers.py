import enum
import functools

import tensorflow as tf


class UpScalingType(enum.Enum):
  TRANSPOSE_CONV = 0,
  RESIZE_NN = 1


def convolution_down(inputs, num_filters, filter_size=3, strides=[1, 1],
                     padding="valid"):
  conv1 = convolution2d(inputs, num_filters, filter_size, strides, padding)
  conv2 = convolution2d(conv1, num_filters, filter_size, strides, padding)
  max_pool = max_pooling2d(conv2)
  return max_pool, conv2


def convolution_up(inputs, inputs_contracting_path, num_filters, up_scaling_type, filter_size=3,
                   strides=[1, 1], padding="valid"):
  inputs_upsampled = up_scaling2d(inputs, up_scaling_type)
  inputs_cropped = crop(inputs_contracting_path, inputs_upsampled)
  concat_inputs = concat_by_depth(inputs_upsampled, inputs_cropped)
  conv1 = convolution2d(concat_inputs, num_filters, filter_size, strides,
                        padding)
  conv2 = convolution2d(conv1, num_filters, filter_size, strides,
                        padding)
  return conv2


def convolution2d(inputs, num_filters, filter_size=3, strides=[1, 1],
                  padding="valid"):
  return tf.layers.conv2d(
    inputs=inputs,
    filters=num_filters,
    kernel_size=filter_size,
    strides=strides,
    padding=padding,
    activation=tf.nn.relu
  )


def concat_by_depth(inputs1, inputs2):
  return tf.concat([inputs1, inputs2], axis=-1)


def max_pooling2d(inputs, pool_size=2, strides=[2, 2], padding="valid"):
  return tf.layers.max_pooling2d(inputs,
                                 pool_size=pool_size,
                                 strides=strides,
                                 padding=padding)


def up_pooling2d():
  raise NotImplementedError


def up_scaling2d(inputs, type=UpScalingType.RESIZE_NN):
  if type is UpScalingType.TRANSPOSE_CONV:
    return conv2d_transpose(inputs)
  elif type is UpScalingType.RESIZE_NN:
    return resize_nearest_neighbor(inputs)
  else:
    raise NotImplementedError


def conv2d_transpose(inputs):
  filters = inputs.get_shape().as_list()[-1]
  conv_2d = tf.layers.conv2d_transpose(inputs, filters=filters, kernel_size=3, strides=[2, 2],
                                       padding='same')
  return conv_2d


def resize_nearest_neighbor(inputs):
  current_size = inputs.get_shape().as_list()[1]
  return tf.image.resize_nearest_neighbor(inputs, size=[2 * current_size, 2 * current_size])


def crop(image_to_crop, target_image):
  image_to_crop_height = image_to_crop.get_shape().as_list()[1]
  target_image_height = target_image.get_shape().as_list()[1]
  offset_height = offset_width = compute_offset(image_to_crop_height,
                                                target_image_height)
  crop_height = crop_width = target_image_height
  return tf.image.crop_to_bounding_box(
    image=image_to_crop,
    offset_height=offset_height,
    offset_width=offset_width,
    target_height=crop_height,
    target_width=crop_width
  )


def feed_forward_relu_layer(inputs, units=1024):
  dims = inputs.get_shape().as_list()
  assert len(dims) == 4
  new_dim = functools.reduce(lambda x, y: x * y, dims[1:])
  inputs_flat = tf.reshape(inputs, [-1, new_dim])
  ff_layer = tf.layers.dense(inputs_flat, units=units, activation=tf.nn.relu)
  return ff_layer


def logits_layer(inputs, num_classes):
  dims = inputs.get_shape().as_list()
  assert len(dims) == 2
  logists = tf.layers.dense(inputs, units=num_classes)
  return logists


def compute_offset(dim1, dim2):
  return int((dim1 - dim2) / 2)
