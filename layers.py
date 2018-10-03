import tensorflow as tf


def convolution_down(inputs, num_filters, filter_size=3, strides=[1, 1],
                     padding="valid"):
  conv1 = convolution2d(inputs, num_filters, filter_size, strides, padding)
  conv2 = convolution2d(conv1, num_filters, filter_size, strides, padding)
  max_pool = max_pooling2d(conv2)
  return max_pool, conv2


def convolution_up(inputs, inputs_contracting_path, num_filters, filter_size=3,
                   strides=[1, 1], padding="valid"):
  inputs_upsampled = up_scaling2d(inputs)
  concat_inputs = concat_by_depth(inputs_upsampled, inputs_contracting_path)
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
  pass


def up_scaling2d(inputs):
  current_size = inputs.get_shape().as_list()[1]
  return tf.image.resize_nearest_neighbor(inputs,
                                          size=[2 * current_size,
                                                2 * current_size])
