import tensorflow as tf


def convolution_down(inputs, num_filters, filter_size=3, strides=[1, 1],
                     padding="valid"):
  return tf.layers.conv2d(
    inputs=inputs,
    filters=num_filters,
    kernel_size=filter_size,
    strides=strides,
    padding=padding,
    activation=tf.nn.relu
  )


def convolution_up(inputs, inputs_contracting_path, num_filters, filter_size=3,
                   strides=[1, 1], padding="valid"):
  concat_inputs = concat_by_filters(inputs, inputs_contracting_path)
  return tf.layers.conv2d(
    inputs=concat_inputs,
    filters=num_filters,
    kernel_size=filter_size,
    strides=strides,
    padding=padding,
    activation=tf.nn.relu
  )


def concat_by_filters(inputs1, inputs2):
  return tf.concat([inputs1, inputs2], axis=-1)


def max_pooling2d(inputs, pool_size=2, strides=[2, 2], padding="valid"):
  return tf.layers.max_pooling2d(inputs,
                                 pool_size=pool_size,
                                 strides=strides,
                                 padding=padding)
