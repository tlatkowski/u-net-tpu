import tensorflow as tf


def convolution_down(inputs, filter_size, num_filters, strides, padding):
  return tf.layers.conv2d(
    inputs=inputs,
    filters=num_filters,
    kernel_size=filter_size,
    strides=strides,
    padding=padding
  )
