import numpy as np
import tensorflow as tf

import layers


class LayersTest(tf.test.TestCase):

  def testConcatInputsByFiltersDimension(self):
    inputs = tf.constant(np.ones(shape=[1, 10, 10, 4]))
    inputs_contracting_path = tf.constant(np.ones(shape=[1, 10, 10, 2]))
    concat_inputs_by_filters = layers.concat_by_filters(inputs,
                                                        inputs_contracting_path)

    expected_num_filters = 6
    actual_num_filters = concat_inputs_by_filters.get_shape().as_list()[-1]
    self.assertEqual(expected_num_filters, actual_num_filters)

  def testConvolutionDownNumFilters(self):
    inputs = tf.constant(np.ones(shape=[1, 10, 10, 1]))
    conv_down = layers.convolution_down(inputs, num_filters=4)

    expected_num_filters = 4
    actual_num_filters = conv_down.get_shape().as_list()[-1]
    self.assertEqual(expected_num_filters, actual_num_filters)

  def testConvolutionDownDefaultFilterSize(self):
    inputs = tf.constant(np.ones(shape=[1, 10, 10, 1]))
    conv_down = layers.convolution_down(inputs,
                                        num_filters=1,
                                        filter_size=3,
                                        strides=[1, 1],
                                        padding="valid")

    # [(height/width - filter_size + 2 * padding) // stride] + 1
    expected_image_dim = 8
    actual_image_dim = conv_down.get_shape().as_list()[1]
    self.assertEqual(expected_image_dim, actual_image_dim)

  def testConvolutionDown2Strides(self):
    inputs = tf.constant(np.ones(shape=[1, 10, 10, 1]))
    conv_down = layers.convolution_down(inputs,
                                        num_filters=1,
                                        filter_size=3,
                                        strides=[2, 2],
                                        padding="valid")

    # [(height/width - filter_size + 2 * padding) // stride] + 1
    expected_image_dim = 4
    actual_image_dim = conv_down.get_shape().as_list()[1]
    self.assertEqual(expected_image_dim, actual_image_dim)

  def testConvolutionUpDefaultFilterSize(self):
    inputs1 = tf.constant(np.ones(shape=[1, 10, 10, 1]))
    inputs2 = tf.constant(np.ones(shape=[1, 10, 10, 1]))
    conv_down = layers.convolution_up(inputs1,
                                      inputs2,
                                      num_filters=1,
                                      filter_size=3,
                                      strides=[1, 1],
                                      padding="valid")

    # [(height/width - filter_size + 2 * padding) // stride] + 1
    expected_image_dim = 8
    actual_image_dim = conv_down.get_shape().as_list()[1]
    self.assertEqual(expected_image_dim, actual_image_dim)

  def testConvolutionUp2Strides(self):
    inputs1 = tf.constant(np.ones(shape=[1, 10, 10, 1]))
    inputs2 = tf.constant(np.ones(shape=[1, 10, 10, 1]))

    conv_down = layers.convolution_up(inputs1,
                                      inputs2,
                                      num_filters=1,
                                      filter_size=3,
                                      strides=[2, 2],
                                      padding="valid")

    # [(height/width - filter_size + 2 * padding) // stride] + 1
    expected_image_dim = 4
    actual_image_dim = conv_down.get_shape().as_list()[1]
    self.assertEqual(expected_image_dim, actual_image_dim)

  def testMaxPoolingDefaultValues(self):
    inputs = tf.constant(np.ones(shape=[1, 10, 10, 2]))

    max_pool_layer = layers.max_pooling2d(inputs,
                                     pool_size=2,
                                     strides=[2, 2],
                                     padding="valid")

    expected_image_dim = 5
    actual_image_dim = max_pool_layer.get_shape().as_list()[1]
    self.assertEqual(expected_image_dim, actual_image_dim)

