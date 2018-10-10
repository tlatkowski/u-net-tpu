import numpy as np
import tensorflow as tf

from layers import common_layers


class TestCommonLayers(tf.test.TestCase):

  def testDepthConcat(self):
    inputs = tf.constant(np.ones(shape=[1, 10, 10, 4]))
    inputs_contracting_path = tf.constant(np.ones(shape=[1, 10, 10, 2]))
    concat_inputs_by_filters = common_layers.concat_by_depth(inputs,
                                                             inputs_contracting_path)

    expected_num_filters = 6
    actual_num_filters = concat_inputs_by_filters.get_shape().as_list()[-1]
    self.assertEqual(expected_num_filters, actual_num_filters)

  def testConvolution2dNumFilters(self):
    inputs = tf.constant(np.ones(shape=[1, 10, 10, 1]))
    conv_down = common_layers.convolution2d(inputs, num_filters=4)

    expected_num_filters = 4
    actual_num_filters = conv_down.get_shape().as_list()[-1]
    self.assertEqual(expected_num_filters, actual_num_filters)

  def testConvolution2dDefaultFilterSize(self):
    inputs = tf.constant(np.ones(shape=[1, 10, 10, 1]))
    conv_down = common_layers.convolution2d(inputs,
                                            num_filters=1,
                                            filter_size=3,
                                            strides=[1, 1],
                                            padding="valid")

    # [(height/width - filter_size + 2 * padding) // stride] + 1
    expected_image_dim = 8
    actual_image_dim = conv_down.get_shape().as_list()[1]
    self.assertEqual(expected_image_dim, actual_image_dim)

  def testConvolution2dStrides2(self):
    inputs = tf.constant(np.ones(shape=[1, 10, 10, 1]))
    conv_down = common_layers.convolution2d(inputs,
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

    max_pool_layer = common_layers.max_pooling2d(inputs,
                                                 pool_size=2,
                                                 strides=[2, 2],
                                                 padding="valid")

    expected_image_dim = 5
    actual_image_dim = max_pool_layer.get_shape().as_list()[1]
    self.assertEqual(expected_image_dim, actual_image_dim)

  def testUpScaling2d(self):
    inputs = tf.constant(np.ones(shape=[1, 10, 10, 2]))
    inputs_upsampled = common_layers.up_scaling2d(inputs)

    expected_image_dim = 20
    actual_image_dim = inputs_upsampled.get_shape().as_list()[1]
    self.assertEqual(expected_image_dim, actual_image_dim)

  def testConvolutionDown(self):
    raise NotImplementedError

  def testConvolutionUp(self):
    inputs1 = tf.constant(np.ones(shape=[1, 28, 28, 10]))
    inputs2 = tf.constant(np.ones(shape=[1, 56, 56, 10]))

    conv_up_output = common_layers.convolution_up(inputs1, inputs2,
                                                  num_filters=10)
    expected_image_dim = 52
    actual_image_dim = conv_up_output.get_shape().as_list()[1]
    self.assertEqual(expected_image_dim, actual_image_dim)

  def testCrop(self):
    image_to_crop = tf.constant(np.ones(shape=[1, 64, 64, 10]))
    target_image = tf.constant(np.ones(shape=[1, 56, 56, 10]))

    cropped_image = common_layers.crop(image_to_crop, target_image)
    expected_cropped_image_dim = 56
    actual_cropped_image_dim = cropped_image.get_shape().as_list()[1]
    self.assertEqual(expected_cropped_image_dim, actual_cropped_image_dim)

  def testCroppedMatrix(self):
    image_to_crop = tf.constant(np.arange(16).reshape([1, 4, 4, 1]))
    target_image = tf.constant(np.ones(shape=[1, 2, 2, 1]))

    cropped_image = common_layers.crop(image_to_crop, target_image)

    with self.test_session() as session:
      actual_cropped_image = session.run(cropped_image)
      expected_cropped_image = np.array([5., 6., 9., 10.]).reshape([1, 2, 2, 1])
      self.assertAllEqual(expected_cropped_image, actual_cropped_image)
