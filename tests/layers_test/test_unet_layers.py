import numpy as np
import tensorflow as tf

from layers import unet_layers


class TestUnetLayers(tf.test.TestCase):

  def testEncoderLayerNumSubLayers(self):
    inputs = tf.constant(np.ones(shape=[1, 572, 572, 3]))
    encoder_layers, encoder_cache = unet_layers.encoder(inputs)

    expected_num_encoder_layers = 5
    actual_num_encoder_layers = len(encoder_layers)
    self.assertEqual(expected_num_encoder_layers, actual_num_encoder_layers)

  def testDecoderLayerShapes(self):
    encoder_cache = dict()
    encoder_cache["down-layer-0"] = tf.constant(np.ones(shape=[1, 568, 568, 64]))
    encoder_cache["down-layer-1"] = tf.constant(np.ones(shape=[1, 280, 280, 128]))
    encoder_cache["down-layer-2"] = tf.constant(np.ones(shape=[1, 136, 136, 256]))
    encoder_cache["down-layer-3"] = tf.constant(np.ones(shape=[1, 64, 64, 512]))
    encoder_cache["down-layer-4"] = tf.constant(np.ones(shape=[1, 28, 28, 1024]))

    decoder_output = unet_layers.decoder(encoder_cache)

    expected_num_filters = 64
    expected_x_shape = expected_y_shape = 388

    actual_num_filters = decoder_output.get_shape().as_list()[-1]
    actual_x_shape = decoder_output.get_shape().as_list()[1]
    actual_y_shape = decoder_output.get_shape().as_list()[2]

    self.assertEqual(expected_num_filters, actual_num_filters)
    self.assertEqual(expected_x_shape, actual_x_shape)
    self.assertEqual(expected_y_shape, actual_y_shape)

  def testOutputLayer(self):
    raise NotImplementedError
