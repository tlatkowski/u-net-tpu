import numpy as np
import tensorflow as tf

import unet_layers


class UnetLayersTest(tf.test.TestCase):

  def testEncoder(self):
    inputs = tf.constant(np.ones(shape=[1, 572, 572, 3]))
    encoder_layers, encoder_cache = unet_layers.encoder(inputs)

    expected_num_encoder_layers = 5
    actual_num_encoder_layers = len(encoder_layers)
    self.assertEqual(expected_num_encoder_layers, actual_num_encoder_layers)

  def testDecoder(self):
    raise NotImplementedError