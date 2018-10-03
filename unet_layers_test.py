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
    encoder_cache = dict()
    encoder_cache["down-layer-0"] = tf.constant(np.ones(shape=[1, 568, 568, 64]))
    encoder_cache["down-layer-1"] = tf.constant(np.ones(shape=[1, 280, 280, 128]))
    encoder_cache["down-layer-2"] = tf.constant(np.ones(shape=[1, 136, 136, 256]))
    encoder_cache["down-layer-3"] = tf.constant(np.ones(shape=[1, 64, 64, 512]))
    encoder_cache["down-layer-4"] = tf.constant(np.ones(shape=[1, 28, 28, 1024]))

    unet_layers.decoder(encoder_cache)

    raise NotImplementedError