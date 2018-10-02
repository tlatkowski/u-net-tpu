import layers
import tensorflow as tf

NUM_FILTERS = [64, 128, 256, 512, 1024]
NUM_LAYERS = len(NUM_FILTERS)


def encoder(inputs):
  encoder_layers = dict()
  encoder_cache = dict()
  with tf.variable_scope("encoder"):
    for i in range(NUM_LAYERS):
      layer_name = "down-layer-{}".format(i)

      if i == 0:
        current_layer_output, conv2 = layers.convolution_down(inputs,
                                                              NUM_FILTERS[i])
        encoder_layers[layer_name] = current_layer_output
        encoder_cache[layer_name] = conv2
      else:
        previous_layer_output = encoder_layers["down-layer-{}".format(i - 1)]
        current_layer_output, conv2 = layers.convolution_down(
          previous_layer_output,
          NUM_FILTERS[i])
        encoder_layers[layer_name] = current_layer_output
        encoder_cache[layer_name] = conv2
  return encoder_layers, encoder_cache


def decoder(inputs, contracting_path):
  output = None
  return output
