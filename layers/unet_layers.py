from layers import common_layers
import tensorflow as tf

NUM_FILTERS = [64, 128, 256, 512, 1024]
NUM_LAYERS = len(NUM_FILTERS)

ENCODER_LAYER_NAME_PATTERN = "down-layer-{}"


def get_layer_name(i):
  return ENCODER_LAYER_NAME_PATTERN.format(i)


def encoder(inputs):
  encoder_layers = dict()
  encoder_cache = dict()
  with tf.variable_scope("encoder"):
    for i in range(NUM_LAYERS):
      layer_name = get_layer_name(i)

      if i == 0:
        current_encoder_output, conv2 = common_layers.convolution_down(inputs,
                                                                       NUM_FILTERS[
                                                                         i])
        encoder_layers[layer_name] = current_encoder_output
        encoder_cache[layer_name] = conv2
      else:
        previous_encoder_output = encoder_layers[get_layer_name(i - 1)]
        current_encoder_output, conv2 = common_layers.convolution_down(
          previous_encoder_output,
          NUM_FILTERS[i])
        encoder_layers[layer_name] = current_encoder_output
        encoder_cache[layer_name] = conv2
  return encoder_layers, encoder_cache


def decoder(inputs):
  num_layers = len(inputs)
  current_decoder_output = None
  with tf.variable_scope("decoder"):
    for i in reversed(range(1, num_layers)):
      num_filters = NUM_FILTERS[i - 1]
      if i == (num_layers - 1):
        last_layer_name = get_layer_name(i)
        current_layer_name = get_layer_name(i - 1)
        previous_layer = inputs[last_layer_name]
        current_layer = inputs[current_layer_name]
        current_decoder_output = common_layers.convolution_up(previous_layer,
                                                              current_layer,
                                                              num_filters)
      else:
        layer_name = get_layer_name(i - 1)
        encoder_layer = inputs[layer_name]
        previous_layer = current_decoder_output
        current_decoder_output = common_layers.convolution_up(previous_layer,
                                                              encoder_layer,
                                                              num_filters)

    return current_decoder_output
