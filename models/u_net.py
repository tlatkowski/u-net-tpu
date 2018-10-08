import tensorflow as tf

from layers import unet_layers


def create_model(inputs):
  _, encoder_outputs = unet_layers.encoder(inputs)
  decoder_output = unet_layers.decoder(encoder_outputs)
  return unet_layers.output_layer(decoder_output)


def model_fn(features, labels, mode, params):
  image = features
  model = create_model(image)

  if mode == tf.estimator.ModeKeys.TRAIN:
    pass

  if mode == tf.estimator.ModeKeys.PREDICT:
    pass


def run_u_net():
  u_net_model = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir="/path/to/model",
    config=None
  )

  def train_input_fn():
    pass

  def eval_input_fn():
    pass
