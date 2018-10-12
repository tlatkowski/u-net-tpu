import tensorflow as tf

from datasets import places
from layers import unet_layers
from layers import common_layers

LEARNING_RATE = 1e-4


def create_model(inputs):
  _, encoder_outputs = unet_layers.encoder(inputs)
  decoder_output = unet_layers.decoder(encoder_outputs)
  unet_output = unet_layers.output_layer(decoder_output)
  unet_clf_output = common_layers.feed_forward_relu_layer(unet_output)
  return unet_clf_output


def model_fn(features, labels, mode, params):
  image = features

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    model_output = create_model(image)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=model_output)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=tf.arg_max(model_output, axis=1))

    tf.identity(LEARNING_RATE, "learning_rate")
    tf.identity(loss, "cross_entropy")
    tf.identity(accuracy[1], "train_accuracy")

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

  if mode == tf.estimator.ModeKeys.PREDICT:
    pass


def run_u_net():
  u_net_model = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir="/path/to/model",
    config=None
  )

  def train_input_fn():
    train_data = places.train()
    train_data = train_data.cache().shuffle(buffer_size=50000).batch(
      batch_size=16)
    train_data = train_data.repeat(10)
    return train_data

  def eval_input_fn():
    pass

  u_net_model.train(input_fn=train_input_fn)


if __name__ == '__main__':
  run_u_net()