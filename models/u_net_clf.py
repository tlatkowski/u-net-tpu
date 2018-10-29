import argparse
from datasets import problems

import tensorflow as tf

from datasets import places
from layers import common_layers
from layers import unet_layers

LEARNING_RATE = 1e-4
NUM_CLASSES = 10
BATCH_SIZE = 1
NUM_EPOCHS = 10


def create_model(inputs, num_classes):
  _, encoder_outputs = unet_layers.encoder(inputs)
  decoder_output = unet_layers.decoder(encoder_outputs)
  unet_output = unet_layers.output_layer(decoder_output)
  unet_clf_output = common_layers.feed_forward_relu_layer(unet_output)
  logits = common_layers.logits_layer(unet_clf_output, num_classes)
  return logits


def model_fn(features, labels, mode, params):
  image = features

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    logits = create_model(image, params["num_classes"])
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits)

    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)

    tf.identity(LEARNING_RATE, "learning_rate")
    tf.identity(loss, "cross_entropy")
    tf.identity(accuracy[1], "train_accuracy")

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = create_model(image)
    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.PREDICT,
      predictions=predictions,
      export_outputs={
        'classify': tf.estimator.export.PredictOutput(predictions)
      })


def run_u_net(train_dir, eval_dir, model_dir):
  u_net_model = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=model_dir,
    config=None,
    params={
      "num_classes": NUM_CLASSES
    }
  )

  def train_input_fn():
    train_data = places.train(train_dir)
    train_data = train_data.cache().shuffle(buffer_size=50000).batch(
      batch_size=BATCH_SIZE)
    train_data = train_data.repeat(NUM_EPOCHS)
    return train_data

  def eval_input_fn():
    pass

  u_net_model.train(input_fn=train_input_fn)


if __name__ == '__main__':
  args_parser = argparse.ArgumentParser()

  args_parser.add_argument("--train_dir",
                           required=True,
                           type=str,
                           help="Path to training examples")

  args_parser.add_argument("--eval_dir",
                           required=True,
                           type=str,
                           help="Path to evaluation examples")

  args_parser.add_argument("--problem",
                           required=True,
                           type=str,
                           choices=problems.Problem,
                           help="Problem to solve")

  args_parser.add_argument("--model_dir",
                           default="./u-net",
                           type=str,
                           help="Path to model")

  args = args_parser.parse_args()
  run_u_net(args.train_dir, args.eval_dir, args.model_dir)
