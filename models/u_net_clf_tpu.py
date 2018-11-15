import argparse

import tensorflow as tf

from datasets import problems
from layers import common_layers
from layers import unet_layers

LEARNING_RATE = 0.05
GLOBAL_BATCH_SIZE = 32
NUM_EPOCHS = 10
ITERATIONS = 50
NUM_SHARDS = 8
TRAIN_STEPS = 10000

tf.logging.set_verbosity(tf.logging.INFO)

logger = tf.logging


def create_model(inputs, params):
  num_classes = params['num_classes']
  if len(inputs.shape.as_list()) != 4:
    input_shape = params['input_shape']
    inputs = tf.reshape(inputs, shape=input_shape)

  _, encoder_outputs = unet_layers.encoder(inputs)
  decoder_output = unet_layers.decoder(encoder_outputs,
                                       up_scaling_type=common_layers.UpScalingType.TRANSPOSE_CONV)
  unet_output = unet_layers.output_layer(decoder_output)
  unet_clf_output = common_layers.feed_forward_relu_layer(unet_output)
  logits = common_layers.logits_layer(unet_clf_output, num_classes)
  return logits


def metric_fn(labels, logits):
  accuracy = tf.metrics.accuracy(
    labels=labels, predictions=tf.argmax(logits, axis=1))
  return {"accuracy": accuracy}


def model_fn(features, labels, mode, params):
  image = features

  logits = create_model(image, params)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = tf.train.exponential_decay(LEARNING_RATE,
                                               tf.train.get_global_step(),
                                               decay_steps=100000,
                                               decay_rate=0.96)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    distributed_optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)

    tf.identity(LEARNING_RATE, "learning_rate")
    tf.identity(loss, "cross_entropy")
    tf.identity(accuracy[1], "train_accuracy")

    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=distributed_optimizer.minimize(loss, tf.train.get_global_step()))

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits])
    )

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = create_model(image)
    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                      predictions=predictions,
                                      export_outputs={
                                        'classify': tf.estimator.export.PredictOutput(predictions)
                                      })


def run_u_net(problem, train_dir, eval_dir, tpu_name, tpu_zone, gcp_project, model_dir,
              use_tpu=True):
  def train_input_fn(params):
    batch_size = params["batch_size"]

    train_data = problem.train(train_dir)
    logger.info("Number of training images: %s", problem.num_training())

    ds = train_data.cache().repeat().shuffle(buffer_size=50000).apply(
      tf.contrib.data.batch_and_drop_remainder(batch_size))

    images, labels = ds.make_one_shot_iterator().get_next()

    return images, labels

  def eval_input_fn(params):
    batch_size = params["batch_size"]

    eval_data = problem.test(eval_dir)
    logger.info("Number of test images: %s", problem.num_test())

    eval_data = eval_data.apply(
      tf.contrib.data.batch_and_drop_remainder(batch_size))

    images, labels = eval_data.make_one_shot_iterator().get_next()

    return images, labels

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    tpu_name,
    zone=tpu_zone,
    project=gcp_project
  )

  run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=model_dir,
    session_config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True),
    tpu_config=tf.contrib.tpu.TPUConfig(ITERATIONS, NUM_SHARDS),
  )

  estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn,
    use_tpu=use_tpu,
    train_batch_size=GLOBAL_BATCH_SIZE,
    eval_batch_size=GLOBAL_BATCH_SIZE,
    predict_batch_size=GLOBAL_BATCH_SIZE,
    params={
      "data_dir": train_dir,
      "num_classes": problem.num_classes(),
      "input_shape": problem.input_shape()
    },
    config=run_config)

  estimator.train(input_fn=train_input_fn, max_steps=TRAIN_STEPS)
  estimator.evaluate(input_fn=eval_input_fn, steps=10)


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
                           choices=problems.all_problems(),
                           help="Problem to solve")

  args_parser.add_argument("--tpu_name",
                           required=True,
                           type=str,
                           help="TPU name")

  args_parser.add_argument("--tpu_zone",
                           required=True,
                           type=str,
                           help="TPU zone")

  args_parser.add_argument("--gcp_project",
                           required=True,
                           type=str,
                           help="Google Cloud Platform project")

  args_parser.add_argument("--model_dir",
                           required=True,
                           type=str,
                           help="Path to model")

  args = args_parser.parse_args()
  problem = problems.get_problem(args.problem)
  run_u_net(problem, args.train_dir, args.eval_dir, args.tpu_name, args.tpu_zone, args.gcp_project,
            args.model_dir)
