import os

import tensorflow as tf


def get_files(dir):
  pass


def dataset(image_files, files):
  def decode_image(image_file):
    pass

  def decode_label(label):
    pass

  images = tf.data.Dataset.from_tensor_slices(image_files).map(decode_image)
  labels = tf.data.Dataset.from_tensor_slices(files).map(decode_label)
  return tf.data.Dataset.zip((images, labels))


def train(train_dir):
  image_paths, image_files = get_files(train_dir)
  return dataset(image_paths, image_files)


def test(test_dir):
  return dataset(get_files(test_dir))
