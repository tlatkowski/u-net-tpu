import os

import tensorflow as tf


def get_files(dir):
  paths = []
  files = []
  path = os.path.expanduser(dir)
  for file in os.listdir(path):
    paths.append(os.path.join(path, file))
    files.append(file.split(".")[0])
  return paths, files


def dataset(image_files, files):
  def decode_image(image_file):
    image_contents = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image_contents)
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, size=[512, 512])
    return image

  def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.to_int32(label)

  images = tf.data.Dataset.from_tensor_slices(image_files).map(decode_image)
  labels = tf.data.Dataset.from_tensor_slices(files).map(decode_label)
  return tf.data.Dataset.zip((images, labels))


def train(train_dir):
  return dataset(get_files(train_dir))


def test(test_dir):
  return dataset(get_files(test_dir))
