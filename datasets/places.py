import numpy as np
import tensorflow as tf
from scipy import misc
import os


def get_files(dir):
  paths = []
  path = os.path.expanduser(dir)
  for file in os.listdir(path):
    paths.append(os.path.join(path, file))
  return paths


def dataset(image_files):
  def decode_image(image_file):
    image_contents = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image_contents)
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, size=[512, 512])
    return image

  images = tf.data.Dataset.from_tensor_slices(image_files).map(decode_image)

  return images


def train(train_dir):
  return dataset(get_files(train_dir))


def test(test_dir):
  return dataset(get_files(test_dir))


# iter = train().make_initializable_iterator()
# el = iter.get_next()
#
# with tf.Session() as session:
#   session.run(tf.global_variables_initializer())
#   session.run(iter.initializer)
#   a = session.run(el)
#   misc.imsave('save.jpg', a[:, :, 0])
#   print(np.shape(a))
