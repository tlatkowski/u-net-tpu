import numpy as np
import tensorflow as tf
from scipy import misc
import os


def get_paths():
  paths = []
  path = os.path.expanduser("~/tcl-research/git/u-net-tpu/places")
  for file in os.listdir(
      os.path.expanduser("~/tcl-research/git/u-net-tpu/places")):
    paths.append(os.path.join(path, file))
  return paths


PLACES_TRAIN_DIR = get_paths()


def dataset(image_files):
  def decode_image(image):
    image_contents = tf.read_file(image)
    image = tf.image.decode_jpeg(image_contents)
    image = tf.cast(image, tf.float32)
    # image = tf.reshape(image, [256 * 256])
    return image

  images = tf.data.Dataset.from_tensor_slices(image_files).map(decode_image)

  return images


def train():
  return dataset(PLACES_TRAIN_DIR)


iter = train().make_initializable_iterator()
el = iter.get_next()

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(iter.initializer)
  a = session.run(el)
  misc.imsave('save.jpg', a[:, :, 0])
  print(np.shape(a))
