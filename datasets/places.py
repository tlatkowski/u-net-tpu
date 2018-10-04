import tensorflow as tf


def dataset(image_files):
  def decode_image(image):
    image_contents = tf.read_file(image)
    image = tf.image.decode_jpeg(image_contents)
    image = tf.cast(image, tf.float32)
    # image = tf.reshape(image, [256 * 256])
    return image

  images = tf.data.Dataset.from_tensor_slices(image_files).map(decode_image)

  return images
