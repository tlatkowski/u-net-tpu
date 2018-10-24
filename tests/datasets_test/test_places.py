import os

import tensorflow as tf

from datasets import places


class TestPlacesDataset(tf.test.TestCase):

  def testDataset(self):
    path = os.path.join(os.path.abspath('.'), "data_test")
    image_paths, image_files = places.get_files(path)

    dataset = places.dataset(image_paths, image_files)

    dataset_iterator = dataset.make_initializable_iterator()
    instance = dataset_iterator.get_next()
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(dataset_iterator.initializer)
      image, label = session.run(instance)
      expected_shape = (512, 512, 3)
      actual_shape = image.shape
      self.assertSequenceEqual(expected_shape, actual_shape)
