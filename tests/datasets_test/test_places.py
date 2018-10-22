import numpy as np
import tensorflow as tf
import os
from datasets import places


class TestPlacesDataset(tf.test.TestCase):

  def testDataset(self):
    image_files = os.listdir("./tests/data_test")
    images, labels = places.dataset(image_files)
    raise NotImplementedError
