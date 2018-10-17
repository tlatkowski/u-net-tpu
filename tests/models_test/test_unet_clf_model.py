import numpy as np
import tensorflow as tf

from models import u_net_clf


class UNetModelTest(tf.test.TestCase):

  def testOutputShapes(self):
    inputs = tf.constant(np.ones(shape=[1, 572, 572, 1]))
    u_net_output = u_net_clf.create_model(inputs, num_classes=10)

    expected_num_logits = 10
    actual_num_logits = u_net_output.get_shape().as_list()[-1]

    self.assertEqual(expected_num_logits, actual_num_logits)
