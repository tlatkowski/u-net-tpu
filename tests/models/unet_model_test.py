import numpy as np
import tensorflow as tf

from models import u_net


class UNetModelTest(tf.test.TestCase):

  def testOutputShapes(self):
    inputs = tf.constant(np.ones(shape=[1, 572, 572, 1]))
    u_net_output = u_net.create_model(inputs)

    expected_num_filters = 2
    expected_x_shape = expected_y_shape = 388

    actual_num_filters = u_net_output.get_shape().as_list()[-1]
    actual_x_shape = u_net_output.get_shape().as_list()[1]
    actual_y_shape = u_net_output.get_shape().as_list()[2]

    self.assertEqual(expected_num_filters, actual_num_filters)
    self.assertEqual(expected_x_shape, actual_x_shape)
    self.assertEqual(expected_y_shape, actual_y_shape)
