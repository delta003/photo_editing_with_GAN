import tensorflow as tf


class DCGANAutoEncoder:
    def __init__(self, img_size, channels):
        """
        Same as DCGANCritic except output isn't 1 but configurable.
        """
        pass

    def __call__(self, image, output_size):
        """
        This reuses critic variables in all except last layer.

        :param image:
        :param output_size:
        :return:
        """
        # We must keep same scope name to reuse weights from critic
        with tf.variable_scope("Critic", reuse = True):
            kwargs = {"kernel_size": (5, 5), "strides": (2, 2), "padding": "same", "activation": tf.nn.relu}

            # Same names as DCGANCritic variables
            image = tf.layers.conv2d(image, filters = 64, **kwargs)
            image = tf.layers.conv2d(image, filters = 128, **kwargs)
            image = tf.layers.conv2d(image, filters = 256, **kwargs)
            image = tf.layers.conv2d(image, filters = 1024, **kwargs)
            image = tf.reshape(image, [-1, 4 * 4 * 1024])

        # Last layer isn't reused from critic, it's encoder specific
        with tf.variable_scope("Encoder", reuse = None):
            image = tf.layers.dense(image, output_size)
            return image
