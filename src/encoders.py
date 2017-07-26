import tensorflow as tf


class DCGANAutoEncoder:
    def __init__(self, img_size, channels):
        """
        Same as DCGANCritic except output isn't 1 but configurable.
        """
        pass

    def __call__(self, image, output_size, reuse=None):
        """
        Works only for 64x64

        :param image:
        :param reuse:
        :return:
        """
        with tf.variable_scope("Encoder", reuse=reuse):
            kwargs = {"kernel_size": (5, 5), "strides": (2, 2), "padding": "same", "activation": tf.nn.relu}

            image = tf.layers.conv2d(image, filters=64, **kwargs)
            image = tf.layers.conv2d(image, filters=128, **kwargs)
            image = tf.layers.conv2d(image, filters=256, **kwargs)
            image = tf.layers.conv2d(image, filters=1024, **kwargs)
            image = tf.reshape(image, [-1, 4 * 4 * 1024])
            image = tf.layers.dense(image, output_size)
            return image
