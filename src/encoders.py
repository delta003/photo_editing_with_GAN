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


class ConditionalAutoEncoder:
    def __init__(self, img_size, channels):
        """
        Same as ConvCritic except output isn't 1 but configurable.
        """
        pass

    def __call__(self, image, condition, output_size):
        """
        This reuses critic variables in all except last layer.

        :param image:
        :param output_size:
        :return:
        """
        # We must keep same scope name to reuse weights from critic
        with tf.variable_scope("Critic", reuse = True):
            act = tf.nn.relu
            pad1 = [[0, 0], [1, 1], [1, 1], [0, 0]]

            kwargs3 = {"kernel_size" : (3, 3), "strides" : (1, 1),
                       "padding" : "valid"}
            kwargs4 = {"kernel_size" : (4, 4), "strides" : (4, 4),
                       "padding" : "valid"}

            image = tf.pad(image, pad1, mode = "SYMMETRIC")
            image = tf.layers.conv2d(image, filters = 64, **kwargs3,
                                     activation = act)

            # image is 64x64x1024
            image = tf.layers.conv2d(image, filters = 128, **kwargs4,
                                     activation = act)

            # image is 16x16x1024
            image = tf.pad(image, pad1, mode = "SYMMETRIC")
            image = tf.layers.conv2d(image, filters = 256, **kwargs3,
                                     activation = act)

            # image is 16x16x1024
            image = tf.layers.conv2d(image, filters = 512, **kwargs4,
                                     activation = act)

            # image is 4x4x1024
            image = tf.pad(image, pad1, mode = "SYMMETRIC")
            image = tf.layers.conv2d(image, filters = 1024, **kwargs3,
                                     activation = act)

            # image is 4x4x1024
            image = tf.reshape(image, [-1, 4 * 4 * 1024])
            image = tf.layers.dense(image, 256)

            # concatenate with critic
            image = tf.concat([image, condition], axis = 1)

        # Last layer isn't reused from critic, it's encoder specific
        with tf.variable_scope("Encoder", reuse = None):
            image = tf.layers.dense(image, 200)
            image = tf.layers.dense(image, output_size, activation = tf.nn.sigmoid)
            return image
