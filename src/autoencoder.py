import sys
import tensorflow as tf
from utils import Timer


class AutoEncoder:
    max_summary_images = 4

    def __init__(self,
                 encoder,
                 generator,
                 z_size,
                 session,
                 model_path,
                 img_size,
                 channels,
                 optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001,
                                                    beta1 = 0.5, beta2 = 0.9)):

        self.session = session
        self.model_path = model_path
        self.dataset = None
        self.img_size = img_size
        self.channels = channels

        self.encoder = encoder
        self.generator = generator

        self.optimizer = optimizer
        self.z_size = z_size

        # image shape is [batch_size, height, width, channels]
        self.real_image = tf.placeholder(tf.float32,
                                         [None, self.img_size, self.img_size,
                                          self.channels],
                                         name = "Real_image")

        # ======================================================================
        #  Core logic is here
        self.z = self.encoder(self.real_image, self.z_size)
        self.fake_image = self.generator(self.z, reuse = True)  # Reuse existing generator

        self.e_cost = tf.reduce_mean(tf.square(self.fake_image - self.real_image))

        # IMPORTANT: This will mess up trained critic because of variables namespace,
        # but we don't need it after we're done training generator
        e_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Critic") \
                     + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Encoder")
        self.e_optimizer = self.optimizer.minimize(self.e_cost,
                                                   var_list = e_var_list,
                                                   name = "Encoder_optimizer")
        # ======================================================================

        # Defining summaries for tensorflow until the end of the method
        summaries = [
            tf.summary.scalar("Encoder cost", self.e_cost),
            tf.summary.image("Input image", self.real_image, max_outputs = AutoEncoder.max_summary_images),
            tf.summary.image("Recovered image", self.fake_image, max_outputs = AutoEncoder.max_summary_images)
        ]
        self.merged = tf.summary.merge(summaries)

    def train(self, dataset, batch_size, steps):
        self.dataset = dataset

        writer = tf.summary.FileWriter(self.model_path, self.session.graph)
        saver = tf.train.Saver()

        timer = Timer()
        for step in range(steps):
            print(step, end = " ")
            sys.stdout.flush()

            real_images = self.dataset.next_batch_real(batch_size)
            self.session.run(self.e_optimizer, feed_dict = {
                self.real_image: real_images
            })

            if step % 100 == 0:
                self.add_summary(self.session, step, writer, timer)
                saver.save(self.session, self.model_path)

        self.add_summary(self.session, steps, writer, timer)
        saver.save(self.session, self.model_path)

    def add_summary(self, sess, step, writer, timer):
        data_batch = self.dataset.next_batch_real(AutoEncoder.max_summary_images)

        summary = sess.run(self.merged, feed_dict = {self.real_image: data_batch})
        writer.add_summary(summary, step)
        print("\rSummary generated. Step", step,
              " Time == %.2fs" % timer.time())

    def extract_z(self, image):
        z = self.session.run(self.z, feed_dict = {self.real_image: image})
        return z
