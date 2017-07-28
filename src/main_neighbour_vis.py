from autoencoder import AutoEncoder
from dataset import *
from encoders import DCGANAutoEncoder
from wgan import *
from generators import *
from critics import *
import argparse
import tensorflow as tf
from loaders import load_session
import numpy as np

if __name__ == '__main__':

    img_size = 64
    channels = 3
    z_size = 100
    log_dir = 'log_transfer_27_21_24'

    generator = DCGANGenerator(img_size=img_size, channels=channels)
    critic = DCGANCritic(img_size=img_size, channels=channels)

    sess = tf.Session()

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9)

    wgan = WGAN(generator=generator,
                critic=critic,
                z_size=z_size,
                session=sess,
                model_path=project_path.model_path,
                img_size=64,
                channels=3,
                optimizer=optimizer)

    encoder = DCGANAutoEncoder(img_size=img_size, channels=channels)
    ae = AutoEncoder(encoder=encoder,
                     generator=generator,
                     z_size=z_size,
                     session=sess,
                     model_path=project_path.model_path,
                     img_size=64,
                     channels=3,
                     optimizer=optimizer)

    # Initialize variables
    tf.global_variables_initializer().run(session=sess)

    # Do NOT restore Encoder namespace variables
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Critic") \
                + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
    loaded = load_session(wgan.session, log_dir, variables)
    if not loaded:
        sys.exit(0)

    dataset = CelebAData(img_size=img_size, dataset_size=-1)
    # TODO:
    # Read vectors from cherry_pick.txt and generate images
    # Read cherry_pick_neighbour.txt for images files and load them
    # Merge two array of images and show them as grid binary
    # See wgan.generate_random_with_neighbor for how to visualize two column grid

    cherry_pick_file = 'cherry_pick.txt'
    with open(cherry_pick_file) as f:
        lines = f.readlines()
        z_cherry = np.array(len(lines), z_size)
        for idx, vector in enumerate(lines):
            z_cherry[idx] = vector.split()

    imgs = wgan.generate(z_cherry)
    neighbor_imgs = np.load('cherry_neighbours.npy')

    images = np.concatenate((imgs, neighbor_imgs))
    plt.imshow(visualize_grid_binary(np.array(images).astype(np.float32)))
    plt.axis("off")
    plt.show()




