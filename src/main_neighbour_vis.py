from autoencoder import AutoEncoder
from dataset import *
from encoders import DCGANAutoEncoder
from wgan import *
from generators import *
from critics import *
import tensorflow as tf
from loaders import load_session
import numpy as np

img_size = 64
channels = 3
z_size = 100
log_dir = 'log_transfer_28_02_37'

generator = DCGANGenerator(img_size=img_size, channels=channels)
critic = DCGANCritic(img_size=img_size, channels=channels)

# Create session
sess = tf.Session()

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001, beta1 = 0.5, beta2 = 0.9)

wgan = WGAN(generator=generator,
            critic=critic,
            z_size=z_size,
            session=sess,
            model_path=project_path.model_path,
            img_size=img_size,
            channels=channels,
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
tf.global_variables_initializer().run(session = sess)

# Do NOT restore Encoder namespace variables
variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

loaded = load_session(sess, log_dir, variables)
if not loaded:
    sys.exit(0)

dataset = CelebAData(img_size = img_size, dataset_size = -1)

cherry_pick_file = 'cherry_pick.txt'
with open(cherry_pick_file) as f:
    lines = f.readlines()
    z_cherry = np.zeros((len(lines), z_size))
    for idx, vector in enumerate(lines):
        z_cherry[idx] = np.array(vector.split())

imgs = wgan.generate(z_cherry)
neighbor_imgs = np.load('cherry_neighbours.npy')

images = np.concatenate((imgs, neighbor_imgs))
plt.imshow(visualize_grid_binary(np.array(images).astype(np.float32)))
plt.axis("off")
plt.show()
