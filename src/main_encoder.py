"""
    Trains AutoEncoder. Assumes pre-trained WGAN.
"""

from autoencoder import AutoEncoder
from dataset import *
from encoders import DCGANAutoEncoder
from wgan import *
from generators import *
from critics import *
import argparse
import tensorflow as tf
from loaders import load_session

parser = argparse.ArgumentParser(description='Arguments.')
# Load from checkpoint
parser.add_argument('--load', dest='load', action='store_true')

# Configurable parameters
parser.add_argument('--steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dataset_size', type=int, default=-1)

args = parser.parse_args()
steps = args.steps
batch_size = args.batch_size
dataset_size = args.dataset_size

print('Training AutoEncoder... {}'.format('Loading pre-trained' if args.load else ''))
print('Configuration: steps = {}, batch_size = {}, dataset_size = {}'.format(
    steps, batch_size, dataset_size))

img_size = 64
channels = 3
z_size = 100

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
variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "Critic") \
            + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "Generator")
# If load, than restore Encoder too
if args.load:
    variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "Encoder")

loaded = load_session(sess, 'log_transfer', variables)
if not loaded:
    sys.exit(0)

dataset = CelebAData(img_size = img_size, dataset_size = args.dataset_size)

ae.train(dataset=dataset, batch_size=batch_size, steps=steps)

