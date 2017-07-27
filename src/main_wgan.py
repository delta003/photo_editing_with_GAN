"""
    Trains WGAN on CelebA.
"""

from dataset import *
from wgan import *
from generators import *
from critics import *
import argparse
import tensorflow as tf
from loader import load_session

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

print('Training WGAN... {}'.format('Loading pre-trained' if args.load else ''))
print('Configuration: steps = {}, batch_size = {}, dataset_size = {}'.format(
    steps, batch_size, dataset_size))

img_size = 64
channels = 3
z_size = 100

generator = DCGANGenerator(img_size=img_size, channels=channels)
critic = DCGANCritic(img_size=img_size, channels=channels)

# Create session
sess = tf.Session()

wgan = WGAN(generator=generator,
            critic=critic,
            z_size=z_size,
            session=sess,
            model_path=project_path.model_path,
            img_size=img_size,
            channels=channels)

# Initialize variables
tf.global_variables_initializer().run(session = sess)

if args.load:
    # Do NOT restore Encoder namespace variables
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Critic") \
                + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Generator")
    loaded = load_session(wgan.session, 'log_transfer', variables)
    if not loaded:
        sys.exit(0)

dataset = CelebAData(img_size = img_size, dataset_size = dataset_size)

wgan.train(dataset = dataset, batch_size = batch_size, steps = steps)
