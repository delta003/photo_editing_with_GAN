"""
    Trains AutoEncoder. Assumes pre-trained WGAN.
"""

from cwgan import CWGAN
from dataset import *
from generators import *
from critics import *
import argparse
import tensorflow as tf
import sys

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

print('Training CWEGAN...')
print('Configuration: steps = {}, batch_size = {}, dataset_size = {}'.format(
    steps, batch_size, dataset_size))

img_size = 64
channels = 3
z_size = 100
log_dir = 'log'

generator = ConditionalGenerator(img_size=img_size, channels=channels)
critic = ConditionalCritic(img_size=img_size, channels=channels)

# Create session
sess = tf.Session()

dataset = CelebAData(img_size = img_size, dataset_size = args.dataset_size)
dataset.load_attributes()
conditions_size = len(dataset.attributes)

cwegan = CWGAN(generator=generator,
               critic=critic,
               z_size=z_size,
               session=sess,
               model_path=project_path.model_path,
               img_size=img_size,
               channels=channels,
               conditions_size=conditions_size)

# Initialize variables
tf.global_variables_initializer().run(session = sess)

if args.load:
    variables = tf.global_variables()
    loaded = load_session(sess, log_dir, variables)
    if not loaded:
        sys.exit(0)

cwegan.train(dataset=dataset, batch_size=batch_size, steps=steps)
