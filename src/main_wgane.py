"""
    Trains AutoEncoder. Assumes pre-trained WGAN.
"""

from autoencoder import AutoEncoder
from dataset import *
from encoders import DCGANAutoEncoder, ConvAutoEncoder
from wgan import *
from generators import *
from critics import *
import argparse
import tensorflow as tf
from loaders import load_session
from wgane import WGANE

parser = argparse.ArgumentParser(description='Arguments.')

# Configurable parameters
parser.add_argument('--steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dataset_size', type=int, default=-1)

args = parser.parse_args()
steps = args.steps
batch_size = args.batch_size
dataset_size = args.dataset_size

print('Training WGANE...')
print('Configuration: steps = {}, batch_size = {}, dataset_size = {}'.format(
    steps, batch_size, dataset_size))

img_size = 64
channels = 3
z_size = 100

generator = ConvGenerator(img_size=img_size, channels=channels)
critic = ConvCritic(img_size=img_size, channels=channels)
encoder = ConvAutoEncoder(img_size=img_size, channels=channels)

# Create session
sess = tf.Session()

wgane = WGANE(generator=generator,
              critic=critic,
              encoder=encoder,
              z_size=z_size,
              session=sess,
              model_path=project_path.model_path,
              img_size=img_size,
              channels=channels)

# Initialize variables
tf.global_variables_initializer().run(session = sess)

dataset = CelebAData(img_size = img_size, dataset_size = args.dataset_size)

wgane.train(dataset=dataset, batch_size=batch_size, steps=steps)
