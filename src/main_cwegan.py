"""
    Trains AutoEncoder. Assumes pre-trained WGAN.
"""

from cwegan import CWEGAN
from dataset import *
from encoders import ConvAutoEncoder
from generators import *
from critics import *
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='Arguments.')

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

generator = ConvGenerator(img_size=img_size, channels=channels)
critic = ConvCritic(img_size=img_size, channels=channels)
encoder = ConvAutoEncoder(img_size=img_size, channels=channels)

# Create session
sess = tf.Session()

dataset = CelebAData(img_size = img_size, dataset_size = args.dataset_size)
dataset.load_attributes()

cwegan = CWEGAN(generator=generator,
              critic=critic,
              encoder=encoder,
              z_size=z_size,
              session=sess,
              model_path=project_path.model_path,
              img_size=img_size,
              channels=channels,
              attrs = len(dataset.attributes))

# Initialize variables
tf.global_variables_initializer().run(session = sess)

cwegan.train(dataset=dataset, batch_size=batch_size, steps=steps)
