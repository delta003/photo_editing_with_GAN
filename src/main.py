from dataset import *
from wgan import *
from generators import *
from critics import *
from utils_celeb import *
from PIL import Image
import numpy as np
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='Arguments.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--steps', type=int, default=101)
parser.add_argument('--dataset_size', type=int, default=-1)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--generate', type=bool, default=False)

args = parser.parse_args()
batch_size = args.batch_size
steps = args.steps

img_size = 64
channels = 3

generator = DCGANGenerator(img_size=img_size, channels=channels)
critic = DCGANCritic(img_size=img_size, channels=channels)

sess = tf.Session()

if args.load:
    model_path = 'log'
else:
    model_path = project_path.model_path

wgan = WGAN(generator=generator,
            critic=critic,
            z_size=100,
            session=sess,
            model_path=model_path,
            img_size = 64,
            channels = 3)

if args.load:
    loaded = wgan.load()
    if not loaded:
        sys.exit(0)
else:
    dataset = CelebAData(img_size = img_size, dataset_size = args.dataset_size)
    wgan.train(dataset=dataset, batch_size=batch_size, steps=steps)

if args.generate:
    wgan.generate()

