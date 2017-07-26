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
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--load', type=bool, default=False)

args = parser.parse_args()
batch_size = args.batch_size
steps = args.steps

dataset = CelebAData(img_size = 64)

generator = FCGenerator(img_size=dataset.img_size,
                           channels=dataset.channels)
critic = FCCritic(img_size=dataset.img_size,
                     channels=dataset.channels)

sess = tf.Session()

if args.load:
    model_path = 'log'
else:
    model_path = project_path.model_path

wgan = WGAN(generator=generator,
            critic=critic,
            dataset=dataset,
            z_size=100,
            session=sess,
            model_path=model_path)

if args.load:
    wgan.load()
else:
    wgan.train(batch_size=batch_size, steps=steps)

