from autoencoder import AutoEncoder
from dataset import *
from encoders import DCGANAutoEncoder
from wgan import *
from generators import *
from critics import *
import argparse
import tensorflow as tf
from loader import load_session

parser = argparse.ArgumentParser(description='Arguments.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--steps', type=int, default=101)
parser.add_argument('--dataset_size', type=int, default=-1)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--train_encoder', type=bool, default=False)
parser.add_argument('--generate_random', type=bool, default=False)

args = parser.parse_args()
batch_size = args.batch_size
steps = args.steps

img_size = 64
channels = 3
z_size = 100

generator = DCGANGenerator(img_size=img_size, channels=channels)
critic = DCGANCritic(img_size=img_size, channels=channels)

sess = tf.Session()

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001, beta1 = 0.5, beta2 = 0.9)

wgan = WGAN(generator=generator,
            critic=critic,
            z_size=z_size,
            session=sess,
            model_path=project_path.model_path,
            img_size=64,
            channels=3,
            optimizer=optimizer)

# TODO: Unable to load since it wasn't trained with encoder
# encoder = DCGANAutoEncoder(img_size=img_size, channels=channels)
# ae = AutoEncoder(encoder=encoder,
#                  generator=generator,
#                  z_size=z_size,
#                  session=sess,
#                  model_path=project_path.model_path,
#                  img_size=64,
#                  channels=3,
#                  optimizer=optimizer)

if args.load:
    loaded = load_session(wgan.session, 'log_transfer')
    if not loaded:
        sys.exit(0)

dataset = None
if args.train or args.train_encoder:
    dataset = CelebAData(img_size = img_size, dataset_size = args.dataset_size)

if args.train:
    wgan.train(dataset=dataset, batch_size=batch_size, steps=steps)

# if args.train_encoder:
#     ae.train(dataset=dataset, batch_size=batch_size, steps=steps)

if args.generate_random:
    wgan.generate_random()

