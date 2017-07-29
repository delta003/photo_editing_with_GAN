import random

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
tf.global_variables_initializer().run(session = sess)

# Do NOT restore Encoder namespace variables
variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
loaded = load_session(wgan.session, log_dir, variables)
if not loaded:
    sys.exit(0)

dataset = CelebAData(img_size = img_size, dataset_size = -1)

images = []
for _ in range(10):
    id = random.randint(0, dataset.dataset_size)
    images.append(dataset.get_img_by_idx(id))
images = np.array(images)
z = ae.extract_z(images)
fake_imgs = wgan.generate(z)
print(images.shape)
print(fake_imgs.shape)

arr = np.concatenate((images, fake_imgs), axis = 0)
plt.imshow(visualize_grid_binary(np.array(arr).astype(np.float32)))
plt.axis("off")
plt.show()




