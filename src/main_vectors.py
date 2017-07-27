"""
    Calculates attributes' characteristic vectors.
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
from time import localtime, strftime

parser = argparse.ArgumentParser(description='Arguments.')
parser.add_argument('--dataset_size', type=int, default=-1)

args = parser.parse_args()
dataset_size = args.dataset_size

print('Calculating vectors... {}')
print('Configuration: dataset_size = {}'.format(dataset_size))

img_size = 64
channels = 3
z_size = 100
log_dir = 'log'

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

dataset = CelebAData(img_size = img_size, dataset_size = args.dataset_size)
dataset.load_attributes()

attributes = np.array(dataset.img_attributes)
n = attributes.shape[0]
attr_count = attributes.shape[1]
z_characteristic = np.zeros((attr_count, z_size))
z_characteristic_positive = np.zeros((attr_count, z_size))
z_characteristic_positive_count = np.zeros(attr_count)
z_characteristic_negative = np.zeros((attr_count, z_size))
z_characteristic_negative_count = np.zeros(attr_count)

batch_size = 200
idx = 0
while idx < n:
    images = dataset.get_images_batch(idx, batch_size)
    z = ae.extract_z(images)
    for i in range(attr_count):
        for j in range(idx, idx + batch_size):
            if j >= n:
                break
            if attributes[j][i] > 0:
                z_characteristic_positive[i] += z[j - idx]
                z_characteristic_positive_count[i] += 1
            else:
                z_characteristic_negative[i] += z[j - idx]
                z_characteristic_negative_count[i] += 1
    idx += batch_size
    print('{} / {}'.format(idx, n))

for i in range(attr_count):
    z_characteristic_positive[i] /= z_characteristic_positive_count[i]
    z_characteristic_negative[i] /= z_characteristic_negative_count[i]
    z_characteristic[i] = z_characteristic_positive - z_characteristic_negative

# Save to file
timestamp = strftime("%B_%d__%H_%M", localtime())
np.save("vectors_" + timestamp, z_characteristic)
