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

parser = argparse.ArgumentParser(description='Arguments.')
parser.add_argument('--dataset_size', type=int, default=-1)

args = parser.parse_args()
dataset_size = args.dataset_size

print('Calculating vectors... {}')
print('Configuration: dataset_size = {}'.format(dataset_size))

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
            channels=channels)

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
variables = tf.get_collection(tf.GraphKeys.VARIABLES)

loaded = load_session(sess, 'log_transfer', variables)
if not loaded:
 sys.exit(0)

dataset = CelebAData(img_size = img_size, dataset_size = args.dataset_size)
dataset.load_attributes()
#print(np.array(dataset.img_attributes, dtype=int).shape)

# TODO: compute vectors and save to file
attributes = np.array(dataset.img_attributes, dtype=int)
z_characteristic = np.zeros(attributes.shape[1], z_size)

for attr_idx in range(attributes.shape[1]):
    positive_attr_imgs = []
    negative_attr_imgs = []
    for idx in range(attributes.shape[0]):
        if attributes[idx] > 0:
            positive_attr_imgs.append(dataset.get_img_by_idx(idx))
        else:
            negative_attr_imgs.append(dataset.get_img_by_idx(idx))

    #get z vectors for positive feature
    positive_z = ae.extract_z(positive_attr_imgs)
    avg_pos = sum(positive_z) / positive_z.shape[0]
    #same for negative
    negative_z = ae.extract_z(negative_attr_imgs)
    avg_neg = sum(negative_z) / negative_z.shape[0]
    z_characteristic[attr_idx] = avg_pos - avg_neg

#save to file
from time import localtime, strftime
timestamp = strftime("%B_%d__%H_%M", localtime())
np.save("vectors_" + timestamp, z_characteristic)



















