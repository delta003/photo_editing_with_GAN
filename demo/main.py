"""
    Demo app.
"""
import json

from flask import Flask, render_template, request, redirect, url_for, send_file
from scipy.misc import imsave

from autoencoder import AutoEncoder
from dataset import *
from encoders import DCGANAutoEncoder
from wgan import *
from generators import *
from critics import *
import tensorflow as tf
from loaders import load_session, load_attributes_vectors

dataset_size = -1
img_size = 64
channels = 3
z_size = 100
log_dir = 'log_transfer_28_02_33'

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

dataset = CelebAData(img_size = img_size, dataset_size = dataset_size)
dataset.load_attributes()
vectors = load_attributes_vectors('vectors_July_28__11_40.npy', dataset)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html', vectors = vectors)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_file(os.path.join('../uploads', filename))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/random')
def random():
    z = np.random.rand(1, z_size)
    image = wgan.generate(z)
    imsave('uploads/random.png', image[0])
    return str(z[0])

def resize_image(filename):
    image = imread(os.path.join('uploads', filename))
    height = image.shape[0]
    width = image.shape[1]
    image = transform(image, height, width)
    imsave(os.path.join('uploads', filename), image)

@app.route('/editor/<filename>', methods = ['GET'])
def editor(filename):
    resize_image(filename)
    image = imread(os.path.join('uploads', filename))
    z = ae.extract_z([image])
    image_z = z[0]
    new_image = wgan.generate([image_z])
    imsave(os.path.join('uploads', 'edit-' + str(filename)), new_image[0])
    attributes = vectors.keys()
    return render_template('editor.html', filename = filename, z = str(image_z), attributes = attributes)

@app.route('/editor/<filename>', methods = ['POST'])
def edit(filename):
    confdata = request.form
    conf = {}
    for key, value in confdata.items():
        conf[key] = float(value)
    image = imread(os.path.join('uploads', filename))
    z = ae.extract_z([image])
    image_z = z[0]
    for key, vector in vectors.items():
        image_z += np.multiply(vector, conf[key])
    new_image = wgan.generate([image_z])
    imsave(os.path.join('uploads', 'edit-' + str(filename)), new_image[0])
    return str(image_z)

@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        app.logger.warning('File not found')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        app.logger.warning('File name empty')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return redirect(url_for('editor', filename=file.filename))
    app.logger.warning('File not allowed')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
