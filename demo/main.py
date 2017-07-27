# TODO for demo:
# 1. random images
# 2. vector to image
# 3. image to vector
# 4. load image and edit it with vectors

from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'