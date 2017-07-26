from dataset import *
from wgan import *
from generators import *
from critics import *
from utils_celeb import *
from PIL import Image
import numpy as np

#dataset = FacesData(img_size=32)
# dataset = MNISTData()
dataset = CelebAData(64)

generator = FCGenerator(img_size=dataset.img_size,
                        channels=dataset.channels)
critic = FCCritic(img_size=dataset.img_size,
                  channels=dataset.channels)

wgan = WGAN(generator=generator,
            critic=critic,
            dataset=dataset,
            z_size=100)

wgan(batch_size=8, steps=100000, model_path=project_path.model_path)
