from dataset import *
import numpy as np

img_size = 64

dataset = CelebAData(img_size = img_size, dataset_size = -1)
dataset.load_attributes()

attributes = dataset.attributes
img_attributes = dataset.img_attributes

positive = np.zeros(len(attributes))
negative = np.zeros(len(attributes))
for cnt, img_att in enumerate(img_attributes):
    for i, att in enumerate(attributes):
        if img_att[i] > 0:
            positive[i] += 1
        else:
            negative[i] += 1
    if cnt % 100 == 0:
        print(cnt)
for i, att in enumerate(attributes):
    print('{}, {}, {}, {}, {}'.format(i, att, int(positive[i]), int(negative[i]), positive[i] / negative[i]))
