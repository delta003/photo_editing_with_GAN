from dataset import *

img_size = 64

dataset = CelebAData(img_size = img_size, dataset_size = -1)
dataset.load_attributes()

attributes = dataset.attributes
print(attributes)
img_attributes = dataset.img_attributes

for i, att1 in enumerate(attributes):
    cor = np.zeros(len(attributes))
    for j, att2 in enumerate(attributes):
        add = 0
        for img_att in img_attributes:
            if img_att[i] > 0 and img_att[j] > 0:
                add += 1
        cor[j] = add
    cor /= cor[i]
    print(list(cor))
