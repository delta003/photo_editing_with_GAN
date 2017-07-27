import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image) / 256

def show_image(image):
    plt.imshow(image.astype(np.float32))
    plt.axis("off")
    plt.show()

def pixel_distance(pixel1, pixel2):
    ret = 0
    for i in range(len(pixel1)):
        ret += (pixel1[i] - pixel2[i]) ** 2
    return ret

def image_distance(img1, img2):
    dist_square = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            dist_square += pixel_distance(img1[i][j], img2[i][j])
    return dist_square
