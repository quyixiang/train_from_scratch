import numpy as np
import os
import re
import random
from scipy.stats import bernoulli
from matplotlib.image import imread


def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a

def load_data(coefs, scale, num_samples, num_validation):
    images = [os.path.join("images", path) for path in os.listdir("images/")]
    images = list(filter(lambda ima: re.match("images/.*?.jpg", ima) is not None, images))
    random.shuffle(images)
    images = images[:num_samples + num_validation]
    num = num_samples + num_validation

    array = np.array(list(map(lambda x: re.findall(r"\/(.*?).jpg", x)[0].split("-"), images))).astype(np.float32)
    circumference, area = array[:, 0], array[:, 1]
    x = np.random.normal(0, 3, (num_samples + num_validation, 2)).astype(np.float32)
    x[:, 0] = x[:, 0] / scale + 3 * circumference
    betas = np.array(coefs)
    images_targets = 2.7*(circumference + area * 5)
    p = sigmoid(np.matmul(x, betas) + images_targets)
    labels = bernoulli.rvs(p = p).astype(np.int32)

    images_list = []
    vgg_list = []
    for img in images:
        img_array = imread(img)[:,:,:3]

        images_list.append(img_array)

    image_array = np.array(images_list)
    train_image = image_array[:num_samples]
    train_label = np.array(labels[:num_samples])
    valid_image = image_array[num_samples:]
    valid_label = np.array(labels[num_samples:])

    return train_image, train_label, valid_image, valid_label



load_data((-2.0,-1.6), 1., 1000, 1000)

    