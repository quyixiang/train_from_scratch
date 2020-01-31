import numpy as np
import os
import re
import random
from scipy.stats import bernoulli
from PIL import Image


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def load_data(coefs, scale, num_samples, num_validation):
    images = [os.path.join("images", path) for path in os.listdir("images/")]
    images = list(filter(lambda ima: re.match("images/.*?.jpg", ima) is not None, images))
    random.shuffle(images)
    images = images[:num_samples + num_validation]

    array = np.array(list(map(lambda x: re.findall(r"\/(.*?).jpg", x)[0].split("-"), images))).astype(np.float32)
    circumference, area = array[:, 0], array[:, 1]
    aug = np.random.normal(0, 3, (num_samples + num_validation, 2)).astype(np.float32)
    aug[:, 0] = aug[:, 0] / scale + 3 * circumference
    betas = np.array(coefs)
    images_targets = 2.7 * (circumference + area * 5)
    p = sigmoid(np.matmul(aug, betas) + images_targets)
    labels = bernoulli.rvs(p=p).astype(np.int32)

    label_array = np.array([[0, 1] if la == 0 else [1, 0] for la in labels])
    images_list = []
    for img in images:
        im = Image.open(img)
        (x, y) = im.size
        x_s, y_s = 32, 32
        out = np.array(im.resize((x_s, y_s), Image.ANTIALIAS))
        images_list.append(out)

    image_array = np.array(images_list)
    train_image = image_array[:num_samples]
    train_label = label_array[:num_samples]
    train_aug = np.array(aug[:num_samples])
    valid_image = image_array[num_samples:]
    valid_label = label_array[num_samples:]
    valid_aug = np.array(aug[num_samples:])

    return train_image, train_label, train_aug, valid_image, valid_label, valid_aug


load_data((-2.0, -1.6), 1., 1000, 1000)


def get_next_batch(max_length, length, train_images, train_labels, train_augs, test_images, test_labels, test_augs, is_training=True):
    if is_training:
        indicies = np.random.choice(max_length, length)
        next_batch = train_images[indicies]
        next_labels = train_labels[indicies]
        next_augs = train_augs[indicies]
    else:
        indicies = np.random.choice(max_length, length)
        next_batch = test_images[indicies]
        next_labels = test_labels[indicies]
        next_augs = test_augs[indicies]

    return np.array(next_batch), np.array(next_labels), np.array(next_augs)
