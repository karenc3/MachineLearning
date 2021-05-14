import os
import cv2
import glob
import random
import numpy as np
from itertools import chain


def read_image(img_path, is_resize=True, resize_dim=64):
	"""
	This function will read each individual image. The image will be converted to gray scale. Finally if specified,
	the image will be resized and returned
	:param img_path: the path of the image to be read
	:param is_resize: Weather to resize the image
	:param resize_dim: The height and width of the resized image
	:return: resized_img
	"""
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if is_resize:
		img = cv2.resize(img, (resize_dim, resize_dim))
	return img


def read_directory(path, is_test=False):
	"""
	This function will read the entire data from the specified directory. The function will return the image data and
	their corresponding classes.
	:param path: the root directory where the images are stored
	:param is_test: defines if we are running code in test or train mode
	:return: classes, resized_img_data
	"""
	str_labels = os.listdir(path)
	img_paths = [*chain(*[glob.glob(path + '/{}/*.png'.format(x)) for x in str_labels])]
	random.shuffle(img_paths)
	img_data = np.array([read_image(x) for x in img_paths], dtype=np.uint8)
	if is_test:
		return img_data
	labels = np.array([str_labels.index(x.split('/')[-1].split('\\')[0]) for x in img_paths], dtype=np.int)
	np.save('data/str_labels.npy', str_labels)
	return img_data, labels


if __name__ == '__main__':
	data_path = '../../data/fish_dataset/data'
	img_data, labels = read_directory(data_path)
	print(img_data.shape, labels.shape)
	print(labels[:10])