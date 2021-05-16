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


def read_directory(img_paths, str_labels, is_test=False):
	"""
	This function will return the numpy images data and the corresponding labels for the image paths provided.
	:param img_paths: the image paths to be read to images
	:param str_labels:
	:param is_test: defines if we are running code in test or train mode
	:return: classes, resized_img_data
	"""
	img_data = np.array([read_image(x) for x in img_paths], dtype=np.uint8)
	if is_test:
		return img_data, None
	labels = np.array([str_labels.index(x.split('/')[-1].split('\\')[0]) for x in img_paths], dtype=np.int)
	return img_data, labels


def get_clf_data(path, train_per=0.8, is_split=True, get_test_labels=True):
	"""
	This function will read the entire data from the specified directory. The function will return the image data and
	their corresponding classes.
	:param path: The root directory path of the images
	:param train_per: if split, the percentage of images to be put into train
	:param is_split: defines if data is to be split in train and test
	:param get_test_labels: whether test labels are to be returned.
	:return:
	"""
	str_labels = os.listdir(path)
	img_paths = [*chain(*[glob.glob(path + '/{}/*.png'.format(x)) for x in str_labels])]
	random.shuffle(img_paths)
	if is_split:
		n_imgs = len(img_paths)
		n_train = int(train_per * n_imgs)
		train_path, test_path = img_paths[:n_train], img_paths[n_train:]
		train_data, train_labels = read_directory(train_path, str_labels)
		test_data, test_labels = read_directory(test_path, str_labels, is_test=(not get_test_labels))
		np.save('data/str_train_labels.npy', str_labels)
		return train_data, test_data, train_labels, test_labels
	else:
		return read_directory(img_paths, str_labels, is_test=(not get_test_labels))


if __name__ == '__main__':
	data_path = '../../data/fish_dataset/data'
	tr_data, te_data, tr_labels, te_labels = get_clf_data(data_path)
	print(tr_data.shape, te_labels[:10])
