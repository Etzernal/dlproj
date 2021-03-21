import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime

class Lung_Train_Dataset(Dataset):
	
	def __init__(self, normalizer=None):
		"""
		Constructor for generic Dataset class - simply assembles
		the important parameters in attributes.
		"""
		self.normalizer = normalizer
		
		# All images are of size 150 x 150
		self.img_size = (150, 150)
		
		# Only two classes will be considered here (normal and infected)
		self.classes = {0: 'infected_covid', 1:'infected_non_covid'}
		
		# The dataset consists only of training images
		self.groups = 'train'
		
		# Number of images in each part of the dataset
		self.dataset_numbers = {'train_infected_covid': 1345,\
								'train_infected_non_covid':2530}
		
		# Path to images for different parts of the dataset
		self.dataset_paths = {'train_infected_covid': './dataset/train/infected/covid',\
							  'train_infected_non_covid':'./dataset/train/infected/non-covid'}
		
		
	def describe(self):
		"""
		Descriptor function.
		Will print details about the dataset when called.
		"""
		
		# Generate description
		msg = "This is the training dataset of the Lung Dataset"
		msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
		msg += " in Feb-March 2021. \n"
		msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
		msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
		msg += "The images are stored in the following locations "
		msg += "and each one contains the following number of images:\n"
		for key, val in self.dataset_paths.items():
			msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
		print(msg)
		
	
	def open_img(self, key, index_val):
		"""
		Opens image with specified parameters.
		
		Parameters:
		- group_val should take values in 'train', 'test' or 'val'.
		- class_val variable should be set to 'normal' or 'infected'.
		- index_val should be an integer with values between 0 and the maximal number of images in dataset.
		
		Returns loaded image as a normalized Numpy array.
		"""
		
		# Open file as before
		path_to_file = '{}/{}.jpg'.format(self.dataset_paths[key], index_val)
		with open(path_to_file, 'rb') as f:
			im = np.asarray(Image.open(f))/255
		
		return im
	
	
	def show_img(self, key, index_val):
		"""
		Opens, then displays image with specified parameters.
		
		Parameters:
		- group_val should take values in 'train', 'test' or 'val'.
		- class_val variable should be set to 'normal' or 'infected'.
		- index_val should be an integer with values between 0 and the maximal number of images in dataset.
		"""
		
		# Open image
		im = self.open_img(key, index_val)
		
		# Display
		plt.imshow(im)
		
		
	def __len__(self):
		"""
		Length special method, returns the number of images in dataset.
		"""
		
		# Length function
		return sum(self.dataset_numbers.values())
	
	
	def __getitem__(self, index):
		"""
		Getitem special method.
		
		Expects an integer value index, between 0 and len(self) - 1.
		
		Returns the image and its label as a one hot vector, both
		in torch tensor format in dataset.
		"""
		
		# Get item special method
		first_val = int(list(self.dataset_numbers.values())[0])
		if index < first_val:
			class_val = 'train_infected_covid'
			label = torch.Tensor([0])
		else:
			class_val = "train_infected_non_covid"
			index = index - first_val
			label = torch.Tensor([1])
		im = self.open_img(class_val, index)
		im = transforms.functional.to_tensor(np.array(im)).float()
		if self.normalizer:
			im = self.normalizer(im)
		return im, label
				
class Lung_Val_Dataset(Dataset):
	
	def __init__(self, normalizer=None):
		"""
		Constructor for generic Dataset class - simply assembles
		the important parameters in attributes.
		"""
		self.normalizer = normalizer
		
		# All images are of size 150 x 150
		self.img_size = (150, 150)
		
		# Only two classes will be considered here (normal and infected)
		self.classes = {0: 'infected covid', 1: 'infected non-covid'}
		
		# The dataset consists only of training images
		self.groups = 'val'
		
		# Number of images in each part of the dataset
		self.dataset_numbers = {'val_infected_covid': 8,\
								'val_infected_non_covid':8}
		
		# Path to images for different parts of the dataset
		self.dataset_paths = {'val_infected_covid': './dataset/val/infected/covid',\
							  'val_infected_non_covid':'./dataset/val/infected/non-covid'}
		
		
	def describe(self):
		"""
		Descriptor function.
		Will print details about the dataset when called.
		"""
		
		# Generate description
		msg = "This is the validation dataset of the Lung Dataset"
		msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
		msg += " in Feb-March 2021. \n"
		msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
		msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
		msg += "The images are stored in the following locations "
		msg += "and each one contains the following number of images:\n"
		for key, val in self.dataset_paths.items():
			msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
		print(msg)
		
	
	def open_img(self, key, index_val):
		"""
		Opens image with specified parameters.
		
		Parameters:
		- group_val should take values in 'train', 'test' or 'val'.
		- class_val variable should be set to 'normal' or 'infected'.
		- index_val should be an integer with values between 0 and the maximal number of images in dataset.
		
		Returns loaded image as a normalized Numpy array.
		"""
		
		
		# Open file as before
		path_to_file = '{}/{}.jpg'.format(self.dataset_paths[key], index_val)
		with open(path_to_file, 'rb') as f:
			im = np.asarray(Image.open(f))/255
		
		return im
	
	
	def show_img(self, key, index_val):
		"""
		Opens, then displays image with specified parameters.
		
		Parameters:
		- group_val should take values in 'train', 'test' or 'val'.
		- class_val variable should be set to 'normal' or 'infected'.
		- index_val should be an integer with values between 0 and the maximal number of images in dataset.
		"""
		
		# Open image
		im = self.open_img(key, index_val)
		
		# Display
		plt.imshow(im)
		
		
	def __len__(self):
		"""
		Length special method, returns the number of images in dataset.
		"""
		
		# Length function
		return sum(self.dataset_numbers.values())
	
	
	def __getitem__(self, index):
		"""
		Getitem special method.
		
		Expects an integer value index, between 0 and len(self) - 1.
		
		Returns the image and its label as a one hot vector, both
		in torch tensor format in dataset.
		"""
		
		# Get item special method
		first_val = int(list(self.dataset_numbers.values())[0])
		if index < first_val:
			class_val = 'val_infected_covid'
			label = torch.Tensor([0])
		else:
			class_val = "val_infected_non_covid"
			index = index - first_val
			label = torch.Tensor([1])
		im = self.open_img(class_val, index)
		im = transforms.functional.to_tensor(np.array(im)).float()
		if self.normalizer:
			im = self.normalizer(im)
		return im, label
		
class Lung_Test_Dataset(Dataset):
	
	def __init__(self, normalizer=None):
		"""
		Constructor for generic Dataset class - simply assembles
		the important parameters in attributes.
		"""
		self.normalizer = normalizer
		
		# All images are of size 150 x 150
		self.img_size = (150, 150)
		
		# Only two classes will be considered here (normal and infected)
		self.classes = {0: 'infected covid', 1: 'infected non-covid'}
		
		# The dataset consists only of training images
		self.groups = 'test'
		
		# Number of images in each part of the dataset
		self.dataset_numbers = {'test_infected_covid': 138,\
								'test_infected_non_covid':242}
		
		# Path to images for different parts of the dataset
		self.dataset_paths = {'test_infected_covid': './dataset/test/infected/covid',\
							  'test_infected_non_covid':'./dataset/test/infected/non-covid'}
		
		
	def describe(self):
		"""
		Descriptor function.
		Will print details about the dataset when called.
		"""
		
		# Generate description
		msg = "This is the testing dataset of the Lung Dataset"
		msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
		msg += " in Feb-March 2021. \n"
		msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
		msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
		msg += "The images are stored in the following locations "
		msg += "and each one contains the following number of images:\n"
		for key, val in self.dataset_paths.items():
			msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
		print(msg)
		
	
	def open_img(self, key, index_val):
		"""
		Opens image with specified parameters.
		
		Parameters:
		- group_val should take values in 'train', 'test' or 'val'.
		- class_val variable should be set to 'normal' or 'infected'.
		- index_val should be an integer with values between 0 and the maximal number of images in dataset.
		
		Returns loaded image as a normalized Numpy array.
		"""

		
		# Open file as before
		path_to_file = '{}/{}.jpg'.format(self.dataset_paths[key], index_val)
		with open(path_to_file, 'rb') as f:
			im = np.asarray(Image.open(f))/255
		
		return im
	
	
	def show_img(self, key, index_val):
		"""
		Opens, then displays image with specified parameters.
		
		Parameters:
		- group_val should take values in 'train', 'test' or 'val'.
		- class_val variable should be set to 'normal' or 'infected'.
		- index_val should be an integer with values between 0 and the maximal number of images in dataset.
		"""
		
		# Open image
		im = self.open_img(key, index_val)
		
		# Display
		plt.imshow(im)
		
		
	def __len__(self):
		"""
		Length special method, returns the number of images in dataset.
		"""
		
		# Length function
		return sum(self.dataset_numbers.values())
	
	
	def __getitem__(self, index):
		"""
		Getitem special method.
		
		Expects an integer value index, between 0 and len(self) - 1.
		
		Returns the image and its label as a one hot vector, both
		in torch tensor format in dataset.
		"""
		
		# Get item special method
		first_val = int(list(self.dataset_numbers.values())[0])
		if index < first_val:
			class_val = 'test_infected_covid'
			label = torch.Tensor([0])
		else:
			class_val = "test_infected_non_covid"
			index = index - first_val
			label = torch.Tensor([1])
		im = self.open_img(class_val, index)
		im = transforms.functional.to_tensor(np.array(im)).float()
		if self.normalizer:
			im = self.normalizer(im)
		return im, label