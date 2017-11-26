import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from multibox_layer import MultiBoxLayer

from NetworkConfig import *

class L2Norm2d(nn.Module):
	'''L2Norm layer across all channels.'''
	def __init__(self, scale):
		super(L2Norm2d, self).__init__()
		self.scale = scale

	def forward(self, x, dim=1):
		'''out = scale * x / sqrt(\sum x_i^2)'''
		if args.using_python_2:
			return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().unsqueeze(1).expand_as(x)
		else:
			return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)

class SSD300(nn.Module):

	def __init__(self):
		super(SSD300, self).__init__()
		self.batch_norm = True
		# modelVGG16
		self.base = self.make_layers()
		#self.base = self.VGG16()
		self.norm4 = L2Norm2d(20)

		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
		self.norm5_1 = nn.BatchNorm2d(512)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
		self.norm5_2 = nn.BatchNorm2d(512)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
		self.norm5_3 = nn.BatchNorm2d(512)


		self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

		self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

		self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
		self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)

		self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
		self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

		self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
		self.conv10_1_dp = nn.Dropout2d(p = 0.2)

		if Network_type == 2:
			self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
			self.conv10_2_dp = nn.Dropout2d(p = 0.2)
		else:
			self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)
			self.conv10_2_dp = nn.Dropout2d(p = 0.2)

		self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
		self.conv11_1_dp = nn.Dropout2d(p = 0.2)
		self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)
		self.conv11_2_dp = nn.Dropout2d(p = 0.2)

		if Network_type == 0:
			pass

		elif Network_type == 1:
			self.conv12_1 = nn.Conv2d(256, 128, kernel_size=1)
			self.conv12_1_dp = nn.Dropout2d(p = 0.2)
			self.conv12_2 = nn.Conv2d(128, 256, kernel_size=3)
			self.conv12_2_dp = nn.Dropout2d(p = 0.2)

		elif Network_type == 2:
			self.conv12_1 = nn.Conv2d(256, 128, kernel_size=1)
			self.conv12_1_dp = nn.Dropout2d(p = 0.2)
			self.conv12_2 = nn.Conv2d(128, 256, kernel_size=2)
			self.conv12_2_dp = nn.Dropout2d(p = 0.2)

		elif Network_type == 3:
			self.conv12_1 = nn.Conv2d(256, 128, kernel_size=1)
			self.conv12_1_dp = nn.Dropout2d(p = 0.2)
			self.conv12_2 = nn.Conv2d(128, 256, kernel_size=3)
			self.conv12_2_dp = nn.Dropout2d(p = 0.2)

			self.conv13_1 = nn.Conv2d(256, 128, kernel_size=1)
			self.conv13_1_dp = nn.Dropout2d(p = 0.2)
			self.conv13_2 = nn.Conv2d(128, 256, kernel_size=3)
			self.conv13_2_dp = nn.Dropout2d(p = 0.2)

		# multibox layer
		self.multibox = MultiBoxLayer()

	def forward(self, x):
		hs = []
		h = self.base(x)

		#print(h.data.numpy().shape)
		
		hs.append(self.norm4(h))  # conv4_3

		h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

		h = F.relu(self.norm5_1(self.conv5_1(h)))
		h = F.relu(self.norm5_2(self.conv5_2(h)))
		h = F.relu(self.norm5_3(self.conv5_3(h)))
		h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

		h = F.relu(self.conv6(h))
		h = F.relu(self.conv7(h))
		
		#print(h.data.numpy().shape)
		hs.append(h)  # conv7

		h = F.relu(self.conv8_1(h))
		h = F.relu(self.conv8_2(h))

		#print(h.data.numpy().shape)
		hs.append(h)  # conv8_2

		h = F.relu(self.conv9_1(h))
		h = F.relu(self.conv9_2(h))

		#print(h.data.numpy().shape)
		hs.append(h)  # conv9_2

		h = F.relu(self.conv10_1(h))
		h = self.conv10_1_dp(h)
		h = F.relu(self.conv10_2(h))
		h = self.conv10_2_dp(h)

		#print(h.data.numpy().shape)
		hs.append(h)  # conv10_2

		h = F.relu(self.conv11_1(h))
		h = self.conv11_1_dp(h)
		h = F.relu(self.conv11_2(h))
		h = self.conv11_2_dp(h)

		#print(h.data.numpy().shape)
		hs.append(h)  # conv11_2


		if Network_type == 1:

			h = F.relu(self.conv12_1(h))
			h = self.conv12_1_dp(h)
			h = F.relu(self.conv12_2(h))
			h = self.conv12_2_dp(h)

			#print(h.data.numpy().shape)
			hs.append(h)  # conv11_2

		if Network_type == 2:

			h = F.relu(self.conv12_1(h))
			h = self.conv12_1_dp(h)
			h = F.relu(self.conv12_2(h))
			h = self.conv12_2_dp(h)

			#print(h.data.numpy().shape)
			hs.append(h)  # conv11_2

		if Network_type == 3:

			h = F.relu(self.conv12_1(h))
			h = self.conv12_1_dp(h)
			h = F.relu(self.conv12_2(h))
			h = self.conv12_2_dp(h)

			#print(h.data.numpy().shape)
			hs.append(h)  # conv11_2

			h = F.relu(self.conv13_1(h))
			h = self.conv13_1_dp(h)
			h = F.relu(self.conv13_2(h))
			h = self.conv13_2_dp(h)

			#print(h.data.numpy().shape)
			hs.append(h)  # conv11_2

			#quit()

		loc_preds, conf_preds = self.multibox(hs)

		return loc_preds, conf_preds

	def VGG16(self):
		'''VGG16 layers.'''
		cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
		layers = []
		in_channels = 1
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.ReLU(True)]
				in_channels = x
		return nn.Sequential(*layers)

	def make_layers(self):
		layers = []
		cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
		cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
		in_channels = 1
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if self.batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v
		return nn.Sequential(*layers)

#print(SSD300())
