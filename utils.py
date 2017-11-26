'''Some helper functions for PyTorch, including:
	- get_mean_and_std: calculate the mean and std value of dataset.
	- msr_init: net parameter initialization.
	- progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn


def get_mean_and_std(dataset, max_load=10000):
	'''Compute the mean and std value of dataset.'''
	# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
	mean = torch.zeros(3)
	std = torch.zeros(3)
	print('==> Computing mean and std..')
	N = min(max_load, len(dataset))
	for i in range(N):
		print(i)
		im,_,_ = dataset.load(1)
		for j in range(3):
			mean[j] += im[:,j,:,:].mean()
			std[j] += im[:,j,:,:].std()
	mean.div_(N)
	std.div_(N)
	return mean, std

def mask_select(input, mask, dim):
	'''Select tensor rows/cols using a mask tensor.

	Args:
	  input: (tensor) input tensor, sized [N,M].
	  mask: (tensor) mask tensor, sized [N,] or [M,].
	  dim: (tensor) mask dim.

	Returns:
	  (tensor) selected rows/cols.

	Example:
	>>> a = torch.randn(4,2)
	>>> a
	-0.3462 -0.6930
	 0.4560 -0.7459
	-0.1289 -0.9955
	 1.7454  1.9787
	[torch.FloatTensor of size 4x2]
	>>> i = a[:,0] > 0
	>>> i
	0
	1
	0
	1
	[torch.ByteTensor of size 4]
	>>> masked_select(a, i, 0)
	0.4560 -0.7459
	1.7454  1.9787
	[torch.FloatTensor of size 2x2]
	'''
	index = mask.nonzero().squeeze(1)
	return input.index_select(dim, index)

def msr_init(net):
	'''Initialize layer parameters.'''
	for layer in net:
		if type(layer) == nn.Conv2d:
			n = layer.kernel_size[0]*layer.kernel_size[1]*layer.out_channels
			layer.weight.data.normal_(0, math.sqrt(2./n))
			layer.bias.data.zero_()
		elif type(layer) == nn.BatchNorm2d:
			layer.weight.data.fill_(1)
			layer.bias.data.zero_()
		elif type(layer) == nn.Linear:
			layer.bias.data.zero_()


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
