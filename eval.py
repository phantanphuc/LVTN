#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 23:05:29 2017

@author: ngocbui
"""

from __future__ import print_function

import os
import argparse
import itertools
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from  medpy.metric import binary 

import torchvision
import torchvision.transforms as transforms

from ssd import SSD300
from datagen import ListDataset
from encoder import DataEncoder
from PIL import Image, ImageDraw, ImageFont

from torch.autograd import Variable
import pdb

use_cuda = False#torch.cuda.is_available()
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
epoch_count = 10

learning_rate = 0.001
resume = False

batch_size = 2  
dice_score = 0
ite = 0
####################################################

# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
							transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


testset = ListDataset(root='./dataset/Exp_Test/Exp_Test_BKNgoc/', list_file='./voc_data/ssd_test_BKN.txt', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)

net = SSD300()

if use_cuda:
	if resume:
		pass
        #net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])
		net.cuda()
		cudnn.benchmark = True

print('==> Resuming from checkpoint..')
#    checkpoint = torch.load('./checkpoint/ssdtrain0511_12.pth')
checkpoint = torch.load('./checkpoint/ssdtrain0511_12.pth', map_location=lambda storage, loc: storage)
checkpoint['net']
net.load_state_dict(checkpoint['net'])

def eval():
	print('\nEval')
	net.eval()
	global dice_score
	global ite 
	target_boxes = testset.boxes
	target_img = create_binaryImage(target_boxes, False)
#	for i in range(2):
#		img = Image.fromarray(target_img[i])
#		img.show()
#	pdb.set_trace()
	for batch_idx, (images, loc_targets, conf_targets) in enumerate(testloader):
		if use_cuda:
			images = images.cuda()
			loc_targets = loc_targets.cuda()
			conf_targets = conf_targets.cuda()

		images = Variable(images, volatile=True)
#		loc_targets = Variable(loc_targets)
#		conf_targets = Variable(conf_targets)

		loc_preds, conf_preds = net(images)
		
		data_encoder = DataEncoder()
#		pdb.set_trace()
		conf_preds_list = []
		for i in range(batch_size):
			s_conf = F.softmax(conf_preds[i]).data
			conf_preds_list.append(s_conf)
#		pdb.set_trace() 
		try:
			boxes, labels, scores = data_encoder.decodeforbatch(loc_preds.data, conf_preds_list)
		
#		pdb.set_trace()
			predicted_img = create_binaryImage(boxes)
		
			for i in range(batch_size):
				dice_score += binary.dc(predicted_img[i],target_img[batch_idx + i])
				ite+=1
#			pdb.set_trace()
				print('ite', i)
				print(binary.dc(predicted_img[i],target_img[batch_idx + i]))
#			img = Image.fromarray(target_img[batch_idx + i])
#			img.show()
		except:
			print('err') 

def create_binaryImage(boxes, isPred=True):
	img = np.zeros((len(boxes),300,300))
#	pdb.set_trace()
	for id_img in range(img.shape[0]):
		for i in range(len(boxes[id_img])):
#			pdb.set_trace()
			if(isPred):
				boxes[id_img][i,::2] *= img.shape[1]
				boxes[id_img][i,1::2] *= img.shape[2]
			for y in range(int(round(boxes[id_img][i,1])), int(round(boxes[id_img][i,3]))):
				for x in range(int(round(boxes[id_img][i,0])), int(round(boxes[id_img][i,2]))):
					img[id_img,y,x] = 255 
#			pdb.set_trace()
#		imgg = Image.fromarray(img[id_img])
#		imgg.show()
#	pdb.set_trace()
	return img
for epoch in range(1):
	eval()
print(dice_score/ite)