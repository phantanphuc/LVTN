import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

import sys

sys.path.insert(0, './../..')

from ssd import SSD300
from encoder import DataEncoder
from PIL import Image, ImageDraw, ImageFont

import os
from NetworkConfig import *


DRC = './../LVTN_MER_SSD-master(1)/LVTN_MER_SSD-master/Network/dataset/im/'
PATH = 'HMER_2017_TEST_BKNGOC_03_6B.png'



class SSD_Core:
	def __init__(self):

		self.dictindex = []

		with open('./label.txt') as f:
			content = f.readlines()
			for symbol in content:
				symbol = symbol.replace('\n','')

				split = symbol.split(' ')

				self.dictindex.append(split[0])


		# Load model
		self.net = SSD300()
		checkpoint = torch.load(args.resuming_model)
		checkpoint['net']
		self.net.load_state_dict(checkpoint['net'])
		self.net.eval()
		
		self.data_encoder = DataEncoder()

		self.transform = transforms.Compose([transforms.ToTensor(),
										transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
		
	def generatePrediction(self, imgpath, outname):
		
		# Load test image
		img = Image.open(imgpath).convert('L')
		img1 = img.resize((InputImgSize,InputImgSize))
		
		img1 = self.transform(img1)

		# Forward
		loc, conf = self.net(Variable(img1[None,:,:,:], volatile=True))

		# Decode
		
		boxes, labels, scores = self.data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)



		draw = ImageDraw.Draw(img)

		return_str = 'null ' + str(len(boxes))
		
		boxes_np = boxes.numpy() * InputImgSize
		labels_np = labels.numpy()
		
		for i in range(len(boxes)):


			return_str = return_str + ' ' + str(int(boxes_np[i][0])) + ' ' + str(int(boxes_np[i][1])) + ' ' + str(int(boxes_np[i][2])) + ' ' + str(int(boxes_np[i][3])) + ' ' + str(int(labels_np[i][0]) - 1)
			
			boxes[i][::2] *= img.width
			boxes[i][1::2] *= img.height
			draw.rectangle(list(boxes[i]), outline='red')

			draw.text((boxes[i][0], boxes[i][1]), self.dictindex[labels.numpy()[i, 0] - 1], font=ImageFont.truetype("./font/arial.ttf"))
			#draw.text((boxes[i][0] * 300, boxes[i][1] * 300), dictindex[labels.numpy()[i, 0]], font=ImageFont.truetype("./font/arial.ttf"))

			
		img.save('./temp/' + outname)

		return return_str

