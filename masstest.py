import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

from NetworkConfig import *

from ssd import SSD300
from encoder import DataEncoder
from PIL import Image, ImageDraw, ImageFont

import os



DRC = args.test_dir



dictionary = {}
dictindex = []

with open('./label.txt') as f:
	content = f.readlines()
	for symbol in content:
		symbol = symbol.replace('\n','')

		split = symbol.split(' ')

		dictindex.append(split[0])

net = SSD300()
checkpoint = torch.load(args.test_model)
checkpoint['net']
net.load_state_dict(checkpoint['net'])
net.eval()



for a, b, c in os.walk(DRC):
	for file in c:
		try:
			# Load test image
			img = Image.open(DRC + '/' + file)
			img1 = img.resize((InputImgSize,InputImgSize))
			transform = transforms.Compose([transforms.ToTensor(),
											transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
			img1 = transform(img1)

			# Forward
			loc, conf = net(Variable(img1[None,:,:,:], volatile=True))

			# Decode
			data_encoder = DataEncoder()
			boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)

			fnt = ImageFont.truetype('./font/arial.ttf', 40)


			draw = ImageDraw.Draw(img)

			for i in range(len(boxes)):
				boxes[i][::2] *= img.width
				boxes[i][1::2] *= img.height
				draw.rectangle(list(boxes[i]), outline='red')

				draw.text((boxes[i][0], boxes[i][1]), dictindex[labels.numpy()[i, 0] - 1], font=ImageFont.truetype("./font/arial.ttf"))


			print('saving to: ' + args.output_dir + '/' + file)
			img.save(args.output_dir + '/' + file)
			
		except ValueError as E:
			print('err')
			

			


#for a, b, c in os.walk(DRC):
#	print(c)
#quit()

#dictindex[90]