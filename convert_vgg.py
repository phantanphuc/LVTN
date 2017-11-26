'''Convert pretrained VGG model to SSD.

VGG model download from PyTorch model zoo: https://download.pytorch.org/models/vgg16-397923af.pth
'''
import torch
import sys

from ssd import SSD300

#from ssd import SSD300


def convertToModel(path):

	vgg = torch.load('./raw_pretrain/pretrain_selectedtotrain.pth')

	vgg = vgg['net']

	#print(vgg.keys())


	ssd_net = SSD300()


	layer_indices = [0,1,3,4,7,8,10,11,14,15,17,18,20,21,24,25,27,28,30,31]#,24,25,27,28,30,31]


	prefix = 'module.'

	for layer_idx in layer_indices:
		ssd_net.base[layer_idx].weight.data = vgg[prefix + 'features.%d.weight' % layer_idx]
		ssd_net.base[layer_idx].bias.data = vgg[prefix + 'features.%d.bias' % layer_idx]

	# [24,26,28]
	ssd_net.conv5_1.weight.data = vgg[prefix + 'features.34.weight']
	ssd_net.conv5_1.bias.data = vgg[prefix + 'features.34.bias']
	ssd_net.conv5_2.weight.data = vgg[prefix + 'features.37.weight']
	ssd_net.conv5_2.bias.data = vgg[prefix + 'features.37.bias']
	ssd_net.conv5_3.weight.data = vgg[prefix + 'features.40.weight']
	ssd_net.conv5_3.bias.data = vgg[prefix + 'features.40.bias']

	ssd_net.norm5_1.weight.data = vgg[prefix + 'features.35.weight']
	ssd_net.norm5_1.bias.data = vgg[prefix + 'features.35.bias']
	ssd_net.norm5_2.weight.data = vgg[prefix + 'features.38.weight']
	ssd_net.norm5_2.bias.data = vgg[prefix + 'features.38.bias']
	ssd_net.norm5_3.weight.data = vgg[prefix + 'features.41.weight']
	ssd_net.norm5_3.bias.data = vgg[prefix + 'features.41.bias']


	torch.save(ssd_net.state_dict(), path)
