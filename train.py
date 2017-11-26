from __future__ import print_function

import os
import itertools
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


from NetworkConfig import *
from ssd import SSD300
from datagen import ListDataset
from multibox_loss import MultiBoxLoss



############ Variable #########################3333

best_loss = float('inf')  
start_epoch = 0  # start from epoch 0 or last epoch
current_best_model = ''

####################################################

# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

trainset = ListDataset(root=args.train_dir, list_file=args.train_meta, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)

testset = ListDataset(root=args.validate_dir, list_file=args.validate_meta, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,drop_last=True)


net = SSD300()

if args.use_cuda:
	net.cuda()
	cudnn.benchmark = True

if args.resume_mode == 'continue':
	print('==> Resuming from checkpoint..')
	checkpoint = torch.load(args.resuming_model)
	net.load_state_dict(checkpoint['net'])
	best_loss = checkpoint['loss']
	start_epoch = checkpoint['epoch']
elif args.resume_mode == 'pretrain':
	net.load_state_dict(torch.load(args.resuming_model))


criterion = MultiBoxLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainloader):

		if (numpy.max(conf_targets.numpy()) == 0):
			continue


		if args.use_cuda:
			images = images.cuda()
			loc_targets = loc_targets.cuda()
			conf_targets = conf_targets.cuda()

		images = Variable(images)
		loc_targets = Variable(loc_targets)
		conf_targets = Variable(conf_targets)

		optimizer.zero_grad()


		loc_preds, conf_preds = net(images)

		#print('------------------')
		#print(loc_preds.data.numpy().shape)
		#print(conf_preds.data.numpy().shape)
		#print(loc_targets.data.numpy().shape)
		#print(conf_targets.data.numpy().shape)

		loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)

		loss.backward()
		optimizer.step()

		train_loss += loss.data[0]
		print('%.3f %.3f' % (loss.data[0], train_loss/(batch_idx+1)))


def test(epoch):
	print('\nTest')
	net.eval()
	test_loss = 0
	for batch_idx, (images, loc_targets, conf_targets) in enumerate(testloader):
		if args.use_cuda:
			images = images.cuda()
			loc_targets = loc_targets.cuda()
			conf_targets = conf_targets.cuda()

		images = Variable(images, volatile=True)
		loc_targets = Variable(loc_targets)
		conf_targets = Variable(conf_targets)

		loc_preds, conf_preds = net(images)
		loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
		test_loss += loss.data[0]
		print('%.3f %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

	# Save checkpoint.
	global best_loss
	global current_best_model

	test_loss /= len(testloader)
	if test_loss < best_loss:
		print('Saving..')
		state = {
			'net': net.state_dict(),
			'loss': test_loss,
			'epoch': epoch,
		}
		if not os.path.isdir(args.output_directory):
			os.mkdir(args.output_directory)
			

		current_best_model = args.output_format % (epoch % args.epoch_cycle)
		save_path = args.output_directory + '/' + current_best_model
		torch.save(state, save_path)
		best_loss = test_loss


for epoch in range(start_epoch, start_epoch + args.epoch_count):
	train(epoch)
	test(epoch)
	#quit()

#-------- UPLOADING --------------------

