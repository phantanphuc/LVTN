'''Encode target locations and labels.'''
import torch

import math
import itertools

from NetworkConfig import *

import numpy
numpy.set_printoptions(threshold=numpy.nan)

class DataEncoder:
	def __init__(self):
		'''Compute default box sizes with scale and aspect transform.'''

		

		scale = float(InputImgSize)


		
		step = int(math.floor((max_ratio - min_ratio) / (len(feature_map_sizes) - 2)))

		sizes_raw = []
		for ratio in range(min_ratio, max_ratio + 1 + step, step):
			sizes_raw.append(scale * ratio / 100.)
		sizes_raw = [scale * min_scale] + sizes_raw
			
		if Network_type == 4:
			sizes_raw = [15.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 280.0]
			sizes_raw = [15.0, 40.0, 80.0 , 120.0, 185.0, 160.0, 270.0, 200.0, 355.0, 240.0, 440.0, 280.0, 500.0]


		steps = [s / scale for s in steps_raw]
		sizes = [s / scale for s in sizes_raw]
		
		if Network_type == 4:
			#feature_map_sizes = (63, 32, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1)
			cross_sizes_raw = []
			cross_sizes_raw.append(sizes[0] * sizes[1]) # 15 40
			cross_sizes_raw.append(sizes[1] * sizes[2]) # 40 80
			cross_sizes_raw.append(sizes[2] * sizes[3]) # 80 120
			cross_sizes_raw.append(sizes[2] * sizes[4]) # 80 185
			cross_sizes_raw.append(sizes[3] * sizes[5]) # 120 160
			cross_sizes_raw.append(sizes[4] * sizes[6]) # 185 270
			cross_sizes_raw.append(sizes[5] * sizes[7]) # 160 200
			cross_sizes_raw.append(sizes[6] * sizes[8]) # 270 355
			cross_sizes_raw.append(sizes[7] * sizes[9]) # 200 240
			cross_sizes_raw.append(sizes[8] * sizes[10]) # 355 440
			cross_sizes_raw.append(sizes[9] * sizes[11]) # 240 280
			cross_sizes_raw.append(sizes[10] * sizes[12]) # 440 500



		num_layers = len(feature_map_sizes)


		boxes = []
		for i in range(num_layers):
			fmsize = feature_map_sizes[i]
			for h,w in itertools.product(range(fmsize), repeat=2):
				cx = (w + 0.5)*steps[i]
				cy = (h + 0.5)*steps[i]

				s = sizes[i]
				boxes.append((cx, cy, s, s))

				if Network_type == 4:
					s = math.sqrt(cross_sizes_raw[i])
					boxes.append((cx, cy, s, s))
				else:
					s = math.sqrt(sizes[i] * sizes[i+1])
					boxes.append((cx, cy, s, s))

				s = sizes[i]
				for ar in aspect_ratios[i]:
					boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
					boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

		self.default_boxes = torch.Tensor(boxes)


	def iou(self, box1, box2):
		'''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

		Args:
		  box1: (tensor) bounding boxes, sized [N,4].
		  box2: (tensor) bounding boxes, sized [M,4].

		Return:
		  (tensor) iou, sized [N,M].
		'''
		N = box1.size(0)
		M = box2.size(0)

		lt = torch.max(
			box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
			box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
		)

		rb = torch.min(
			box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
			box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
		)

		wh = rb - lt  # [N,M,2]
		wh[wh<0] = 0  # clip at 0
		inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

		area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
		area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
		area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
		area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

		iou = inter / (area1 + area2 - inter)
		return iou

	def encode(self, boxes, classes, threshold=0.5):
		'''Transform target bounding boxes and class labels to SSD boxes and classes.

		Match each object box to all the default boxes, pick the ones with the
		Jaccard-Index > 0.5:
			Jaccard(A,B) = AB / (A+B-AB)

		Args:
		  boxes: (tensor) object bounding boxes (xmin,ymin,xmax,ymax) of a image, sized [#obj, 4].
		  classes: (tensor) object class labels of a image, sized [#obj,].
		  threshold: (float) Jaccard index threshold

		Returns:
		  boxes: (tensor) bounding boxes, sized [#obj, 8732, 4].
		  classes: (tensor) class labels, sized [8732,]
		'''
		default_boxes = self.default_boxes
		num_default_boxes = default_boxes.size(0)
		num_objs = boxes.size(0)

		iou = self.iou(  # [#obj,8732]
			boxes,
			torch.cat([default_boxes[:,:2] - default_boxes[:,2:]/2,
					   default_boxes[:,:2] + default_boxes[:,2:]/2], 1)
		)

		iou, max_idx = iou.max(0)  # [1,8732]
		max_idx.squeeze_(0)		# [8732,]
		iou.squeeze_(0)			# [8732,]

		boxes = boxes[max_idx]	 # [8732,4]
		variances = [0.1, 0.2]
		cxcy = (boxes[:,:2] + boxes[:,2:])/2 - default_boxes[:,:2]  # [8732,2]
		cxcy /= variances[0] * default_boxes[:,2:]
		wh = (boxes[:,2:] - boxes[:,:2]) / default_boxes[:,2:]	  # [8732,2]
		wh = torch.log(wh) / variances[1]
		loc = torch.cat([cxcy, wh], 1)  # [8732,4]

		conf = 1 + classes[max_idx]   # [8732,], background class = 0
		

		iou[conf == 70] *= 1.2


		conf[iou<threshold] = 0	   # background

		#print(conf.numpy())
		#idx = numpy.zeros(150)
		#for i in conf:
		#	idx[i] += 1
		#for i in range(150):
		#	if (idx[i] > 0.5):
		#		print(i, idx[i])
		#quit()

		return loc, conf

	def nms(self, bboxes, scores, threshold=0.15, mode='union'):
		'''Non maximum suppression.

		Args:
		  bboxes: (tensor) bounding boxes, sized [N,4].
		  scores: (tensor) bbox scores, sized [N,].
		  threshold: (float) overlap threshold.
		  mode: (str) 'union' or 'min'.

		Returns:
		  keep: (tensor) selected indices.

		Ref:
		  https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
		'''
		x1 = bboxes[:,0]
		y1 = bboxes[:,1]
		x2 = bboxes[:,2]
		y2 = bboxes[:,3]

		areas = (x2-x1) * (y2-y1)
		_, order = scores.sort(0, descending=True)

		keep = []
		while order.numel() > 0:
			i = order[0]
			keep.append(i)

			if order.numel() == 1:
				break

			xx1 = x1[order[1:]].clamp(min=x1[i])
			yy1 = y1[order[1:]].clamp(min=y1[i])
			xx2 = x2[order[1:]].clamp(max=x2[i])
			yy2 = y2[order[1:]].clamp(max=y2[i])

			w = (xx2-xx1).clamp(min=0)
			h = (yy2-yy1).clamp(min=0)
			inter = w*h

			if mode == 'union':
				ovr = inter / (areas[i] + areas[order[1:]] - inter)
			elif mode == 'min':
				ovr = inter / areas[order[1:]].clamp(max=areas[i])
			else:
				raise TypeError('Unknown nms mode: %s.' % mode)

			ids = (ovr<=threshold).nonzero().squeeze()
			if ids.numel() == 0:
				break
			order = order[ids+1]
		return torch.LongTensor(keep)

	def decode(self, loc, conf):
		'''Transform predicted loc/conf back to real bbox locations and class labels.

		Args:
		  loc: (tensor) predicted loc, sized [8732,4].
		  conf: (tensor) predicted conf, sized [8732,21].

		Returns:
		  boxes: (tensor) bbox locations, sized [#obj, 4].
		  labels: (tensor) class labels, sized [#obj,1].
		'''

		
		if abs(args.bacground_conf_multiplier - 1.0) > 0.001:
			for i in conf:
				i[0] = i[0] / args.bacground_conf_multiplier


		variances = [0.1, 0.2]
		wh = torch.exp(loc[:,2:]*variances[1]) * self.default_boxes[:,2:]
		cxcy = loc[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
		boxes = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)  # [8732,4]

		max_conf, labels = conf.max(1)  # [8732,1]
		ids = labels.squeeze(1).nonzero().squeeze(1)  # [#boxes,]

		keep = self.nms(boxes[ids], max_conf[ids].squeeze(1))
		return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep]


	def decodeforbatch(self, loc, conf):
		batch_size = len(conf)
		b_boxes = []
		b_labels = []
		b_scores = []
		for b_id in range(batch_size):
			boxes, labels, scores = self.decode(loc[b_id,:,:], conf[b_id][:,:])
			b_boxes.append(boxes)
			b_labels.append(labels)
			b_scores.append(scores)
		return b_boxes, b_labels, b_scores 
