#####################################################
############## IMPORT ###############################
#####################################################
import argparse
import dl_gdrive
import os
import zipfile
import platform

#####################################################
############## PARAMETER ############################
#####################################################

###
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
###
######### HYPER PARAM
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--decay', default=1e-4, type=float, help='decay')
parser.add_argument('--use_cuda', default=True, type=bool, help='Use CUDA for training')
parser.add_argument('--epoch_count', default=10, type=int, help='Number of training epoch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--resume_mode', default='pretrain', type=str, help='Continue training mode: \'none\': From nothing,\'pretrain\': From pretrain model, \'continue\': Continue from SSD Model ')

######### Core Component
parser.add_argument('--using_python_2', default=True, type=bool, help='Current python version')
parser.add_argument('--class_count', default=107, type=int, help='Number of classes')
parser.add_argument('--network', default='SSD500', type=str, help='network type: \'SSD300\': use original SSD300, \'SSD500\': Improved version ')
parser.add_argument('--resuming_model', default='./trainingmodel/ssd.pth', type=str, help='Model to load (Only valid for resume_mode: pretrain and continue)')

######### PATH 
parser.add_argument('--train_dir', default='./dataset/train', type=str, help='training set directory')
parser.add_argument('--train_meta', default='./metafile/train.txt', type=str, help='training set metafile location')

parser.add_argument('--validate_dir', default='./dataset/train', type=str, help='validation set directory')
parser.add_argument('--validate_meta', default='./metafile/train.txt', type=str, help='validateion set metafile location')

parser.add_argument('--output_directory', default='./checkpoint', type=str, help='Output model directory')
parser.add_argument('--output_format', default='ckpt_%d.pth', type=str, help='Format of output model\'s name, this file must contain symbol %%d for indexing purpose [For example: ckpt_%%d.pth]')

######### MISC.
parser.add_argument('--epoch_cycle', default=50, type=int, help='For output model name format')
parser.add_argument('--upload_model', default=True, type=bool, help='Upload trained model after training process')


#############################
###### ARG for masstest #####
#############################

parser.add_argument('--test_dir', default='./dataset/test', type=str, help='path of test directory')
parser.add_argument('--test_model', default='./trainingmodel/ckpt_resize_scale_p2_500_4.pth', type=str, help='path of test model')
parser.add_argument('--output_dir', default='./result', type=str, help='path of output image')

##########################################################
################ PRE - INITIALIZATION ####################
##########################################################
args = parser.parse_args()

##############################
#### NETWORK TYPE
# 0: SSD 300
# 1: Deprecated
# 2: SSD 500 improved
# 3: Deprecated
InputImgSize = 300
Network_type = 0
if args.network == 'SSD400':
	Network_type = 1
	InputImgSize = 400
elif args.network == 'SSD500':
	Network_type = 2
	InputImgSize = 500
elif args.network == 'SSD600':
	Network_type = 3
	InputImgSize = 600

################ NETWORK ACHITECHTURE ####################

if Network_type == 0: #SSD 300
	feature_map_sizes = (38, 19, 10, 5, 3, 1)
	steps_raw = (8, 16, 32, 64, 100, 300)
	aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,))
	min_ratio = 20
	max_ratio = 90
	min_scale = 0.1

	in_planes = [512,1024,512,256,256,256]

elif Network_type == 1: # SSD 400 (deprecated)
	feature_map_sizes = (50, 25, 13, 7, 5, 3, 1)
	steps_raw = (8, 16, 32, 64, 80, 133, 400)
	aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2,), (2,), (2,))
	min_ratio = 20
	max_ratio = 90
	min_scale = 0.1
	
	in_planes = [512,1024,512,256,256,256,256]

elif Network_type == 2: # SSD 500
	feature_map_sizes = (63, 32, 16, 8, 4, 2, 1)
	steps_raw = (8, 16, 32, 64, 128, 250, 500)
	aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2, ), (2, ), (2, ))
	min_ratio = 8
	max_ratio = 50
	min_scale = 0.03

	in_planes = [512,1024,512,256,256,256,256]

elif Network_type == 3: # SSD 600 (Deprecated)
	feature_map_sizes = (75, 38, 19, 10, 8, 6, 4, 2) 
	steps_raw = (8, 16, 32, 60, 75, 100, 150, 300)
	aspect_ratios = ((2,), (2,3), (2,3), (2,3), (2, 3), (2,), (2,), (2, ))
	min_ratio = 20
	max_ratio = 90
	min_scale = 0.1

	in_planes = [512,1024,512,256,256,256,256, 256]

num_anchors = []

for aspectratio in aspect_ratios:
	num_anchors.append(len(aspectratio) * 2 + 2)

###### For masstest

import sys
if os.path.basename(sys.argv[0]) != 'train.py':

	if not os.path.isdir(args.test_dir):

		print('testing directory not found ...')
		print('Downloading testset ...')

		os.makedirs(args.test_dir)

		gdrive_url = 'https://drive.google.com/file/d/1UMZykU_8EEEsahopT405n98t5UfkA70V/view?usp=sharing'
		outpath = args.test_dir + '/test1.zip'
		downloader = dl_gdrive.GdriveDownload(gdrive_url.strip(), outpath.strip())
		downloader.download()
		gdrive_url = 'https://drive.google.com/file/d/1RC5GCkpUPqBPkaLcMfcwcB6Bo2mtZPLu/view?usp=sharing'
		outpath = args.test_dir + '/test2.zip'
		downloader = dl_gdrive.GdriveDownload(gdrive_url.strip(), outpath.strip())
		downloader.download()


		zip_ref = zipfile.ZipFile(args.test_dir + '/test1.zip', 'r')
		zip_ref.extractall(args.test_dir)
		zip_ref.close()

		zip_ref = zipfile.ZipFile(args.test_dir + '/test2.zip', 'r')
		zip_ref.extractall(args.test_dir)
		zip_ref.close()

		os.remove(args.test_dir + '/test1.zip')
		os.remove(args.test_dir + '/test2.zip')


	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)

	if not os.path.isfile(args.test_model):
		print('test model not found ....')

else: ## For training

	###### 

	if args.resuming_model == './model/ssd.pth':
		if args.resume_mode == 'pretrain':
			print('Use are using network with resume_mode: pretrain')
			print('Be sure to specify path of pretrain model using argument: --resuming_model or default path ./model/ssd.pth will be used')
		if args.resume_mode == 'continue':
			print('Use are using network with resume_mode: pretrain')
			print('Be sure to specify path of model using argument: --resuming_model or default path ./model/ssd.pth will be used')

	if '%d' not in args.output_format:
		print('--output_format param must contain %d')
		quit()

	##########################################################
	################ DATASET CHECKING ########################
	##########################################################

	print('Checking Dataset Availability ...')
	if not os.path.isdir(args.train_dir):
		os.makedirs(args.train_dir)
		print ('training dataset Unavailable... ')
		print ('Downloading training Dataset ....')

		gdrive_url = 'https://drive.google.com/file/d/1jXkAHT_CfB-kHaOQI09XWdFTUlWIqfj0/view?usp=sharing'
		outpath = args.train_dir + '/train.zip'

		downloader = dl_gdrive.GdriveDownload(gdrive_url.strip(), outpath.strip())
		downloader.download()

		print ('Download Complete ....')

		zip_ref = zipfile.ZipFile(args.train_dir + '/train.zip', 'r')
		zip_ref.extractall(args.train_dir)
		zip_ref.close()

		os.remove(args.train_dir + '/train.zip')

		print ('Training dataset preparation complete ....')

	#-----------------------------------------

	if not os.path.isdir(args.validate_dir):
		os.makedirs(args.validate_dir)
		print ('Validation dataset Unavailable... ')
		print ('Downloading Validation Dataset ....')

		gdrive_url = 'https://drive.google.com/file/d/1jXkAHT_CfB-kHaOQI09XWdFTUlWIqfj0/view?usp=sharing'
		outpath = args.validate_dir + '/validation.zip'

		downloader = dl_gdrive.GdriveDownload(gdrive_url.strip(), outpath.strip())
		downloader.download()

		print ('Download Complete ....')

		zip_ref = zipfile.ZipFile(args.validate_dir + '/validation.zip', 'r')
		zip_ref.extractall(args.validate_dir)
		zip_ref.close()

		os.remove(args.validate_dir + '/validation.zip')

		print ('Validation dataset preparation complete ....')

	################

	print('Checking Model Availability ...')
	if args.resume_mode == 'pretrain':

		if not os.path.isfile(args.resuming_model):  

			if not os.path.isdir(os.path.dirname(args.resuming_model)):
				os.makedirs(os.path.dirname(args.resuming_model))

			print('pretrain model is not found, downloading from server ....')

			if not os.path.isdir('./raw_pretrain'):
				os.makedirs('./raw_pretrain')

			if not os.path.isfile('./raw_pretrain/pretrain_selectedtotrain.pth'): 

				gdrive_url = 'https://drive.google.com/file/d/122UDu5_3FPOvgpEkqo4lbUVVACBsRtZt/view?usp=sharing'
				outpath = './raw_pretrain/pretrain_selectedtotrain.pth'

				downloader = dl_gdrive.GdriveDownload(gdrive_url.strip(), outpath.strip())
				downloader.download()

			import convert_vgg
			convert_vgg.convertToModel(args.resuming_model)

	elif args.resume_mode == 'continue':
		if not os.path.isfile(args.resuming_model): 
			print('model not found ... ')
			print('Please make sure path to existing model is valid')
			print('If you have no model available, use resume_mode : \'pretrain\' , the the network will download a pretrained model from GoogleDrive')
			quit()

	##############################################
	MODEL_UPLOAD = False
	if args.upload_model:
		if platform.system() == 'Windows':
			print('Uploading is unsupported for windows')
		else:
			MODEL_UPLOAD = True

###########################################
# Debugging session
# H:\anaconda_3\Install\python train.py
# Drive: https://drive.google.com/drive/folders/1Zi8SR5BhU2MxAEoF2qn1ovK1Yrl3hdMb?usp=sharing
# Train Dataset: https://drive.google.com/file/d/1jXkAHT_CfB-kHaOQI09XWdFTUlWIqfj0/view?usp=sharing
# Pretrain model: https://drive.google.com/file/d/1jEpWfFCj11mcEDryVYMdJDCDLWyoDI_4/view?usp=sharing