
# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
	import matplotlib
	# Agg backend runs without a display
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

# Root directory of the cells sample
ROOT_DIR = os.path.abspath("../../")

# Directory of entire project (not just cells sample)
PROJECT_DIR = os.path.abspath("../../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/cells/")

# Number of images in the dataset 
NUM_IMAGES_TOTAL = 670
NUM_IMAGES_VAL = 67 

############################################################
#  Configurations
############################################################

class CellConfig(Config):
	"""Configuration for training on the biomedical image segmentation dataset."""

	NAME = 'cell'

	# NUMBER OF GPUs to use. For CPU training, use 1
	GPU_COUNT = 1

	# Adjust depending on your GPU memory
	IMAGES_PER_GPU = 10

	# Number of classes (including background)
	NUM_CLASSES = 2

	# These are both approximate because Tensorbaord updates are saved after each epoch
	# STEPS_PER_EPOCH = 700
	# VALIDATION_STEPS = 80

	STEPS_PER_EPOCH = (NUM_IMAGES_TOTAL - NUM_IMAGES_VAL) // IMAGES_PER_GPU
	VALIDATION_STEPS = max(1, NUM_IMAGES_VAL // IMAGES_PER_GPU)

	# Don't exclude based on confidence. Since we have two classes
	# then 0.5 is the minimum anyway as it picks between nucleus and BG
	DETECTION_MIN_CONFIDENCE = 0

	# Backbone network architecture
	# Supported values are: resnet50, resnet101
	BACKBONE = "resnet50"

	# Input image resizing
	# Generally, use the "square" resizing mode for training and inferencing
	# and it should work well in most cases. In this mode, images are scaled
	# up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
	# scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
	# padded with zeros to make it a square so multiple images can be put
	# in one batch.
	# Available resizing modes:
	# none:   No resizing or padding. Return the image unchanged.
	# square: Resize and pad with zeros to get a square image
	#         of size [max_dim, max_dim].
	# pad64:  Pads width and height with zeros to make them multiples of 64.
	#         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
	#         up before padding. IMAGE_MAX_DIM is ignored in this mode.
	#         The multiple of 64 is needed to ensure smooth scaling of feature
	#         maps up and down the 6 levels of the FPN pyramid (2**6=64).
	# crop:   Picks random crops from the image. First, scales the image based
	#         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
	#         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
	#         IMAGE_MAX_DIM is not used in this mode.
	IMAGE_RESIZE_MODE = "crop" #should change depending on data version
	IMAGE_MIN_DIM = 512
	IMAGE_MAX_DIM = 512
	# Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
	# up scaling. For example, if set to 2 then images are scaled up to double
	# the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
	# Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
	IMAGE_MIN_SCALE = 2.0

	# Length of square anchor side in pixels --> Scaled down for cell nuclei
	RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

	# ROIs kept after non-maximum supression (training and inference)
	POST_NMS_ROIS_TRAINING = 1000
	POST_NMS_ROIS_INFERENCE = 2000

	# Non-max suppression threshold to filter RPN proposals.
	# You can increase this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.9

	# How many anchors per image to use for RPN training
	RPN_TRAIN_ANCHORS_PER_IMAGE = 64

	# Minimum probability value to accept a detected instance
	# ROIs below this threshold are skipped
	DETECTION_MIN_CONFIDENCE = 0

	# Number of ROIs per image to feed to classifier/mask heads
	# The Mask RCNN paper uses 512 but often the RPN doesn't generate
	# enough positive proposals to fill this and keep a positive:negative
	# ratio of 1:3. You can increase the number of proposals by adjusting
	# the RPN NMS threshold.
	TRAIN_ROIS_PER_IMAGE = 128

	# Max number of final detections
	DETECTION_MAX_INSTANCES = 300

	# Maximum number of ground truth instances to use in one image
	MAX_GT_INSTANCES = 200

	# If enabled, resizes instance masks to a smaller size to reduce
	# memory load. Recommended when using high-resolution images.
	USE_MINI_MASK = True
	MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

	# Image mean (RGB)
	# MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

	# Mean Pixel for Processed
	MEAN_PIXEL = np.array([26.04922138, 26.00638582, 26.00646754])

	# Learning rate and momentum
	# The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
	# weights to explode. Likely due to differences in optimzer
	# implementation.
	LEARNING_RATE = 0.001
	LEARNING_MOMENTUM = 0.9

	# Weight decay regularization
	WEIGHT_DECAY = 0.0001

# Configuration for test set
class CellInferenceConfig(CellConfig):
	# Set batch size to 1 to run one image at a time
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	# Don't resize imager for inferencing
	IMAGE_RESIZE_MODE = "pad64" # ------> consider chaging if pipeline changes
	# Non-max suppression threshold to filter RPN proposals.
	# You can increase this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.7

class CellDataset(utils.Dataset):

	def load_cells(self, dataset_dir, subset, version):
		""" Load subset of cell image dataset.
		dataset_dir: Root directory of the dataset
		subset: Subset to load. Name of the sub-directory,
						such as stage1_train, stage1_test, ...etc.
		"""
		self.add_class("cell", 1, "cell")
		self.version = version

		# Code to handle multiple dataset versions:
		assert version in ['processed', 'raw']
		assert subset in ['train', 'test', 'val']
		dataset_dir = os.path.join(PROJECT_DIR, dataset_dir)
		print(dataset_dir)
		if version == 'raw':
			print("Running on raw image data")
			val_image_ids_file = os.path.join(dataset_dir, 'val_image_ids.txt')
			with open(val_image_ids_file) as f:
				s = f.read()
				val_image_ids = s.split()
			if subset == 'val':
				dataset_dir = os.path.join(dataset_dir, 'stage1_train')
				image_ids = val_image_ids
			elif subset == 'train':
				dataset_dir = os.path.join(dataset_dir, 'stage1_train')
				image_ids = next(os.walk(dataset_dir))[1]
				image_ids = list(set(image_ids) - set(val_image_ids))
			else:
				dataset_dir = os.path.join(dataset_dir, 'stage2_test')
				image_ids = next(os.walk(dataset_dir))[1]
			for image_id in image_ids:
				self.add_image(
					"cell",
					image_id=image_id,
					path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))
		else:
			print("Running on processed image data")
			if subset == 'val':
				dataset_dir = os.path.join(dataset_dir, 'processed_dev', 'images')
			elif subset == 'train':
				dataset_dir = os.path.join(dataset_dir, 'processed_train', 'images')
			else:
				dataset_dir = os.path.join(dataset_dir, 'processed_test2', 'images')
			for filename in os.listdir(dataset_dir):
				if filename.endswith('.png'):
					image_id = filename[:filename.find('.')]
					self.add_image(
						"cell",
						image_id=image_id,
						path=os.path.join(dataset_dir, filename))


	def load_mask(self, image_id):
		info = self.image_info[image_id]

		# get mask_dir depending on the data version
		mask_dir = None  
		if self.version == 'raw':
			mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
		else:
			mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), 'masks', info['id'])

		# fetch masks from directory: mask_dir
		mask = []
		for f in next(os.walk(mask_dir))[2]:
			if f.endswith('.png'):
				m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
				mask.append(m)
		mask = np.stack(mask, axis=-1)
		return mask, np.ones([mask.shape[-1]], dtype=np.int32)

	def image_reference(self, image_id):
		# returns the path of the image
		info = self.image_info[image_id]
		if info['source'] == 'cell':
			return info['id'] 
		else:
			super(self.__class__, self).image_reference(image_id)


#---------------------------- Training --------------------------------------#
def train(model, dataset_dir, subset, version):

	# Training dataset.
	dataset_train = CellDataset();
	dataset_train.load_cells(dataset_dir, subset, version)
	dataset_train.prepare()

	# Validation dataset
	dataset_val = CellDataset()
	dataset_val.load_cells(dataset_dir, 'val', version)
	dataset_val.prepare()

	# Image augmentation
	# http://imgaug.readthedocs.io/en/latest/source/augmenters.html
	augmentation = iaa.SomeOf((0, 2), [
			iaa.Fliplr(0.5),
			iaa.Flipud(0.5),
			iaa.OneOf([iaa.Affine(rotate=90),
								 iaa.Affine(rotate=180),
								 iaa.Affine(rotate=270)]),
			iaa.Multiply((0.8, 1.5)),
			iaa.GaussianBlur(sigma=(0.0, 5.0))
	])

	# Adjust model layer training below
	print("Training network heads")
	model.train(dataset_train, dataset_val,
								learning_rate=config.LEARNING_RATE,
								epochs=15,
								augmentation=augmentation,
								layers='heads')

	print("Training all layers")
	model.train(dataset_train, dataset_val,
							learning_rate=config.LEARNING_RATE,
							epochs=30,
							augmentation=augmentation,
							layers='all')

#------------------------- RLE Encoding ------------------------------------#
def rle_encode(mask):
	"""Encodes a mask in Run Length Encoding (RLE).
	Returns a string of space-separated values.
	"""
	assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
	# Flatten it column wise
	m = mask.T.flatten()
	# Compute gradient. Equals 1 or -1 at transition points
	g = np.diff(np.concatenate([[0], m, [0]]), n=1)
	# 1-based indicies of transition points (where gradient != 0)
	rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
	# Convert second index in each pair to lenth
	rle[:, 1] = rle[:, 1] - rle[:, 0]
	return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
	"""Decodes an RLE encoded list of space separated
	numbers and returns a binary mask."""
	rle = list(map(int, rle.split()))
	rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
	rle[:, 1] += rle[:, 0]
	rle -= 1
	mask = np.zeros([shape[0] * shape[1]], np.bool)
	for s, e in rle:
		assert 0 <= s < mask.shape[0]
		assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
		mask[s:e] = 1
	# Reshape and transpose
	mask = mask.reshape([shape[1], shape[0]]).T
	return mask


def mask_to_rle(image_id, mask, scores):
	"Encodes instance masks to submission format."
	assert mask.ndim == 3, "Mask must be [H, W, count]"
	# If mask is empty, return line with image ID only
	if mask.shape[-1] == 0:
		return "{},".format(image_id)
	# Remove mask overlaps
	# Multiply each instance mask by its score order
	# then take the maximum across the last dimension
	order = np.argsort(scores)[::-1] + 1  # 1-based descending
	mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
	# Loop over instance masks
	lines = []
	for o in order:
		m = np.where(mask == o, 1, 0)
		# Skip if empty
		if m.sum() == 0.0:
			continue
		rle = rle_encode(m)
		lines.append("{}, {}".format(image_id, rle))
	return "\n".join(lines)

#--------------------------- Detection (Testing) ----------------------------#

def detect(model, dataset_dir, subset, version):
	print("Running on {}".format(dataset_dir))

	# Create dictionary for storing test results
	if not os.path.exists(RESULTS_DIR):
		os.makedirs(RESULTS_DIR)
	submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
	submit_dir = os.path.join(RESULTS_DIR, submit_dir)
	os.makedirs(submit_dir)

	# load dataset
	dataset = CellDataset()
	dataset.load_cells(dataset_dir, subset, version)
	dataset.prepare()

	# Build submission output
	i = 0
	submission = []
	for image_id in dataset.image_ids:
		image = dataset.load_image(image_id)
		output = model.detect([image], verbose=0)[0] # detect only a single image at once

		# Encode image for Run-Length-Encoding
		source_id = dataset.image_info[image_id]['id']
		rle = mask_to_rle(source_id, output['masks'], output['scores'])
		submission.append(rle)

		# Only save 1/x images for speed 
		if i % 100 == 0:
			# Save images with predicted masks for visualiztion
			visualize.display_instances(
				image, output['rois'], output['masks'], output['class_ids'],
				dataset.class_names, output['scores'],
				show_bbox=False, show_mask=False,
				title='Predictions')
			print("Displayed, attempting to save")
			plt.savefig('{}/{}.png'.format(submit_dir, dataset.image_info[image_id]["id"]))
			plt.close()
		i += 1

	# Save to csv
	print("Saving to CSV")
	submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
	file_path = os.path.join(submit_dir, 'submit.csv')
	with open(file_path, 'w') as f:
		f.write(submission)
	print('Saved to', submit_dir)


#---------------------- Running from Command Line ---------------------------#

if __name__ == '__main__':
	import argparse

	# Parse command line arguments
	parser = argparse.ArgumentParser(
			description='Mask R-CNN for nuclei counting and segmentation')
	parser.add_argument("command",
											metavar="<command>",
											help="'train' or 'detect'")
	parser.add_argument('--dataset', required=False,
											metavar="/path/to/dataset/",
											help='Root directory of the dataset')
	parser.add_argument('--weights', required=True,
											metavar="/path/to/weights.h5",
											help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--logs', required=False,
											default=DEFAULT_LOGS_DIR,
											metavar="/path/to/logs/",
											help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--subset', required=False,
											metavar="Dataset sub-directory",
											help="Subset of dataset to run prediction on")
	parser.add_argument('--version', required=False,
											metavar="Data version - default is processed",
											help="Which data files to run on (raw or processed)")
	args = parser.parse_args()

	# Validate arguments
	if args.command == "train":
		assert args.dataset, "Argument --dataset is required for training"
	elif args.command == "detect":
		assert args.subset, "Provide --subset to run prediction on"

	print("Weights: ", args.weights)
	print("Dataset: ", args.dataset)
	if args.subset:
		print("Subset: ", args.subset)
	if args.version:
		print("Version: ", args.version)
		version = args.version
	else:
		version = 'processed' # default to processed data if version is ignored
	print("Logs: ", args.logs)

	# Configurations
	if args.command == "train":
		config = CellConfig()
	else:
		config = CellInferenceConfig()
	config.display()

	# Create model
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,
															model_dir=args.logs)
	else:
		model = modellib.MaskRCNN(mode="inference", config=config,
															model_dir=args.logs)

	# Select weights file to load
	if args.weights.lower() == "coco":
		weights_path = COCO_WEIGHTS_PATH
		# Download weights file
		if not os.path.exists(weights_path):
				utils.download_trained_weights(weights_path)
	elif args.weights.lower() == "last":
		# Find last trained weights
		weights_path = model.find_last()[1]
	elif args.weights.lower() == "imagenet":
		# Start from ImageNet trained weights
		weights_path = model.get_imagenet_weights()
	else:
		weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	if args.weights.lower() == "coco":
		# Exclude the last layers because they require a matching
		# number of classes
		model.load_weights(weights_path, by_name=True, exclude=[
				"mrcnn_class_logits", "mrcnn_bbox_fc",
				"mrcnn_bbox", "mrcnn_mask"])
	else:
		model.load_weights(weights_path, by_name=True)

	# Train or evaluate
	if args.command == "train":
		train(model, args.dataset, args.subset, version)
	elif args.command == "detect":
		detect(model, args.dataset, args.subset, version)
	else:
		print("'{}' is not recognized. "
					"Use 'train' or 'detect'".format(args.command))

