import torchvision.transforms as transforms
import imgaug as ia
import imgaug.augmenters as iaa
import skimage
import numpy as np

# Augment original training images using PyTorch torchvision.transforms.
# ----------------------------------------------------------------------

# Change these probabilities if you want
transformer_probs = {'h_flip':0.5, 'v_flip':0.5}

basic_transformer = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomCrop(size=192, padding=0), # randomly select a 192x192 portion of the image. Padding=0 because we'll resize
    transforms.Resize([256 ,256]),
    transforms.ToTensor()])

'''
train_transformer = transforms.Compose([
    transforms.Resize(256),  # resize the image to 256x256 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(p=transformer_probs['h_flip']),  # randomly flip image horizontally
    transforms.RandomVerticalFlip(p=transformer_probs['v_flip']), # randomly flip image vertically
    transforms.ToTensor()])  # transform it into a torch tensor
'''

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize([256, 256]),  # resize the image to 256x256 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor

#transformer_list = [basic_transformer, train_transformer, eval_transformer]

## Takes in image and mask (as numpy arrays), normalizes both to be between 0 and 1
## and resizes them to 256 * 256.  Also inverts images that have an average pixel intensity >
## a specific threshold to acocunt for variations in microscopy
def resize_and_normalize(image, mask):
    if np.sum(image) / (1. * image.size) > 25000:
        image = skimage.util.invert(image)
    image = skimage.util.img_as_ubyte(image)
    image = skimage.transform.resize(image, (256, 256))
    if mask is not None:
        mask = skimage.util.img_as_ubyte(mask).astype(float) / 255
        mask = np.around(skimage.transform.resize(mask, (256, 256))) * 255
        return image, mask
    return image, None

def train_transform(image, mask):
    h, w = image.shape
    tf = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.045))),
        #iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=(0, 0.5), sigma=0.2)),
        #iaa.Dropout(p=(0, 0.1))
    ])
    deterministic_tf = tf.to_deterministic()
    image, mask =  deterministic_tf.augment_images(image.reshape((1, h, w))), deterministic_tf.augment_images(mask.reshape((1, h, w)))
    return image.reshape((h, w)), mask.reshape((h, w))
