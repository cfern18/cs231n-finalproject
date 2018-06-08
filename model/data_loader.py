import random
import os

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from data_augmentation import resize_and_normalize, eval_transformer, train_transform
import numpy as np
import skimage
from skimage import io
import torch

# Adapted from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.

# Transformers moved to data_augmentation.py

class NUCLEIDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, image_dir, mask_dir, augment = False):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
        data_dir: (string) directory containing the dataset
        transform: (torchvision.transforms) transformation to apply on image
        """
        if mask_dir is not None:
            # For train and dev
            self.image_files = os.listdir(image_dir)
            self.image_files = [os.path.join(image_dir, f) for f in self.image_files if f.endswith('.png')]
            self.image_files = sorted(self.image_files)
        else:
            # For test
            self.mask_files = None
            self.test_dir = image_dir

            self.image_files = list(os.listdir(image_dir))
            self.image_files = [os.path.join(image_dir, f) for f in self.image_files if f.endswith('.png')]

        if mask_dir is not None:
            self.mask_files = os.listdir(mask_dir)
            self.mask_files = [os.path.join(mask_dir, f) for f in self.mask_files if f.endswith('.png')]
            assert len(self.mask_files) == len(self.image_files)
            self.isolated_mask_folders = []
            with os.scandir(mask_dir) as it:
                for entry in it:
                    if not entry.name.startswith('.') and entry.is_dir():
                        self.isolated_mask_folders.append(entry.path)
            self.mask_files = sorted(self.mask_files)
            self.isolated_mask_folders = sorted(self.isolated_mask_folders)

        self.augment = augment

    def __len__(self):
        # return size of dataset
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Fetch image and masks from dataset. Perform transforms on image, returns
        image and combined mask, along with folder for individual masks
        """
        image = skimage.io.imread(self.image_files[idx])  # PIL image

        if self.mask_files is None:
            img, _ = resize_and_normalize(image, None)
            img = np.expand_dims(img, axis = -1)
            return transforms.ToTensor()(img).float(), None, None #image.float(), None

        mask = skimage.io.imread(self.mask_files[idx])

        if self.augment:
            image, mask = train_transform(image, mask)

        img, mask = resize_and_normalize(image, mask)
        #mask = torch.where(mask > 0.5, 1, 0)
        img, mask = np.expand_dims(img, axis = -1), np.expand_dims(mask, axis = -1)
        assert img.shape == (256, 256, 1)

        return transforms.ToTensor()(img).float(), transforms.ToTensor()(mask).float(), self.isolated_mask_folders[idx]

        # # Concatenate the image and mask to apply transform so if we apply it to one we apply it to the other
        # stack_image = Image.fromarray(np.uint8(np.dstack((image, mask)))) # Concatenate along the z axis
        #
        # stack_image = self.transform(stack_image)
        #
        # # Unpack the transformed images
        # image, mask = stack_image[0, :, :].view(1, 256, 256), stack_image[1, :, :].view(1, 256, 256)
        #
        # ## Make sure we apply the same transform to image and mask!!!
        # ## Randomness means we could be flipping one and not the other
        # print(self.image_files[idx], self.mask_files[idx], self.isolated_mask_folders[idx])
        # return image.float(), mask.float(), self.isolated_mask_folders[idx]

def fetch_dataloader(types, data_dir, params, augment = False):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
    types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
    data_dir: (string) directory containing the dataset
    params: (Params) hyperparameters

    Returns:
    data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'dev', 'test']:
        if split in types:

            if split == 'test':
                image_path = os.path.join(data_dir, 'processed_test2', 'images')
                dl = DataLoader(NUCLEIDataset(image_path, None), batch_size=params.batch_size, shuffle=True,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda)

            if split == 'dev':
                image_path = os.path.join(data_dir, 'processed_{}'.format(split), 'images')
                mask_path = os.path.join(data_dir, 'processed_{}'.format(split), 'masks')
                dl = DataLoader(NUCLEIDataset(image_path, mask_path), batch_size=params.batch_size, shuffle=True,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda)

            if split == 'train':
                image_path = os.path.join(data_dir, 'processed_{}'.format(split), 'images')
                mask_path = os.path.join(data_dir, 'processed_{}'.format(split), 'masks')
                dl = DataLoader(NUCLEIDataset(image_path, mask_path, augment), batch_size=params.batch_size, shuffle=False,
                num_workers=params.num_workers,
                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
