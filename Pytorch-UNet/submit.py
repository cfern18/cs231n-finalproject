# used to predict all test images and encode results in a csv file

import os
from PIL import Image
from predict import *
from utils import encode
from unet import UNet

import torch
import json
import logging
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='Data', help="Directory containing the dataset to evaluate on.")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--model_type', default='unet', help="Optional, specify the model architecture you would like to use. \
                    \"unet\" is the normal Unet Model; \"less_convs\" removes two convolutional layers at the output; \
                    \"more_convs\" adds extra convolutional layers at the output.")
parser.add_argument('--postprocess_filter', default='otsu', help='Optional. Specify which postprocessing filter to use.')

def submit(net, gpu=False):
    dir = '../Data/stage2_test/'

    N = len(list(os.listdir(dir)))
    with open('SUBMISSION_HYST_STAGE_2.csv', 'a') as f:

        f.write('img,rle_mask\n')
        for index, i in enumerate(os.listdir(dir)):
            print('{}/{}'.format(index, N))

            img = Image.open(dir + index)

            mask = predict_img(net, img, gpu)
            enc = rle_encode(mask)
            f.write('{},{}\n'.format(i, ' '.join(map(str, enc))))

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


if __name__ == '__main__':
    #net = UNet(3, 1).cuda()

    model_path = '../experiments/post_processing/hysteresis/best.pth.tar'
    #net.load_state_dict(torch.load(model_path))
    model = net.UNet(params).cuda() if params.cuda else net.UNet(params)
    submit(net, True)
