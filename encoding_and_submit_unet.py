'''
Predict on test images and creat run-length encoding for submission.


Run-length encoding code adapted from public Kaggle kernels at:
https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277?scriptVersionId=2164855/code
and
https://www.kaggle.com/paulorzp/run-length-encode-and-decode
'''

import os
import argparse
import sys
import random
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd

import utils

import torch
import torch.optim as optim
from torch.autograd import Variable
import model.unet_model as net
import model.data_loader as data_loader
import model.eval_functions as eval

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='Data/', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional. Name of the file in --model_dir containing weights to reload before \
                    testing")  # 'best' or 'train'
parser.add_argument('--model_type', default='unet', help="Optional, specify the model architecture you would like to use. \
                    \"unet\" is the normal Unet Model; \"less convs\" removes two convolutional layers at the output; \
                    \"more convs\" adds extra convolutional layers at the output.")
parser.add_argument('--postprocess_filter', default='otsu', help='Optional. Specify which postprocessing filter to use.')
parser.add_argument('--save_file', default='submission', help='name of file you would like to save this submission to.')

# def rle_encode(img):
#     pixels = img.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return runs

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def predict_and_rle(model, data_dir, params, postprocess_filter, save_file, dataloader):
    # Set the model in evaluation mode
    model.eval()

    batch = 0
    rles = []

    dl_test = dataloader['test']

    image_files = os.path.join(data_dir, 'processed_test2', 'images')
    image_files = list(os.listdir(image_files))
    test_ids = [f[0:-4] for f in image_files if f.endswith('.png')]
    print('before loop')
    for test_batch, labels_batch, folder_batch in dl_test:
        print('inside loop')
        if params.cuda:
            test_batch_raw = test_batch_raw.cuda(async=True)

        # Pull the batch for testing
        test_batch = Variable(test_batch_raw)

        # Model outputs
        output_batch = model(test_batch)

        # Extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        
        # Post processing
        output_batch = eval.postprocess_output(output_batch, postprocess_filter)

        for i, id_ in enumerate(output_batch):
            rle = rle_encoding(output_batch[i][0])
            print(rle)
            rels.extend(rle)

    # Create submission DataFrame
    print('Creating submission file...')
    sub = pd.DataFrame()
    sub['ImageId'] = test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('{}.csv'.format(save_file), index=False)

if __name__ == '__main__':

    # Parse the call arguments
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Define the model
    model = None
    if args.model_type == 'less_convs':
        model = net_less_convs.UNet(params).cuda() if params.cuda else net_less_convs.UNet(params)
    elif args.model_type == 'more_convs':
        model = net_more_convs.UNet(params).cuda() if params.cuda else net_more_convs.UNet(params)
    else:
        assert args.model_type == 'unet', "{} is not a valid --model_type argument".format(args.model_type)
        model = net.UNet(params).cuda() if params.cuda else net.UNet(params)

    # Check if GPU is available
    params.cuda = torch.cuda.is_available()

    # Choose the postprocessing filter, if specified
    postprocess_filter = 'otsu'
    if args.postprocess_filter == 'hysteresis':
        postprocess_filter = 'hysteresis'
    elif args.postprocess_filter == 'watershed':
        postprocess_filter = 'watershed'

    # Reload weights from the saved file into the model
    print("Loading weights from .tar file...")
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)
    if params.cuda:
        model = model.cuda()

    # fetch dataloaders
    print('Fetching dataloaders...')
    dataloader = data_loader.fetch_dataloader(['test'], args.data_dir, params)

    # Make predictions on test set and save a .csv for submission.
    predict_and_rle(model, args.data_dir, params, postprocess_filter, args.save_file, dataloader)
