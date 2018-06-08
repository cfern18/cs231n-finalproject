#STARTER RLE CODE FROM: https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines
"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
from skimage import io, transform
import torch
from torch.autograd import Variable
import utils
import model.unet_model as net
import model.data_loader as data_loader
import model.eval_functions as eval ##import postprocess_output, metrics
from eval_functions import convert_to_mask

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='Data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def load_masks_from_files(mask_folder):
    mask_files = [os.path.join(mask_folder, mask) for mask in os.listdir(mask_folder) if mask.endswith('.png')]
    masks = []
    for mask in mask_files:
        masks.append(transform.resize(io.imread(mask, as_grey = True), (256, 256)))
    return masks

def generate_RLE(model, params, dataloader):
    # set model to evaluation mode
    model.eval()
    
    # compute metrics over the dataset
    for data_batch, labels_batch, folder_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        #turns it into a list of outputs
        output_batch = eval.postprocess_output(output_batch)

        for image in range(len(output_batch)):
        	print(image)
            global_prediction, individual_predictions = output_batch[image]
            combined_mask = np.around(labels_batch[image, 0, :, :])
            masks = load_masks_from_files(folder_batch[image])


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU if available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)


    # Create the input data pipeline
    print("Creating the dataset...")

    # fetch dataloaders
    datadict = fetch

    print("- done.")

    # Define the model
    model = net.UNet(params).cuda() if params.cuda else net.UNet(params)

    print("Starting RLE Generation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = generate_RLE(model, params)
    print("- done.")
