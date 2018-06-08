"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
#import model.net as net
import model.unet_model_less_convs as net_less_convs
import model.unet_model_more_convs as net_more_convs
import model.unet_model as net
import model.data_loader as data_loader
from evaluate import evaluate
import model.eval_functions as eval
from skimage.filters import threshold_otsu
from skimage.filters import apply_hysteresis_threshold
import skimage
from scipy import ndimage

import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='Data/', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--model_type', default='unet', help="Optional, specify the model architecture you would like to use. \
                    \"unet\" is the normal Unet Model; \"less convs\" removes two convolutional layers at the output; \
                    \"more convs\" adds extra convolutional layers at the output.")
parser.add_argument('--augment', default='False', help="Optional, specify whether or not you would like to augment the \
                    training examples during training.")
parser.add_argument('--postprocess_filter', default='otsu', help='Optional. Specify which postprocessing filter to use.')

def train(model, optimizer, loss_fn, dataloader, metrics, params, postprocess_filter):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # move to GPU for training
    # model = model.cuda()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch, folder_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            # convert to torch Variables
            # print('train_batch before ', Variable(train_batch).type())
            # print('labels_batch before ', Variable(labels_batch).type())
            train_batch, labels_batch = Variable(train_batch).float(), Variable(labels_batch).float()
            # print('-------')
            # print('after ', train_batch.type())
            # print('after ', labels_batch.type())
            # print('-------')
            # compute model output and loss
            output_batch = model(train_batch)
            #print('output batch type ', output_batch.type())
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                for sample in range(output_batch.shape[0]):
                    thresh = threshold_otsu(output_batch[sample, 0, :, :])
                    output_batch[sample, 0, :, :] = np.where(output_batch[sample, 0, :, :] > thresh, 1, 0)

                    if postprocess_filter == 'hysteresis':
                        output_batch[sample, 0, :, :] = run_hysteresis(output_batch[sample, 0, :, :], thresh, reduction_factor=5)
                    elif postprocess_filter == 'watershed':
                        output_batch[sample, 0, :, :] = run_watershed(output_batch[sample, 0, :, :], dist_scale=1.0)

                summary_batch = {metric:metrics[metric](output_batch, None, np.around(labels_batch))
                                 for metric in metrics if metric != "Kaggle Score"}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

def run_hysteresis(otsu_prediction, thresh, reduction_factor=5):
    '''
    Uses hysteresis thresholding to smooth over image
    '''
    delta = thresh / reduction_factor
    low_thresh  = thresh - delta
    high_thresh = thresh

    # apply_hysteresis_threshold() returns an array of bools, so convert to ints
    hyst_mask = (skimage.filters.apply_hysteresis_threshold(otsu_prediction, low_thresh, high_thresh)).astype(int)
    return hyst_mask

def run_watershed(otsu_prediction, dist_scale=1.0):
    '''
    Adapted from
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html

    Uses watershed algorithm to postprocess image. Assumes the image is the output of an
    otsu thresholding.

    Returns the global mask with watershed applied

    dist_scale is an optional float. The second argument to cv2.threshold(dist_transform, ...)
    has a large impact on the performance of watershed. See comment below. Possible hyperparameter
    search later (similar to reduction_factor for hysteresis).
    '''
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(otsu_prediction.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # ****the second argument to the threshold here could be a learnable hyperparameter****
    ret, sure_fg = cv2.threshold(dist_transform, dist_scale*dist_transform.mean(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    otsu_3chan = skimage.color.gray2rgb(otsu_prediction)
    otsu_3chan = skimage.img_as_ubyte(otsu_3chan)

    # Watershed call
    markers = cv2.watershed(otsu_3chan, markers)
    markers = np.where(markers == 1, 0, 1)

    return markers

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, postprocess_filter, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_kaggle = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params, postprocess_filter)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, postprocess_filter, kaggle = True)
        
        val_kaggle = val_metrics['IoU']
        is_best = val_kaggle >= best_val_kaggle

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_kaggle = val_kaggle

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    #val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, kaggle = True)

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Convert the augment argument from String to a Bool
    augment = False
    if args.augment.lower() == 'true':
        augment = True

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'dev'], args.data_dir, params, augment)
    train_dl = dataloaders['train']
    dev_dl = dataloaders['dev']

    logging.info("- done.")

    # Define the model and optimizer
    model = None
    if args.model_type == 'less_convs':
        model = net_less_convs.UNet(params).cuda() if params.cuda else net_less_convs.UNet(params)
    elif args.model_type == 'more_convs':
        model = net_more_convs.UNet(params).cuda() if params.cuda else net_more_convs.UNet(params)
    else:
        assert args.model_type == 'unet', "{} is not a valid --model_type argument".format(args.model_type)
        model = net.UNet(params).cuda() if params.cuda else net.UNet(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = model.get_loss_fn()
    metrics = eval.metrics

    # Choose the postprocessing filter, if specified
    # Warning: will do otsu unless string matches exactly for hysteresis or
    # watershed
    postprocess_filter = 'otsu'
    if args.postprocess_filter == 'hysteresis':
        postprocess_filter = 'hysteresis'
    elif args.postprocess_filter == 'watershed':
        postprocess_filter = 'watershed'

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params, postprocess_filter, args.model_dir, args.restore_file)
