"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
from skimage import io, transform
import torch
from torch.autograd import Variable
import utils
import model.unet_model_less_convs as net_less_convs
import model.unet_model_more_convs as net_more_convs
import model.unet_model as net
import model.data_loader as data_loader
import model.eval_functions as eval ##import postprocess_output, metrics

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='Data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--model_type', default='unet', help="Optional, specify the model architecture you would like to use. \
                    \"unet\" is the normal Unet Model; \"less_convs\" removes two convolutional layers at the output; \
                    \"more_convs\" adds extra convolutional layers at the output.")
parser.add_argument('--augment', default='augment', help="Optional, specify whether or not you would like to augment the \
                    training examples during training.")
parser.add_argument('--postprocess_filter', default='otsu', help='Optional. Specify which postprocessing filter to use.')

def load_masks_from_files(mask_folder):
    mask_files = [os.path.join(mask_folder, mask) for mask in os.listdir(mask_folder) if mask.endswith('.png')]
    masks = []
    for mask in mask_files:
        masks.append(transform.resize(io.imread(mask, as_grey = True), (256, 256)))
    return masks

def evaluate(model, loss_fn, dataloader, metrics, params, postprocess_filter='otsu', kaggle = True):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        postprocess_filter: (string) postprocessing filter to apply after otsu (hysteresis or watershed)
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    batch = 0
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
        output_batch = eval.postprocess_output(output_batch, postprocess_filter)
        # compute all metrics on an individual image:
        # print(data_batch[0, 0, :, :])
        # print(labels_batch[0, 0, :, :])
        for image in range(len(output_batch)):
            global_prediction, individual_predictions = output_batch[image]
            combined_mask = np.around(labels_batch[image, 0, :, :])
            masks = load_masks_from_files(folder_batch[image])
            if kaggle:
                summary_batch = {metric: metrics[metric](global_prediction.astype(bool), individual_predictions, combined_mask.astype(bool), masks)
                         for metric in metrics}
            else:
                summary_batch = {metric: metrics[metric](global_prediction.astype(bool), individual_predictions, combined_mask.astype(bool), masks)
                         for metric in metrics if metric != "Kaggle Score"}
            summ.append(summary_batch)
        # metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        # logging.info("- Eval metrics : " + metrics_string)
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Convert the augment argument from String to a Bool
    augment = False
    if args.augment.lower() == 'true':
        augment = True

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    # change back to 'dev'
    dataloaders = data_loader.fetch_dataloader(['dev'], args.data_dir, params, augment)
    test_dl = dataloaders['dev']

    logging.info("- done.")

    # Define the model
    model = None
    if args.model_type == 'less_convs':
        model = net_less_convs.UNet(params).cuda() if params.cuda else net_less_convs.UNet(params)
    elif args.model_type == 'more_convs':
        model = net_more_convs.UNet(params).cuda() if params.cuda else net_more_convs.UNet(params)
    else:
        assert args.model_type == 'unet', "{} is not a valid --model_type argument".format(args.model_type)
        model = net.UNet(params).cuda() if params.cuda else net.UNet(params)

    loss_fn = model.loss_fn
    metrics = eval.metrics

    # Choose the postprocessing filter, if specified
    # Warning: will do otsu unless string matches exactly for hysteresis or
    # watershed
    postprocess_filter = 'otsu'
    if args.postprocess_filter == 'hysteresis':
        postprocess_filter = 'hysteresis'
    elif args.postprocess_filter == 'watershed':
        postprocess_filter = 'watershed'

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params, postprocess_filter)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
