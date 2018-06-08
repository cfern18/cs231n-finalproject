"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys
import numpy as np

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='Data/', help="Directory containing the dataset")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--model_type', default='unet', help="Optional, specify the model architecture you would like to use. \
                    \"unet\" is the normal Unet Model; \"less_convs\" removes two convolutional layers at the output; \
                    \"more_convs\" adds extra convolutional layers at the output.")
parser.add_argument('--augment', default='False', help="Optional, specify whether or not you would like to augment the \
                    training examples during training.")


def launch_training_job(parent_dir, data_dir, model, augment, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        model: (string) type of model to use
        augment: (string) whether or not to augment the images in the training set
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir} --model_type {model_type} --augment {augment}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir, model_type=model, augment=augment)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.in_channels = 1
    params.out_channels = 1
    params.batch_size = 16

    augment = args.augment
    assert augment.lower() in ['true', 'false'], "{} is not a valid augment flag".format(augment)

    model = args.model_type
    assert model.lower() in ['more_convs', 'less_convs', 'unet'], "{} is not a valid model type".format(model)

    # Perform hypersearch over learning rate
    # Sample the log space to randomize learning rate choices
    #learning_rates = [1e-4, 1e-3, 1e-2]
    NUM_RATES = 4
    np.random.seed(10) # Set seed to predict output/use same learning rates for each model.
    random_rates = np.random.uniform(-7, -4, NUM_RATES)
    learning_rates = [10**x for x in random_rates]
    # Comment out the next line after use:
    learning_rates = [learning_rates[0], learning_rates[3]]

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate
        # Hypersearch over number of epochs
        num_epochs = [20] # Set to maximum of 20 for now because we'll watch its progress as it goes
        for num_epoch in num_epochs:
            # Modify the relevant parameter in params
            params.num_epochs = num_epoch

            job_name = "num_epochs_{epoch}_learning_rate_{lr}".format(epoch=num_epoch,lr=learning_rate)
            launch_training_job(args.parent_dir, args.data_dir, model, augment, job_name, params)
    
