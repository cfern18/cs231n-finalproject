import numpy as np
import cv2
import skimage
from scipy import ndimage

'''
Returns a list of (global_mask, list of individual cell masks) for each image.

N.b. This file has the postprocessing in it.
'''

def postprocess_output(output_batch, postprocess_filter='otsu'):
    batch_size, _, height, width = output_batch.shape
    outputs = []
    for idx in range(batch_size):
        outputs.append(convert_to_mask(output_batch[idx, 0, :, :], postprocess_filter))
    return outputs

def convert_to_mask(output, postprocess_filter='otsu'):
    '''
    Takes in the output of the net on a single image, which has shape (height, width) and returns
    a list of all the (height, width) masks for individual cells that it finds

    postprocess_filter (string; optional) argument tells the function which postprocessing method we
    should use to convert the output to a binary mask.
        Possible values are:
            'otsu' (default): Uses otsu thresholding only,
                http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html
            'hysteresis': Uses hysteresis double thresholding (with otsu threshold reference),
                http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.apply_hysteresis_threshold
            'watershed': Uses watershed algorithm with otsu (no hysteresis).
                http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
    '''
    thresh = skimage.filters.threshold_otsu(output)
    global_mask = np.where(output > thresh, 1, 0)

    if postprocess_filter == 'hysteresis':
        global_mask = run_hysteresis(global_mask, thresh, reduction_factor=5)
    elif postprocess_filter == 'watershed':
        global_mask = run_watershed(global_mask, dist_scale=1.0)

    # ---------------------------
    # mask_path = 'local_debugging/mask_images/'
    # np.save(mask_path + 'output', output)
    # np.save(mask_path + 'hysteresis', global_mask)
    # np.save(mask_path + 'otsu', global_mask_otsu)
    # np.save(mask_path + 'fill', global_mask_fill)
    # ---------------------------
    labels, nlabels = ndimage.label(global_mask)
    masks = []
    for label_num in range(1, nlabels+1):
        mask = np.where(labels == label_num, 1, 0)
        masks.append(mask)
    return global_mask, masks


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


def global_accuracy(prediction, split_predictions, ground_truth, split_ground_truth = None):
    '''
    Given a global prediction mask across a whole image and the ground truth mask of all the cells in the image,
    computes accuracy of the prediciton
    '''
    tp = np.sum(np.logical_and(prediction, ground_truth).astype(np.float))
    return tp/np.sum(ground_truth.astype(float))

def global_iou(prediction, split_predictions, ground_truth, split_ground_truth = None):
    i = np.sum(np.logical_and(prediction, ground_truth).astype(np.float))
    u = np.sum(np.logical_or(prediction, ground_truth).astype(np.float))
    return i / u

def global_precision(prediction, split_predictions, ground_truth, split_ground_truth = None):
    tp = np.sum(np.logical_and(prediction, ground_truth).astype(np.float))
    fn = np.sum(np.logical_and(prediction, ground_truth).astype(np.float))
    fp = np.sum(np.logical_and(prediction, (ground_truth == 0)).astype(np.float))
    return (float(tp) /(tp + fn + fp + 1e-8)) # Plus epsilon for stability

def img_precision(predicted_masks, true_masks, threshold = 0.5):
    '''
    For a single image, calculate its overall precision across a variety of thresholds

    Args:
        global_predicion:(np array) Unused, just so it fits the format of the other evaluation functions
        predicted_masks: (list) List of predicted masks for each object in the image.
        ground_truth:    (np array) Unused, so it fits format of other eval functions
        true_masks:      (list) List of ground truth masks for each obejct in the image.

    Returns:
        (float) Precision metric for the image.
    '''

    num_predicted_objs = len(predicted_masks)
    num_true_objs = len(true_masks)

    tp = 0
    fn = 0
    fp = 0

    # Find the preicted mask that maximizes IoU for the true mask. If the max IoU value is > threshold,
    # count as true positive (we correctly identified that object in the image).
    for true_mask in true_masks:
        found_match = False
        for predicted_mask in predicted_masks:
            curr_iou = global_iou(predicted_mask, None, true_mask)
            if curr_iou > threshold:
                found_match = True
                break

        if found_match:
            tp += 1
        else:
            fp += 1

    fn = num_true_objs - (num_predicted_objs - fp)
    if tp + fn + fp == 0:
        return 0.
    return float(tp) / (tp + fn + fp)

def avg_precision(global_prediciton, predicted_masks, ground_truth, true_masks): ##predicted_masks, true_masks):
    '''
    Returns the precision of a single image given the outputs of the detected objects and the ground truth masks,
    averaged over a range of threshold values (threshold values given explicitly in the definition of the Kaggle evaluation metric).

    Args:
        predicted_masks: (iterable) List of numpy arrays containing masks for each individual predicted image
        true_masks:  (iterable) List of numpy arrays containing ground truth masks for objects in the image.

    Returns:
        (float) Precision score for the single image, averaged across different threshold values.
    '''

    thresholds = np.arange(0.50, 0.96, 0.05) # Specified by Kaggle
    precisions = np.empty(len(thresholds))

    for i,t in enumerate(thresholds):
        precisions[i] = img_precision(predicted_masks, true_masks, t)

    return np.mean(precisions)

def kaggle_metric(avg_precisions):
    '''
    Final Kaggle evaluation metric.

    Args:
        avg_precisions: (iterable) Length = num_images. The average precision values (from avg_precision()) for each image.

    Returns:
        (float) The final Kaggle evaluation metric. Precision score averaged over all images.
    '''
    return np.mean(avg_precisions)

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': global_accuracy,
    'IoU' : global_iou,
    'Precision' : global_precision,
    'Kaggle Score' : avg_precision
    # could add more metrics such as accuracy for each token type
}
