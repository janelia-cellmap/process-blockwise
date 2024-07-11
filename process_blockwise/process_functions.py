#!/usr/bin/env python3

from skimage.filters import gaussian
from skimage.measure import label
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import numpy as np

class PostProcessFunctionNotFoundError(Exception):
    pass


def gaussian_blur(input_data, sigma):
    return gaussian(input_data, sigma)

def threshold(input_data, threshold):
    dtype = input_data.dtype
    return (input_data > threshold).astype(dtype)

def instances(input_data,  threshold, gaussian_kernel=None):
    if gaussian_kernel is not None:
        input_data = gaussian(input_data, sigma=gaussian_kernel)

    binary_data = input_data > threshold

    markers, _ = ndi.label(binary_data)
    ws_labels = watershed(-input_data, markers, mask=binary_data)
    result_data = label(ws_labels)

    return result_data.astype(np.uint64)


process_functions = {
    'gaussian': gaussian_blur,
    'threshold': threshold,
    'instances': instances
    # 'merge_blocks': merge_blocks,
    # 'mask_filter': mask_filter,
    # 'size_filter': size_filter,
    # 'relabel': relabel
}


