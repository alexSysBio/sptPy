# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:00:05 2025

@author: Alexandros Papagiannakis, HHMI at Stanford University
"""

from skimage import filters
from scipy import ndimage
from skimage.filters import threshold_local
import numpy as np

#--------- IMAGE FILTERS ---------#
# Customized image filters used to sharpen or smooth, as well as threshold images.
def log_adaptive_filter(image, parameters): 
    """
    This fucntion constructs an LoG filer (Laplace of Gaussian) as well as an adaptive filter to segment particles
    
    Parameters
    ---------
    image: numpy array - the image to be filtered and thresholded (usually this is the background subtracted image)
    particle_detection_parameters: list - the parameters for the LoG/adaptive filter
        [0] Smoothing factor for the Gaussian filter (https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian)
        [1] Laplace threshold (https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.laplace)
        [2] Hard threshold on the LoG image as a percentile of the brightest pixels
        [3] Gaussian smoothing factor before applying the adaptive threshold (https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian)
        [4] Block_size for the adaptive threshold (https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_local)
        [5] Offset for the adaptive threshold (https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_local)
        [6] Mask erosion applied after the adaptive threshold (it can be set to zero). Otherwise it is a positive integer (erosion rounds).

    Returns
    -------
    [0] The thresholded image after the LoG filter
    [1] The image after the adaptive threshold
    [2] The filter after multiplying the thresholded smoothed image with the adaptively thresolded image
    """
    
    # parameters = [4, 1000, 99.5] for testing
    image[image<0]=0
    # LoG filter with hard threshold
    # image_smoothed = filters.gaussian(image, parameters[0])
    image_smoothed = ndimage.gaussian_filter(image, sigma = parameters[0])
    image_laplace = filters.laplace(image_smoothed, parameters[1])
    image_pixel_intensities = image_laplace.ravel()
    sorted_pixel_intensities = np.sort(image_pixel_intensities)
    pixel_intensity_threshold = sorted_pixel_intensities[int((parameters[2]/100)*len(sorted_pixel_intensities))]
    log_image = image_laplace > pixel_intensity_threshold
    # plt.figure(figsize=(20,20))
    # plt.imshow(masked_image)
    # # plt.colorbar()
    # plt.show()
    # Adaptice threshold
    # image_smoothed_2 = filters.gaussian(image, parameters[3])
    image_smoothed_2 = ndimage.gaussian_filter(image, sigma=parameters[3])
    adaptive_threshold = threshold_local(image_smoothed_2, block_size=parameters[4], offset=parameters[5])
    # adaptive_threshold = threshold_local(image, block_size=parameters[4], offset=parameters[5])
    adaptively_masked_image = image_smoothed_2 > adaptive_threshold
    # plt.figure(figsize=(20,20))
    # plt.imshow(adaptively_masked_image)
    # # plt.colorbar()
    # plt.show()
    masked_image = adaptively_masked_image * log_image
        # erode the image if the erosion iterations are higher than 1
    if parameters[6] > 0:
        adaptively_masked_image = ndimage.morphology.binary_erosion(masked_image, iterations = parameters[6])
        final_image = adaptively_masked_image
    elif parameters[6] == 0:
        final_image = masked_image

    return log_image, adaptively_masked_image, final_image