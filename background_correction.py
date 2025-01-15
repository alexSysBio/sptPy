# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:46:16 2024

@author: Alexandros Papagiannakis, HHMI at Stanford University
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy import ndimage


def cell_free_bkg_estimation(masked_signal_image, step):
    """
    This function scans the image using squared regions of specified size (step) 
    and applies the average cell-free background fluorescence per region.
    This function is used in the self.back_sub() function.
    
    Parameters
    ----------
    masked_signal_image: 2D numpy array - the signal image were the cell pixels are annotated as 0 
                         and the non-cell pixels maintain their original grayscale values
    step: integer (should be a divisor or the square image dimensions) - the dimensions of the squared region where 
          the cell-free background fluorescence is averaged
            example: for an 2048x2048 image, 128 is a divisor and can be used as the size of the edge of the square 

    Returns
    -------
    A 2D numpy array where the cell-free average background is stored for each square region with specified step-size
    """
    
    sensor = masked_signal_image.shape
    
    zero_image = np.zeros(sensor) # initiated an empty image to store the average cell-free background
    
    for y in range(0, sensor[0], step):
        for x in range(0, sensor[1], step):
            # cropped_image = img_bkg_sig[y:(y+step), x:(x+step)]
            cropped_mask = masked_signal_image[y:(y+step), x:(x+step)]
#                mean_bkg = np.mean(cropped_mask[np.nonzero(cropped_mask)].ravel()) # get the mean of the non-zero pixels
#                mean_bkg = scipy.stats.mode(cropped_mask[cropped_mask!=0].ravel())[0][0] # get the mode of the non-zero pixels
            mean_bkg = np.nanmedian(cropped_mask[np.nonzero(cropped_mask)].ravel()) # get the mean of the non-zero pixels
            zero_image[y:(y+step), x:(x+step)] = mean_bkg # apply this mean fluorescence to the original empty image
                   
    return zero_image

def get_inverted_mask(phase_image):
    """returns a masked image using the inverted phase contrast image and the Otsu threshold

    Args:
        phase_image (2D numpy array): phase contrast image

    Returns:
        phase_mask: binary 2D numpy array
    """
    # invert the image and apply an otsu threshold to separate the dimmest 
    # (or inversely brightest pixels) which correspond to the cells
    inverted_phase_image = 1/phase_image
    inverted_threshold = threshold_otsu(inverted_phase_image.ravel())
    phase_mask = inverted_phase_image > inverted_threshold
    
    return phase_mask

def back_sub(signal_image, phase_mask, dilation=15, estimation_step=128, smoothing_sigma=60, show=False):
    """
    Subtracts an n_order second degree polynomial fitted to the non-cell pixels.
    The 2D polynomial surface is fitted to the non-cell pixels only.
        The order of the polynomial depends on whether there is uneven illumination or not
    The non-cell pixels are masked as thos below the otsu threshold estimated on the basis of the inverted phase image.
    
    Parameters
    ----------
    signal_image: numpy.array - the image to be corrected
    phase_mask: numpy.array - the inverted mask return by the get_inverted_mask_function
    dilation: non-negative integer - the number of dilation rounds for the cell mask
    estimation_step: positive_integer - the size of the square edge used for average background estimation. Must divide the dimensions of the image perfectly
    smoothing_sigma: non-negative integer - the smoothing factor of the cell free background
    show: binary - True if the user wants to visualize the 2D surface fit
    
    Returns
    -------
    [0] The background corrected image (2D numpy array) also corrected for uneven excitation
    [1] The background corrected image (2D numpy array) 
    [2] The background pixel intensities
    """
    print('Subtracting background...')
    
    # dilate the masked phase images
    threshold_masks_dil = ndimage.binary_dilation(phase_mask, iterations=dilation)
    threshold_masks_dil = np.array(threshold_masks_dil)
    # mask the signal image, excluding the dilated cell pixels
    masked_signal_image = signal_image * ~threshold_masks_dil
    if show == True:
        # plt.figure(figsize=(10,10))
        plt.imshow(threshold_masks_dil)
        plt.show()
    # The dimensions of the averaging square
    step = estimation_step
    img_bkg_sig = cell_free_bkg_estimation(masked_signal_image, step)
    if show == True:
        # plt.figure(figsize=(20,20))
        plt.imshow(img_bkg_sig, cmap='Blues')
        plt.clim(np.mean(img_bkg_sig.ravel())-5*np.std(img_bkg_sig.ravel()),np.mean(img_bkg_sig.ravel())+2.5*np.std(img_bkg_sig.ravel()))
        plt.colorbar()
        plt.show()
    # Smooth the reconstructed background image, with the filled cell pixels.
    # img_bkg_sig = img_bkg_sig.astype(np.int16)
    img_bkg_sig = ndimage.gaussian_filter(img_bkg_sig, sigma=smoothing_sigma)
    norm_img_bkg_sig = img_bkg_sig/np.max(img_bkg_sig.ravel())
    if show == True:
        # plt.figure(figsize=(20,20))
        plt.imshow(img_bkg_sig, cmap='Blues')
        # plt.clim(0,25*np.std(bkg_cor.ravel()))
        plt.colorbar()
        plt.show()
    # subtract the reconstructed background from the original signal image
    bkg_cor = (signal_image - img_bkg_sig)/norm_img_bkg_sig
    bkg_cor_2 = signal_image - img_bkg_sig
    # use this line if you want to convert negative pixels to zero
    # bkg_cor[bkg_cor<0]=0
    if show == True:
        # plt.figure(figsize=(20,20))
        plt.imshow(bkg_cor, cmap='Blues')
        plt.clim(0,25*np.std(bkg_cor.ravel()))
        plt.colorbar()
        plt.show()
        # plt.figure(figsize=(20,20))
        plt.imshow(img_bkg_sig*threshold_masks_dil, cmap='Blues')
        plt.clim(np.mean(img_bkg_sig.ravel())-5*np.std(img_bkg_sig.ravel()),np.mean(img_bkg_sig.ravel())+2.5*np.std(img_bkg_sig.ravel()))
        plt.colorbar()
        plt.show()
    
    return bkg_cor, bkg_cor_2, img_bkg_sig