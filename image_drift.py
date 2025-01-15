# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:52:49 2025

@author: Alexandros Papagiannakis, HHMI at Stanford University, 2025
"""
from skimage.registration import phase_cross_correlation

# GET THE IMAGE DRIFT - USED TO ESTIMATE MOMENTS DURING FAST TIME LAPSE / CROSS CORRELATION ANALYSIS#
def estimate_phase_image_drift(image, offset_image, precision=100):
    """
    Sub-pixel baded determination of offset between two images.
    
    Input:
        image: the first image - numpy array
        offset_image: the second image - numpy array
        precision: 1 for pixel based, 10 for pixel .1 pixel precision, 100 for .01 pixels precesion and so forth
    Returns:
        [0] shift: float - the pixel or subpixel based shift in the two dimensions (y,x) (offset_image_y-image_y, offset_image_x-image_x)
        [1] error: float - the error in the drift (should be lower than the estimated drift)
        [2] diffphase: float - the phase difference in the DFT
    
    Reference:
        Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 
        “Efficient subpixel image registration algorithms,” Optics Letters 33, 156-158 (2008). 
        DOI:10.1364/OL.33.000156
    
    """
#        shift, error, diffphase = register_translation(image, offset_image, precision) # deprecated function
    shift, error, diffphase = phase_cross_correlation(image, offset_image, upsample_factor=precision)
    
    return shift, error, diffphase
