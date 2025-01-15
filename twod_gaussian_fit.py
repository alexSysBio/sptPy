# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:55:30 2025

@author: Alexandros Papagiannakis, HHMI at Stanford University
"""


import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from matplotlib import patches


def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """
    Returns a gaussian function with the given parameters
    
    Reference
    ---------
    Originally implemented by Andrew Giessel
    Code available on GitHub:
        https://gist.github.com/andrewgiessel/6122739
    
    Notes
    -----
    Modified by Alexandros Papagiannakis to correct the rotation 
    of the 2D Gaussian around the central pixel
    """
    width_x = float(width_x)
    width_y = float(width_y)
    
    rotation = np.deg2rad(rotation)
#    center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
#    center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)
    
    def rotgauss(x,y):
        x = x-center_x # modification
        y = y-center_y # modification
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        xp = xp + center_x
        yp = yp + center_y
        g = height*np.exp(
            -(((center_x-xp)/width_x)**2+
              ((center_y-yp)/width_y)**2)/2.)
        return g
    return rotgauss


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    
    def moments(data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total

        col = data[:, int(y)]

        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())

        row = data[int(x), :]

        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y, 0.0
    
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = leastsq(errorfunction, params)
    return p


def show_gaussian_particle_fit(cropped_particle_image, fitted_subpx, fitted_px,  param):
    """
    This function plots the particle fluorescence 2D gaussian fit and the particle center estimation.
    
    Parameters
    ----------
    cropped_particle_image: 2D numpy array - the cropped particle image (within the bounding box)
    fitted_subpx: 2D numpy array - the blurred particle image
    fitted_px: 2D numpy array - the raw particle image
    param: height, y, x, width_y, width_x, rotation - parameters of the 2D Gaussian fit
    """
    (height, y, x, width_y, width_x, rotation) = param
    # PLOT 1
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(fitted_subpx)
    ax = plt.gca()
    plt.text(0.95, 0.05, """
             amp: %.1f
             x : %.1f
             y : %.1f
             width_x : %.1f
             width_y : %.1f""" %(height, x, y, width_x, width_y),
             fontsize=16, horizontalalignment='right',
             verticalalignment='bottom', transform=ax.transAxes)
    plt.show()
    # PLOT 2
    # Fit a citcle at the center of each particle, which corresponds to the peak of the Gaussian
    circle = patches.Circle((x, y), radius=0.1, color='red', fill=False, linewidth=3)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(cropped_particle_image)
    plt.contour(fitted_px, cmap=plt.cm.Greys)
    ax.add_patch(circle)
    print('2D gaussian params:',param)
    plt.show()  
    
    
def estimate_particle_center(bkg_cor_image, particle_center, box_size, gaussian_fit_show=False):
    """
    This function fits the 2D Gaussian to estimate the particle center.
    
    Parameters
    ----------
    bkg_cor_image: 2D numpy array - the background corrected particle fluorescence image
    particle_center: tuple of floats (x,y) corresponding to the center of mass of the particle label
    box_size: odd integer - the size of the box that is used to fit the 2D gaussian for the particle center estimation (5 or 11 suggested)
    gaussian_fit_show: bool - True to show the 2D Gaussian fit
                              False otherwise
    Returns
    -------
    gaussian_particle_center: tupel of (x,y) floats: the estimate particle center (subpixel)
    brightest_raw_pixels: numpy array - the values of the 60% brightest raw pixels
    brightest_fitted_pixels: numpy array - the values of the 60% brightest smoothed pixels
        returns 'none' if one of the following exceptions is met.
    
    Exception
    ---------
    If the approximated particle center falls outside the bounding box, 
            the spot is not diffratcion limited and the particle is aborted.
                    'none' is returned
    If the particle bounding box falls outside the sensor dimensions, 
        'none' is returned
    If the particle eccentricity is below 0.1 or above 19,
        This spot probably corresponds to multiple clustered particles.
            'none' is returned
    """
    half_side = int((box_size-1)/2)
    cropped_particle_image = bkg_cor_image[(int(particle_center[1])-half_side):(int(particle_center[1])+(half_side+1)), (int(particle_center[0])-half_side):(int(particle_center[0])+(half_side+1))]
    if (cropped_particle_image.shape[0] > 0 and cropped_particle_image.shape[1] > 0):
        # create a meshgrid with 0.1 pixel resolution
        Xin, Yin = np.mgrid[0:cropped_particle_image.shape[1]:0.1,0:cropped_particle_image.shape[0]:0.1]
        # create a meshgrid with single pixel resolution
        Xin2, Yin2 = np.mgrid[0:cropped_particle_image.shape[1]:1,0:cropped_particle_image.shape[0]:1]
        # fit a 2D gaussian with rotation to the cropped image
        try:
            param = fitgaussian(cropped_particle_image)
            (height, y, x, width_y, width_x, rotation) = param
            # set an eccentricity threshold and the center coordinates fall within the box
            if (width_y/width_x > 0.1 and width_y/width_x < 10 and x<box_size and y<box_size and x>0 and y>0):
                gaussian_fit = gaussian(*param)
                # fit a guassian to a lattice of Xin by Yin pixels
                fitted_subpx = gaussian_fit(Xin, Yin)
                fitted_px = gaussian_fit(Xin2, Yin2)
                # Get the 60% brightest smoothed pixels
                brightest_fitted_pixels = fitted_px[fitted_px>np.percentile(fitted_px,40)]
                # Get the 60% brightest raw pixels
                brightest_raw_pixels = cropped_particle_image[cropped_particle_image>np.percentile(cropped_particle_image, 40)]
                # Correct the gaussian coordinates to the original image dimensions (from the cropped dimensions)
                gaussian_x_coord = x + int(particle_center[0])-half_side
                gaussian_y_coord = y + int(particle_center[1])-half_side
                gaussian_particle_center = (gaussian_x_coord, gaussian_y_coord)
                if gaussian_fit_show == True:
                    show_gaussian_particle_fit(cropped_particle_image, fitted_subpx, fitted_px,  param)
        
                return gaussian_particle_center, brightest_raw_pixels, brightest_fitted_pixels, param
            else:
                print('This particle did not pass the eccentricity and position conditions. Particle position aborted:', particle_center)
                return 'none'
        except IndexError:
            print('The approximate particle center is outside the square bounds. Particle position aborted:', particle_center)
            return 'none'
    else:
        print('This particle position spec ranges out of bounds. Particle position aborted:', particle_center)
        return 'none'
    
    
def get_particle_fluorescence(metric, operation, brightest_raw_pixels, brightest_fitted_pixels, gaussian_vol):
    """
    This function is used to estimate the particle fluorescence at each position
    
    Paramters
    ---------
    metric: string, the column of the pandas dataframe which is used as a proxy for particle size. 
            Choose 'gaussian volume', 'raw pixels' or 'smoothed pixels'
    operation: function, the mathematical operation applied to the bin_column as a particle size proxy. 
                Choose 'mean', 'median', 'sum' or 'max'
    brightest_raw_pixels: numpy array - the 60% brightest raw pixels 
    brightest_fitted_pixels: numpy array - the 60% brightest smoothed pixels 
    gaussian_vol: the volume of the fitted 2D Gaussian to the particle pixels
    
    Returns
    -------
    particle_vol: float - the particle fluorescence value corresponding to the chosen metric and operation
    
    Raises
    ------
    ValueError if an invalid metric or operation are included as inputs
    """
    # estimate the particle volume statistic based on a given metric and operation
    if metric == 'gaussian volume':
        particle_vol = gaussian_vol.copy()
        operation = 'none'
    elif metric == 'raw pixels':
        if operation == 'mean':
            particle_vol = np.mean(brightest_raw_pixels)
        elif operation == 'median':
            particle_vol = np.median(brightest_raw_pixels)
        elif operation == 'sum':
            particle_vol = np.sum(brightest_raw_pixels)
        elif operation == 'max':
            particle_vol = np.max(brightest_raw_pixels)
        else:
            raise ValueError("Choose 'mean', 'median', 'sum' or 'max' as the operation.")
    elif metric == 'smoothed pixels':
        if operation == 'mean':
            particle_vol = np.mean(brightest_fitted_pixels)
        elif operation == 'median':
            particle_vol = np.median(brightest_fitted_pixels)
        elif operation == 'sum':
            particle_vol = np.sum(brightest_fitted_pixels)
        elif operation == 'max':
            particle_vol = np.max(brightest_fitted_pixels)
        else:
            print("wrong operation for particle fluorescence estimation")
            raise ValueError("Choose 'mean', 'median', 'sum' or 'max' as the operation.")
    else:
        print("wrong metric for particle fluorescence estimation")
        raise ValueError("Choose 'gaussian volume', 'raw pixels', or 'smoothed pixels' as a metric")
    
    return particle_vol