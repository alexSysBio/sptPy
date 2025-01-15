# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:52:29 2025

@author: alexpapa
"""

import numpy as np
import custom_image_filters as cimf
from skimage.measure import label, regionprops
import pandas as pd
import matplotlib.pyplot as plt
import background_correction as bkgf
import matplotlib

    
#---------- SEGMENTATION AND TRACKING ALGORITHMS -----------#
# These functions are used to segment and track the particles

def particle_segmentation(back_sub_image, log_adaptive_parameters, min_spot_size, max_spot_size, min_spot_aspect_ratio, post_processing_threshold):
    """
    This function is used to segment particle in the fluorescence image.
    It first applies an log_adaptive filer (see line 883) to filter and segment the image.
    If the detected spots are larger than the max_spot_size parameter, or have a very low aspect ratio (minor axis / major axis) then the algorithm zooms into
    the spot and applies new segmentation parameters. This happens since very large or elongated spots are probably attributed to many particles recognized as one.
    By applying stricter parameters it is possible to separate the clustered particles. 
    
    Parameters
    ----------
    back_sub_image: 2D numpy ndarray: the particle fluorescence image to be segmented (background corrected)
    log_adaptive_parameters: list - the parameters for segmentation (log_adaptive filter parameters)
    min_spot_size: integer - the expected min size of the particle in pixels
    max_spot_size: integer - the expected max size of the particle in pixels
    min_spot_aspect_ratio: float - the expected minimum aspect ration (minor/major axis) of thge particle
    post_processing_threshold: float or integer - The hard threshold for segmented the clustered particles as a percentage of brightest pixels

    Returns
    -------
    A pandas dataframe:
        columns:
        'centroid' microtubule_centroid -> the centroid coordinates of the particle (x,y tuple)
        'minor_axis' microtubule_minor_axis -> the minor axis of the segmemted particle spot in px
        'major_axis' microtubule_major_axis -> the major axis of the segmented particle spot in px
        'aspect_ratio' particle_aspect_ratio_list -> the minor/major axis ratio
        'area' microtubule_area -> the area of the segmented particle spot in px
        'experiment' self.experiment -> the experiment (inherited from the class)
    """
    def post_processing(back_sub_image, particle_centroid_yx, particle_major_axis, particle_minor_axis, post_processing_threshold, particle_lists):
        """
        This function is used to reprocess the particle ROI in otder to separate clustered particles.
        It zooms into the segmented spot and applies a new LoG filter to separate multiple particles that
        were segmented as one (very proximal to each other).
        This function is applied if the segmented spot is larger than the maximum spot area selected by the user
        or if the aspect ratio (minor/major axis) of the spot is lower than expected.
        
        If the pos-processed regoins of interest result in smaller or rounder segmented spots within the specified ranges of 
        particle areas and aspect ratios, then the particle coordinates and dimenions lists are updated including these
        post-processed particles.
        
        Parameters
        ----------
        back_sub_image: a 2D numpy ndarray of the backgrund subtracted image
        particle_centroid_yx: tuple - the (y,x) coordinates of the segmented spot
        particle_major_axis: float - the major axis of the segmented spot
        particle_minor_axis: float - the minor axis of the segmented spot
        post_processing_threshold: float or integer - The hard threshold for segmented the clustered particles as a percentage of brightest pixels
        particle_lists: a list containing the particle_centroid, particle_minor_axis, particle_major_axis, particle_aspect_ratio and particle_area lists.
            [particle_centroid_list, particle_minor_axis_list, particle_major_axis_list, particle_aspect_ratio_list, particle_area_list]
        Returns
        -------
        The updated particle_centroid [0], particle_minor_axis [1], particle_major_axis [2], particle_aspect_ratio [3] and particle_area lists [4].
        """
        particle_centroid_list, particle_minor_axis_list, particle_major_axis_list, particle_aspect_ratio_list, particle_area_list = particle_lists
        
        sensor = back_sub_image.shape
        
        if type(post_processing_threshold) == int or type(post_processing_threshold) == float:
            if int(particle_centroid_yx[0]-particle_major_axis/2) >= 0 and int(particle_centroid_yx[0]+particle_major_axis/2) < sensor[1] and int(particle_centroid_yx[1]-particle_major_axis/2) >= 0 and int(particle_centroid_yx[1]+particle_major_axis/2) < sensor[0]:
                print(particle_centroid_yx[::-1], particle_area, particle_major_axis, particle_minor_axis,'post processing...')
                # crop the image around the spot position - zooming into the spot of interest and apply a stricter log filter
                cropped_image = back_sub_image[int(particle_centroid_yx[0]-particle_major_axis/2):int(particle_centroid_yx[0]+particle_major_axis/2), int(particle_centroid_yx[1]-particle_major_axis/2):int(particle_centroid_yx[1]+particle_major_axis/2)]
                fluorescence_threshold = np.sort(cropped_image.ravel())[-int(len(cropped_image.ravel())*((100-post_processing_threshold)/100))]
                
                cropped_filtered_image = cropped_image > fluorescence_threshold
                # emumerate the segmented spots in the cropped image
                sub_particle_labels = label(cropped_filtered_image)
                # get the features of the segmented spots
                for sub_particle_region in regionprops(sub_particle_labels):
                    sub_particle_centroid_yx = sub_particle_region.centroid
                    sub_particle_centroid_yx_corrected = (sub_particle_centroid_yx[0] + int(particle_centroid_yx[0]-particle_major_axis/2), sub_particle_centroid_yx[1] + int(particle_centroid_yx[1]-particle_major_axis/2))
                    sub_particle_minor_axis = sub_particle_region.minor_axis_length
                    sub_particle_major_axis = sub_particle_region.major_axis_length
                    sub_particle_area = sub_particle_region.area
                    
                    if sub_particle_area >= min_spot_size  and sub_particle_minor_axis > 0 and sub_particle_major_axis > 0:
                        sub_particle_aspect_ratio = sub_particle_minor_axis / sub_particle_major_axis

                        particle_centroid_list.append(sub_particle_centroid_yx_corrected[::-1])
                        particle_minor_axis_list.append(sub_particle_minor_axis)
                        particle_major_axis_list.append(sub_particle_major_axis)
                        particle_aspect_ratio_list.append(sub_particle_aspect_ratio)
                        particle_area_list.append(sub_particle_area)
        # Only if the post-processing threshold is a % float or integer of brightest pixels it is then used to split clustered spots within the region
        elif post_processing_threshold == None:
            print(particle_centroid_yx[::-1], particle_area, particle_major_axis, particle_minor_axis, 'the post-processing parameters were not defined. post-processing aborted...')
        
        return particle_centroid_list, particle_minor_axis_list, particle_major_axis_list, particle_aspect_ratio_list, particle_area_list

    print('Filtering image...')
    filtered_image = cimf.log_adaptive_filter(back_sub_image, log_adaptive_parameters)[2]
    # ennumerate the separated masks in the image using the labels functioon from the skimage.measure library
    particle_labels = label(filtered_image)
    # These lists will store the particle data if the particles satisfy all the conditions
    particle_centroid_list = []
    particle_minor_axis_list = []
    particle_major_axis_list = []
    particle_aspect_ratio_list = []
    particle_area_list = []

    for particle_label in regionprops(particle_labels):
        particle_centroid_yx = particle_label.centroid
        particle_minor_axis = particle_label.minor_axis_length
        particle_major_axis = particle_label.major_axis_length
        particle_area = particle_label.area
        # This condition is important to avoid numerical erros (very small particles of 1 or 2 pixels in size) and avoids division by zero
        if (particle_area > 0 and particle_minor_axis > 0 and particle_major_axis > 0):
            particle_aspect_ratio = particle_minor_axis/particle_major_axis
            if particle_area >= min_spot_size and particle_area <= max_spot_size and particle_aspect_ratio >= min_spot_aspect_ratio:
                # reversing the centroid ro correspond to the xy and not the yx coordinates
                particle_centroid_list.append(particle_centroid_yx[::-1])
                particle_minor_axis_list.append(particle_minor_axis)
                particle_major_axis_list.append(particle_major_axis)
                particle_aspect_ratio_list.append(particle_aspect_ratio)
                particle_area_list.append(particle_area)
            elif particle_area >= min_spot_size and particle_area <= max_spot_size and particle_aspect_ratio < min_spot_aspect_ratio:
                print(particle_area, particle_minor_axis, particle_major_axis)
                particle_lists = [particle_centroid_list, particle_minor_axis_list, particle_major_axis_list, particle_aspect_ratio_list, particle_area_list]
                particle_centroid_list, particle_minor_axis_list, particle_major_axis_list, particle_aspect_ratio_list, particle_area_list = post_processing(back_sub_image, particle_centroid_yx, particle_major_axis, particle_minor_axis, post_processing_threshold, particle_lists)
            elif (particle_area > max_spot_size):
                print(particle_area, particle_minor_axis, particle_major_axis)
                particle_lists = [particle_centroid_list, particle_minor_axis_list, particle_major_axis_list, particle_aspect_ratio_list, particle_area_list]
                particle_centroid_list, particle_minor_axis_list, particle_major_axis_list, particle_aspect_ratio_list, particle_area_list = post_processing(back_sub_image, particle_centroid_yx, particle_major_axis, particle_minor_axis, post_processing_threshold, particle_lists)
        elif (particle_area <= 0 or particle_minor_axis <= 0 or particle_major_axis <= 0):
            print(particle_centroid_yx[::-1], particle_area, particle_major_axis, particle_minor_axis, 'Error: division by zero, spot aborted')
    # create a pandas dataframe with the segmented particle coordinates and statistics per frame
    particle_segmentation_df = pd.DataFrame()
    particle_segmentation_df['centroid'] = particle_centroid_list
    particle_segmentation_df['minor_axis'] = particle_minor_axis_list
    particle_segmentation_df['major_axis'] = particle_major_axis_list 
    particle_segmentation_df['area'] = particle_area_list
    particle_segmentation_df['aspect_ratio'] = particle_aspect_ratio_list
    
    return particle_segmentation_df


def check_particle_segmentation_parameters(particle_image, phase_image, post_processing=False):
    """
    This code is used to check the segmentation parametetrs.
    The image uneven background is subtracted. 
    Then the user is asked to input the particle segmentation parameters. Default values are recommended.
    All the segmentation parameters are used in the self.particle_segmentation function (line 933)
    
    The segmentation is shown in the particle fluorescence channel.
    
    Parameters
    ----------
    frame: non-negative integer - the frame in the fast time-lapse to be checked
    post_processing: binary - if True the user is asked to provide post-processing parameters

    Returns
    -------
    [0] log_adaptive_parameters
    [1] the maximum particle area
    [2] the minimum particle aspect ratio
    [3] log_adaptive post_processing parameters (if post_processing is False an empty list is returned.
    Providing an empty list in the self.particle_segmentation function will abort post-processing)
    """
    # good default parameters
#        ([4.0, 1000, 98.5, 2.0, 7, -2.0, 0], 3, 150, 0.3, [])
    
#        particle_image = particle.particle_images[0]
#        print('Subtracting background...')
    back_sub_image = bkgf.back_sub(particle_image, bkgf.get_inverted_mask(phase_image), show = True)[0]
#        back_sub_image = particle.back_sub(particle_image)[0]
    
    i = 0
    while i == 0:
        try:
            gaussian_param = float(input('Choose the gaussian smoothing factor (recommended: 4):'))
        except ValueError:
            print('please choose a number')
            gaussian_param = float(input('Choose the gaussian smoothing factor (recommended: 4):'))
        
        try:
            laplace_param = int(input('Choose the laplace filter factor (recommended: 1000):'))
        except ValueError:
            print('please choose a number')
            laplace_param = int(input('Choose the laplace filter factor (recommended: 1000):'))
        
        try:
            log_threshold = float(input('Choose the hard threshold of the LoG filter (recommended: 97):'))
        except ValueError:
            print('please choose a number')
            log_threshold = float(input('Choose the gaussian smoothing factor (recommended: 97):'))
        
        try:
            adaptive_smoothing = float(input('Choose the gaussian smoothing factor before the adaptive thresholding (recommended: 2):'))
        except ValueError:
            print('please choose a number')
            adaptive_smoothing = float(input('Choose the gaussian smoothing factor before the adaptive thresholding (recommended: 2):'))
        
        try:
            block_size_in = int(input('Choose the block size for the adaptive smoothing (recommended: 9, odd number):'))
        except ValueError:
            print('please choose an odd integer positive number')
            block_size_in = int(input('Choose the block size for the adaptive smoothing (recommended: 9, odd number):'))
        
        try:
            offset_in = float(input('Choose the offset for the adaptive smoothing (recommended: -2):'))
        except ValueError:
            print('please choose a number')
            offset_in = float(input('Choose the offset for the adaptive smoothing (recommended: -2):'))
        
        try:
            erosion_in = int(input('Choose the number of erosion rounds for the adaptively thresholded mask (reccomdned: 0):'))
        except ValueError:
            print('please choose a non-negative integer')
            erosion_in = int(input('Choose the number of erosion rounds for the adaptively thresholded mask (reccomdned: 0):'))
        
        try:
            min_particle_size = int(input('Choose the expected minimum size of the particle (recommended: 3):'))
        except ValueError:
            print('please choose a number')
            min_particle_size = int(input('Choose the expected minimum size of the particle (recommended: 3):'))
        
        try:
            max_particle_size = int(input('Choose the expected maximum size of the particle (recommended: 150):'))
        except ValueError:
            print('please choose a number')
            max_particle_size = int(input('Choose the expected maximum size of the particle (recommended: 150):'))
        
        try:
            min_particle_aspect_ratio = float(input('Choose the expected minimum aspect ratio of the aprticle (recommended: 0.3):'))
        except ValueError:
            print('please choose a number')
            min_particle_aspect_ratio = float(input('Choose the expected minimum aspect ratio of the aprticle (recommended: 0.3):'))
        
        if post_processing == True:
            try:
                post_processing_threshold = float(input('Choose the post processing threshold (recommended: 90):'))
            except ValueError:
                print('please choose a number')
                post_processing_threshold = float(input('Choose the post processing threshold (recommended: 90):'))
        elif post_processing ==False:
            post_processing_threshold = None
            

        log_adaptive_parameters = [gaussian_param, laplace_param, log_threshold, adaptive_smoothing, block_size_in, offset_in, erosion_in]
        particle_segmentation_df = particle_segmentation(back_sub_image, log_adaptive_parameters, min_particle_size, max_particle_size, min_particle_aspect_ratio, post_processing_threshold)
        filtered_image = cimf.log_adaptive_filter(back_sub_image, log_adaptive_parameters)
        
        print('The LoG image')
        plt.figure(figsize=(20,20))
        plt.imshow(filtered_image[0])
        plt.show()
        
        print('Press enter to continue')
        input()
        
        print('The adaptively thresholded image')
        plt.figure(figsize=(20,20))
        plt.imshow(filtered_image[1])
        plt.show()
        
        print('Press enter to continue')
        input()
        
        print('Combining the LoG hard thresholded and the adaptively thresholded images')
        plt.figure(figsize=(20,20))
        plt.imshow(filtered_image[2])
        plt.show()
        
        print('Press enter to continue')
        input()
        
        # Print a figure with all the segmented particles and get the particle centers
        print('showing segmentation...')
        fig, ax = plt.subplots(figsize=(20, 20))
        # The vmin and vmax values can be adjusted to change the LUTs, as well as the colormap
        ax.imshow(back_sub_image, cmap='viridis', vmin=100, vmax=1200)
        
        for index, row in particle_segmentation_df.iterrows():
            rect = matplotlib.patches.Rectangle((row['centroid'][0]-row['major_axis']/2,row['centroid'][1]-row['major_axis']/2), row['major_axis'], row['major_axis'], fill=False, edgecolor='coral', linewidth=1)
            ax.add_patch(rect)
            ax.set_axis_off()
            plt.tight_layout()
        
        plt.show()
        
        j = 0
        while j == 0:
            decision = str(input('if the parameters are good choose "g", esle type "b":'))
            decision = decision.lower()
            if decision == 'g':
                i += 1
                j += 1
            elif decision == 'b':
                j += 1
            else:
                print('wrong input, please try again...')
    
    params = [log_adaptive_parameters, min_particle_size, max_particle_size, min_particle_aspect_ratio, post_processing_threshold]
    print(params)
    
    return log_adaptive_parameters, min_particle_size, max_particle_size, min_particle_aspect_ratio, post_processing_threshold
