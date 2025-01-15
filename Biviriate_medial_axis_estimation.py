# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:55:09 2024

@author: Alexandros Papagiannakis, HHMI at Stanford University
"""

import numpy as np
import pandas as pd
from PIL import Image
from skimage.morphology import medial_axis
from shapely.geometry import LineString
from itertools import combinations
import matplotlib.pyplot as plt
import math

 # ANGLE ESTIMATION #
def get_angle_from_slope(displacements, slope='none'):
    """
    This function is used to estimate the angle of each cell segment 
    to determine the search radius of the next segment
    
    Parameters
    ----------
    displacements: tuple - (dx, dy). The displacement between two linked spots in the x and y direction
    slope: float - the slope of the displacement
           if slope == 'none' the slope is estimated as dy/dx
    
    Returns
    -------
    angle: float - the angle of the displacement in degrees
    
    Notes
    -----
    This function is used for the medial axis extimation: self.get_medial_axis()
    """
    dx = displacements[0]
    dy = displacements[1]

    radians = math.atan2(dy,dx)
    degrees = math.degrees(radians)
    if degrees < 0:
        return 360+degrees
    else:
        return degrees
    
    
# THIS CODE CORRECTS THE ANGLE DIFFERENCE
def correct_angle_difference(source_angle, destination_angle):
    """
    This function is used to correct the difference between two angles.
    It returns a positive angle smaller than 180 degrees.
    """
    a = destination_angle - source_angle
    if a >= 180:
        return 360-a
    elif a <= -180:
        return 360+a
    else:
        return abs(a)
    

# THIS CODE ESTIMATES THE NEXT ANCHOR POINT USING THE PREVIOUS ONE AND THE ANGLE
def get_next_position(dist, x, y, angle, list_of_knots, radius_px, half_angle, index_increment):
    """
    This function scans the cell mask distance transformation for the max distance knots
    that will be used to fit the medial axis.
    
    The knots have a resolution of 0.1 pixels.
    """
    dist_temp = dist.copy()
    dist_temp[y][x]=0
    # radius_px = 8 # good parameter
    crop_dist = dist_temp[(y-radius_px):(y+radius_px+1), (x-radius_px):(x+radius_px+1)]
#            col = np.argmax(np.amax(crop_dist, axis=1))
#            row = np.argmax(np.amax(crop_dist, axis=0))
    y_coords, x_coords = np.nonzero(crop_dist)
    intensities = crop_dist[np.nonzero(crop_dist)]
    
    intensity_df = pd.DataFrame()
    intensity_df['x'] = x_coords + x -radius_px
    intensity_df['y'] = y_coords + y -radius_px
    intensity_df['fluor'] = intensities
    
    intensity_df['dx'] =  intensity_df['x']-x
    intensity_df['dy'] =  intensity_df['y']-y
    intensity_df['distance'] = np.sqrt(intensity_df.dx**2 + intensity_df.dy**2)
    intensity_df['angle'] = intensity_df.apply(lambda row: get_angle_from_slope((row.dx, row.dy)), axis=1)
    intensity_df['angle_dif'] = intensity_df.apply(lambda row: correct_angle_difference(angle, row.angle), axis=1)
    
    intensity_df = intensity_df[(intensity_df.angle_dif<=half_angle) & (intensity_df.distance>6)] 
    
    if intensity_df.shape[0]>0:
        max_df = intensity_df[intensity_df.fluor == intensity_df.fluor.max()] #new
        if max_df.shape[0] > 0:
            if max_df.shape[0] > 1:
                max_df = max_df[max_df.angle_dif==max_df.angle_dif.min()] #new

            new_x = max_df.x.values[0]
            new_y = max_df.y.values[0]
            new_angle = max_df.angle.values[0]
            max_fluor = max_df.fluor.values[0]
        
            if (new_x, new_y) not in list_of_knots:
                if max_fluor >= 3:
#                        print(new_x,  new_y, new_angle)
                    return new_x,  new_y, new_angle
                elif max_fluor < 3:
                    print('This is the end of the cell:', index_increment)
                    return False
            elif (new_x, new_y) in list_of_knots:
                print('This is the end of the cell and a loop is formed:', index_increment)
                return 'loop'
        else:
            print('This is the end of the cell:', index_increment)
            return False
    elif intensity_df.shape[0]==0:
        print('This is the end of the cell:', index_increment)
        return False


# RECURSIVE ALGORITHM TO GET ALL ANCHOR POINTS
def recursive_medial_axis(index_increment, dist, x, y, angle, index, list_of_knots, radius_px, half_angle, xyz_coord_list):
    """
    This is a function that runs the "get_next_position" finction recursively.
    """
    new_knot =get_next_position(dist, x, y, angle, list_of_knots, radius_px, half_angle, index_increment)
    if new_knot != False:
        if new_knot != 'loop':
            new_x,new_y,new_angle = new_knot
            list_of_knots.append((new_x, new_y))
#                print(new_x, new_y, new_angle)
            if index_increment == 1:
                index += 1
            if index_increment == -1:
                index -= 1
            xyz_coord_list.append((new_x, new_y, index))
            
            x_list, y_list, z_list = list(zip(*xyz_coord_list))
            pre_df = pd.DataFrame()
            pre_df['x'] = np.array(x_list)/10
            pre_df['y'] = np.array(y_list)/10
            pre_df['z'] = z_list
            pre_df = pre_df.sort_values(['z'])

            line_coords = list(map(lambda x, y:(x,y), pre_df.x, pre_df.y))
            line = LineString(line_coords)
#                input()
            if line.is_simple == True:
                recursive_medial_axis(index_increment, dist, new_x, new_y, new_angle, index, list_of_knots, radius_px, half_angle, xyz_coord_list)
            elif line.is_simple == False:
                print('This is the end of the cell and a loop is formed...')
        # remove the loops
        elif new_knot == 'loop':
            for i in range(20):
                xyz_coord_list.pop()

# radius_px=8, half_angle=22, cap_knot=13, max_degree=60, verbose=True
def get_medial_axis(cropped_cell_mask, radius_px=8, half_angle=22, cap_knot=13, max_degree=60, verbose=False):
    """
    This function construct the medial axis of a signle cell, 
    as well as the relative coordinates of the cell from one pole to the other.
    
    Parameters
    ----------
    cropped_cell_mask: 2D numpy array (binary) of the mask of an individual cell. This mask can be cropped around the cell boundaries
    radius_px: positive integer - the radius beyond which the algorithm searches for the next anchor point in the circle sector
    half_angle: positive_float - the half angle of the circle sector within which the next anchor point is searched for.
        e.g. an angle of 22 degrees indicates that the next angle will be searched for in the secotr of 44 degrees (quartile), 22 degrees adjuscent the previous orientation
    cap_knot: positive_interger - the number of knots excluded at the poles. This region will be extended using the anlge from the previous 10 anchors. 
    max_degree: positive_integer - the maximum degree of the fitted polynomials
    verbose: binary (False/Trye) - determines if the medial axis instances will be plotted
    
    Returns
    -------
    [0] A pandas DataFrame including the absolute, scaled and relative coordinates of the medial axis
        Columns:
            'cropped_x': the cropped x coordinates of the medial axis (cropped by the cell pad)
            'cropped_y': the croppedyx coordinates of the medial axis (cropped by the cell pad)
            'arch_length': the arch length of the medial axis along the cell length
            'arch_length_centered': the arch length of the medial axis scaled by the centroid
            'arch_length_scaled': the relative arch length from -1 to 1
    [1] The cropped x,y coordinates of the cell centroid, cropped by the cell pad
    """

    # GET THE CELL MASK AND THE DISTANCE TRANSFORMATION OF THE CELL MASK
    cell_mask = cropped_cell_mask
    # dilate the cell mask if needed
#        dilated_cell_mask =  scipy.ndimage.morphology.binary_dilation(cell_mask, iterations=2)
    # get the resized cell mask
    resized_cell_mask = np.array(Image.fromarray(cell_mask).resize((cell_mask.shape[1]*10, cell_mask.shape[0]*10), resample=Image.NEAREST))
    skel, dist = medial_axis(resized_cell_mask, return_distance=True)
    
    # GET THE FIRST ANCHOR POINT AT THE CENTER OF THE MAX LENGTH DIMENSION
    # CORRESPONDING TO THE POINT WITH THE MAXIMUM DISTANCE FROM THE CELL EDGES
    # THE ANGLE OF THE CELL AT THE FIRST ANCHOR POINT IS ALSO ESTIMATED
    len_y = len(dist)
    len_x = len(dist[0])
    length = (len_x, len_y)
    # the index of the longest coordinate (x or y)
    max_index = length.index(max(length))
    if max_index == 0:
        half_x = int(round(len_x/2 ,0))
        half_y = np.argmax(np.transpose(dist)[half_x])
    elif max_index == 1:
        half_y = int(round(len_y/2, 0))
        half_x = np.argmax(dist[half_y])
        
    start_x = half_x
    start_y = half_y
    cropped_window = dist[(start_y-10):(start_y+11), (start_x-10):(start_x+11)]
#        plt.imshow(cropped_window)
#        plt.show()
    window_df = pd.DataFrame()
    window_df['x'] = np.nonzero(cropped_window)[1]
    window_df['y'] = np.nonzero(cropped_window)[0]
    window_df['fluor'] = cropped_window[np.nonzero(cropped_window)].ravel()
    window_df['distance'] = np.sqrt((window_df.x-10)**2 + (window_df.y-10)**2)
    window_df = window_df[window_df.distance>5]
    window_df = window_df[window_df.fluor == window_df.fluor.max()]
    if window_df.shape[0] == 1:
        start_angle = get_angle_from_slope((window_df.x.values[0]-10, window_df.y.values[0]-10))
    elif window_df.shape[0] > 1:
        window_df = window_df[window_df.distance == window_df.distance.max()]
        start_angle = get_angle_from_slope((window_df.x.values[0]-10, window_df.y.values[0]-10))
#        print(start_angle)


    # Run the recursive algorithms to get the anchor points for the central line fit
    index_increment = 1 # run towards one dimension
    index = 0
    list_of_knots = [(start_x, start_y)]
    xyz_coord_list = [(start_x, start_y, index)]
    x = start_x
    y = start_y
    angle = start_angle
    recursive_medial_axis(index_increment, dist, x, y, angle, index, list_of_knots, radius_px, half_angle, xyz_coord_list)
    xyz_coord_list_1 = xyz_coord_list.copy()
    index_increment = -1 # run towards the opposite dimension
    index = 0
    list_of_knots = [(start_x, start_y)]
    xyz_coord_list = [(start_x, start_y, index)]
    x = start_x
    y = start_y
    angle = start_angle + 180
    if angle >= 360:
        angle = angle-360
    recursive_medial_axis(index_increment, dist, x, y, angle, index, list_of_knots, radius_px, half_angle, xyz_coord_list)
    
    xyz_coord_list = xyz_coord_list_1 + xyz_coord_list[1:] # combine the two lists of coordinates
    
    # GETTING THE XY COORDINATES OF THE ANCHOR POINTS
    # getting the x,y and z coordinates of the knots
    x_list, y_list, z_list = list(zip(*xyz_coord_list))
#        plt.figure(figsize=(10,10))
#        plt.imshow(resized_cell_mask)
#        plt.plot(x_list, y_list, 'o', markersize=1)
#        plt.show()
#        plt.figure(figsize=(10,10))
#        plt.scatter(x_list, y_list, c=z_list)
#        plt.show()
    # CHECKING IF THE CENTRAL LINE INTERSECTS ITSELF AND REMOVING THE LOOPS
    # rescaling and sorting the coordinates of the knots
    pre_df = pd.DataFrame()
    pre_df['x'] = np.array(x_list)/10
    pre_df['y'] = np.array(y_list)/10
    pre_df['z'] = z_list
    pre_df = pre_df.sort_values(['z'])
    line_coords = list(map(lambda x, y:(x,y), pre_df.x, pre_df.y))
    line_coords = line_coords[::2]
    line = LineString(line_coords)
    positive_intersection = False
    negative_intersection = False
    if line.is_simple == False:
        print('removing the loop...')
        list_of_lines = []
        line_intersections = []
        for pnt in range(len(line_coords)-1):
            list_of_lines.append(LineString(line_coords[pnt:pnt+2]))
        for l1, l2 in combinations(list_of_lines,2): #For all combinations of segments
#                print(l1.intersection(l2).coords[:])
            if l1.crosses(l2) == True: #Find crossings
#                    print('cross')
                line_intersections.append(l1.intersection(l2).coords[:][0])
        intersection_points_positive = []
        intersection_points_negative = []
        for intersect in line_intersections:
            distance_df = pre_df.copy()
            distance_df['inter_distance'] = np.sqrt((distance_df.x - intersect[0])**2+(distance_df.y - intersect[1])**2)
            distance_df = distance_df.sort_values(['inter_distance'])
            distance_df = distance_df[0:4]
            intersection_point = distance_df[distance_df.z.abs()==distance_df.z.abs().min()].z.values[0]
            if intersection_point > 0:
                intersection_points_positive.append(intersection_point)
            elif intersection_point < 0:
                intersection_points_negative.append(intersection_point)
        if len(intersection_points_positive) > 0:
            pre_df = pre_df[pre_df.z < intersection_points_positive[np.argmax(intersection_points_positive)]]
            positive_intersection = True
        elif len(intersection_points_negative) > 0:
            pre_df = pre_df[pre_df.z > intersection_points_negative[np.argmin(intersection_points_negative)]]
            negative_intersection = True
#        plt.imshow(resized_cell_mask)
#        plt.plot(pre_df.x*10, pre_df.y*10, 'o')
#        plt.show()
            
    # TRUNCATE THE MEDIAL AXIS COORDINATES FROM THE EDGES
    if pre_df.shape[0]>2*cap_knot+5:
        pre_df_max_index = cap_knot
    elif pre_df.shape[0]<=2*cap_knot+5:
        pre_df_max_index = pre_df.shape[0]/2 - 5

    if positive_intersection == False and negative_intersection == False:
        truncated_df = pre_df[int(pre_df_max_index):-int(pre_df_max_index)]
    elif positive_intersection == False and negative_intersection == True:
        truncated_df = pre_df[0:-int(pre_df_max_index)]
    elif positive_intersection == True and negative_intersection == False:
        truncated_df = pre_df[int(pre_df_max_index):]
    elif positive_intersection == True and negative_intersection == True:
        truncated_df = pre_df
#        plt.imshow(cell_mask)
#        plt.plot(truncated_df.x, truncated_df.y, 'o', markersize=0.2)
#        plt.show()
    
    # EXTENDING THE CENTRAL LINE AT THE EDGES USING THE AVERAGE ANGLE FROM THE 10 PREVIOUS ANCHORS
    # For the negative side
    if truncated_df.shape[0] >= 10:
        trunc_index = 10
    elif truncated_df.shape[0] < 10:
        trunc_index = truncated_df.shape[0]
    
    slope_1_df = truncated_df[0:trunc_index]
    slope_1 = np.polyfit(slope_1_df.x, slope_1_df.y, 1)[0]
    x_1 = np.array(slope_1_df.x)
    y_1 = np.array(slope_1_df.y)
    dx_1 = round(x_1[0] - x_1[trunc_index-1], 0)
    dy_1 = round(y_1[0] - y_1[trunc_index-1],0)

    angle_1 = get_angle_from_slope((dx_1, dy_1), slope_1)
    
    # For the positive side
    slope_2_df = truncated_df[-trunc_index:]
    slope_2 = np.polyfit(slope_2_df.x, slope_2_df.y, 1)[0]
    x_2 = np.array(slope_2_df.x)
    y_2 = np.array(slope_2_df.y)
    dx_2 = round(x_2[trunc_index-1] - x_2[0],0)
    dy_2 = round(y_2[trunc_index-1] - y_2[0],0)
    angle_2 = get_angle_from_slope((dx_2, dy_2), slope_2)
    
    min_z = truncated_df.z.min()
    max_z = truncated_df.z.max()
    
    x_list_1 = [x_1[0]]
    y_list_1 = [y_1[0]]
    z_list_1 = [min_z]
    
    # extend towards the negative side using the average angle at the edge of the central line
    for i in range(55):
        x_list_1.append(x_list_1[-1]+0.5*np.cos(angle_1*np.pi/180))
        y_list_1.append(y_list_1[-1]+0.5*np.sin(angle_1*np.pi/180))
        z_list_1.append(z_list_1[-1]-1)
    
    x_list_2 = [x_2[-1]]
    y_list_2 = [y_2[-1]]
    z_list_2 = [max_z]
    
    # extend towards the positive side using the average angle at the edge of the central line
    for i in range(55):
        x_list_2.append(x_list_2[-1]+0.5*np.cos(angle_2*np.pi/180))
        y_list_2.append(y_list_2[-1]+0.5*np.sin(angle_2*np.pi/180))
        z_list_2.append(z_list_2[-1]+1)
        
    x_list_final = x_list_1[1:]+x_list_2[1:]
    y_list_final = y_list_1[1:]+y_list_2[1:]
    z_list_final = z_list_1[1:]+z_list_2[1:]
    
    pre_df_2 = pd.DataFrame()
    pre_df_2['x'] = x_list_final
    pre_df_2['y'] = y_list_final
    pre_df_2['z'] = z_list_final
    pre_df = pd.concat([truncated_df, pre_df_2])
    pre_df = pre_df.sort_values(['z'])
#        plt.imshow(cell_mask)
#        plt.plot(pre_df.x, pre_df.y)

    # FIT A nth DEGREE POLYNOMIAL TO THE EXTENDED CENTRAL LINES
    # use this code if a polynomial fit is preferred versus a bivariate spline
    polynomial_degree = int(pre_df.shape[0]/10-5)
    if polynomial_degree > max_degree:
        polynomial_degree = max_degree
    print('fitting polynomial fucntions of degree:', polynomial_degree)
    extended_z = np.arange(pre_df.z.min(), pre_df.z.max(), 0.01)
    fit_x = np.polyfit(pre_df.z, pre_df.x, polynomial_degree) #nth degree polynomial fit long the X axis
    x_hat = np.polyval(fit_x, extended_z)
#        plt.plot(pre_df.z, pre_df.x, 'o')
#        plt.plot(extended_z, x_hat)
#        plt.show()
    fit_y = np.polyfit(pre_df.z, pre_df.y, polynomial_degree) #nth degree polynomail fit along the Y axis
    y_hat = np.polyval(fit_y, extended_z)
#        plt.plot(pre_df.z, pre_df.y, 'o')
#        plt.plot(extended_z, y_hat)
#        plt.show()

    # REMOVE THE CENTRAL LINE COORDINATES THAT DO NOT FALL INTO THE CELL MASK
    # getting only those coordinates of the medial axis that fall into the original cell mask
    x_fine_round = np.around(x_hat,0).astype(int)
    y_fine_round = np.around(y_hat,0).astype(int)
    good_indexes = (x_fine_round<cell_mask.shape[1])*(y_fine_round<cell_mask.shape[0])
    good_indexes_2 = (x_fine_round>=0)*(y_fine_round>=0)
    good_indexes= good_indexes * good_indexes_2
    x_fine = x_hat[good_indexes]
    y_fine = y_hat[good_indexes]
    x_fine_round = x_fine_round[good_indexes]
    y_fine_round = y_fine_round[good_indexes]
    nonzero_medial_indexes = np.nonzero(cell_mask[y_fine_round, x_fine_round])     
    x_fine_good = x_fine[nonzero_medial_indexes]
    y_fine_good = y_fine[nonzero_medial_indexes]
#        plt.imshow(cell_mask)
#        plt.plot(x_fine_good, y_fine_good)
#        plt.show()

    # GENERATE THE RELATIVE CELL COORDINATES AND THE CENTROID
    # generate the medial axis dataframe
    medial_axis_df = pd.DataFrame()
    medial_axis_df['cropped_x'] = x_fine_good
    medial_axis_df['cropped_y'] = y_fine_good
    # get the arch length of the medial axis
    delta_x_sqr = (x_fine_good[1:] - x_fine_good[0:-1])**2
    delta_y_sqr = (y_fine_good[1:] - y_fine_good[0:-1])**2
    disp_array = np.sqrt(delta_x_sqr + delta_y_sqr)
    disp_list = [0]
    for disp in disp_array:
        disp_list.append(disp_list[-1]+disp)
    medial_axis_df['arch_length'] = disp_list 
    medial_axis_df['arch_length_centered'] = disp_list - np.max(disp_list)/2
    medial_axis_df['arch_length_scaled'] = medial_axis_df['arch_length_centered'] / medial_axis_df['arch_length_centered'].max()
    # get the cropped centroid of the medial axis
    center_df = medial_axis_df[medial_axis_df.arch_length_centered.abs()==medial_axis_df.arch_length_centered.abs().min()]
    cropped_centroid = (center_df.cropped_x.mean(), center_df.cropped_y.mean())
    # PLOT THE CELL MASK WITH THE CENTRAL LINE AND THE CENTROID
    # plot and medial axis on the cell mask and the centroid
    if verbose == True:
        plt.imshow(cell_mask)
        plt.plot(medial_axis_df.cropped_x, medial_axis_df.cropped_y, color='red')
        plt.plot(cropped_centroid[0], cropped_centroid[1], 'o')
        plt.show()
        plt.close("all")
        
    # get the original centroid coordinates from the cropped centroid
    
    return medial_axis_df, cropped_centroid


def get_oned_coordinates(cell_mask, medial_axis_df): 
    """Projects the medial axis pixels on the medial axis

    Args:
        cell_mask (numpy array): 2D numpy array of the cropped cell mask (binary)
        medial_axis_df (Pandas datraframe): the product of the get_medial_axis function
    """

    cell_mask_df = pd.DataFrame()
    cell_mask_df['x'] = np.nonzero(cell_mask)[1]
    cell_mask_df['y'] = np.nonzero(cell_mask)[0]
#    cell_mask_df['z'] = fluor_image[np.nonzero(cell_mask)]

    def get_pixel_projection(pixel_x, pixel_y, medial_axis_df):
        
        medial_axis_df['pixel_distance'] = np.sqrt((medial_axis_df.cropped_x-pixel_x)**2+(medial_axis_df.cropped_y-pixel_y)**2)
        min_df = medial_axis_df[medial_axis_df.pixel_distance == medial_axis_df.pixel_distance.min()]
        min_arch_centered_length = min_df.arch_length_centered.values[0]
        min_arch_scaled_length =  min_df.arch_length_scaled.values[0]
        min_distance_abs = min_df.pixel_distance.values[0]
        min_index = min_df.index.values[0]
        medial_axis_coords = (min_df.cropped_x.values[0], min_df.cropped_y.values[0])
        
        def get_relative_distance(min_distance_abs, medial_axis_df, min_index, medial_axis_coords, pixel_x, pixel_y):
            
            half_window = 5
            
            if min_index>=half_window and min_index<medial_axis_df.index.max()-half_window:
                index_range = (min_index-half_window, min_index+half_window)
            elif min_index<half_window and min_index<medial_axis_df.index.max()-half_window:
                index_range = (0, min_index+half_window)
            elif min_index>=half_window and min_index>=medial_axis_df.index.max()-half_window:
                index_range = (min_index-half_window, medial_axis_df.index.max())
            
            delta_x = (medial_axis_df.iloc[index_range[1]].cropped_x -  medial_axis_df.iloc[index_range[0]].cropped_x)
            delta_y = (medial_axis_df.iloc[index_range[1]].cropped_y -  medial_axis_df.iloc[index_range[0]].cropped_y)
            medial_axis_vector = [delta_x, delta_y]
            
            delta_x = pixel_x - medial_axis_coords[0]
            delta_y = pixel_y - medial_axis_coords[1]
            pixel_vector = [delta_x, delta_y]
            
            cross_product = np.cross(medial_axis_vector, pixel_vector)
            if cross_product != 0:
                min_distance = np.sign(cross_product)*min_distance_abs
            elif cross_product == 0:
                min_distance = 0
            return min_distance
        
        min_distance = get_relative_distance(min_distance_abs, medial_axis_df, min_index, medial_axis_coords, pixel_x, pixel_y)
    
        return min_arch_centered_length, min_arch_scaled_length, min_distance
        
            
    cell_mask_df['oned_coords'] = cell_mask_df.apply(lambda x: get_pixel_projection(x.x, x.y, medial_axis_df), axis=1)
    cell_mask_df[['arch_length', 'scaled_length', 'width']] = pd.DataFrame(cell_mask_df.oned_coords.to_list(), index=cell_mask_df.index)
    cell_mask_df = cell_mask_df.drop(['oned_coords'], axis=1)
    
    return cell_mask_df
