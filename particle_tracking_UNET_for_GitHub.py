#Python 3.7.0 (default, Jun 28 2018, 07:39:16)
#Type "copyright", "credits" or "license" for more information.
#
#IPython 7.8.0 -- An enhanced Interactive Python.

"""
Created on Mon Aug  5 14:32:50 2019

@author: Alexandros Papagiannakis, Christine Jacobs-Wagner lab, Stanford University 2021
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import skimage
from skimage import io
import scipy
from scipy import ndimage
from shapely.geometry import Point,Polygon
import pickle
import os
import nd2_to_array as ndtwo # Import from the Image-analysis repository
import background_correction as bkgf
import twod_gaussian_fit as tgf
import particle_segmentation as ps
import particle_tracking_methods as ptm
import image_drift as imd


class particle_tracking(object):
    """
    Developer: Alexandros Papagiannakis, Christine Jacobs-Wagner lab, Stanford University, 2020
    
    This class contains all the functions used for particle tracking, particle MSD and speed calculation, 
    correlation between macromolecular and the cell cycle, the ribosomal concentration or the N/C ratio.
    The disclacement of the tracked particles is estimated in the 2D cell projection, as well as relative
    to the centroid or across the medial axis (1D particle projection).
    
    Additional fucntions construct the medial axis of the single cells and the level of constriction.
    
    The code allows for making movies of the tracked particles. 
    
    Functions included in this class:
        __init__
            unet_to_python
            show_unet_masks
        
        run_medial_axis
        
        check_cell_segmentation

        getting_the_particles
        
        run_particle_tracking
        
        show_all_trajectories
    """
    def __init__(self, unet_path, snapshots_paths, fast_time_lapse_path, experiment, position, interval, save_path):
        """
        The class is iniialized.
        
        Parameters
        ----------
        ___File paths___
        fast_time_lapse_path: the path of the fast time-lapse images of the tracked particle. The path to the nd2 file
        snapshots_paths: the path of the phase and signal images (e.g. HU-mCherry) snapshots. The path to the nd2 file
            A list of paths should be included with the right order (e.g. [phase_path, phase_after_path, signal_path] or [phase_path, signal_path])
            In the case of an empty lists, no snapshots are loaded
        unet_path: the path to the unet .tif cell labels
            If a non-valud file name is used no cell masks are loaded
            For instance, use 'none' to avoid loading cell masks
        save_path: path where the results are saved
        
        ___General parameters___
        experiment: a string describing the experiment (e.g. '07202919_seqA')
        position: a non-negatice integer corresponding to the XY position in the experimenbt (e.g. 0 for the first XY position - XY01)
        interval: the time interval of the stream acquisition in msec
            This is needed since the ND2_Reader function is deprecated and the interval is not returned
            
        The _init_ function contains a number of sub-functions which use the input parameters of the class
        
        Exceptions
        ----------
        If the unet_path is not a file (e.g. 'none') then the unet_to_python function is not implemented
        If the length of the snapshots_paths list is zero, then no snapshots paths are loaded. Only the stream acquisition is loaded.
        """
       
        def unet_to_python(unet_path, pad=5):
            """
            This function incoorporates the masks from Unet to python
            
            Parameters
            ----------
            unet_path: string - the path of the tif image of the cell masks returned by the Unet
            pad: integer - the size of the frame around the cells used to crop the cell masks
            
            Returns
            -------
            [0] cell_masks: a dictionary which inlcudes individual cell masks
            [1] cropped_cell_masks: a dictionary which inlcudes the cropped cell masks. A cropping pad of 3 pixels is used
            """
            # check for bad cells in the results destination
            if self.experiment + '_' + self.position_string + '_bad_cells' in os.listdir(self.save_path):
                with open(self.save_path+'/'+self.experiment+'_'+self.position_string+'_bad_cells', 'rb') as handle:
                    bad_cells = pickle.load(handle)
                    print(len(bad_cells), 'badly segmented cells were identified and excluded from the analysis...')
            else:
                bad_cells = []
                
            mask_array = io.imread(unet_path)
              
            cell_masks = {}
            cropped_cell_masks = {}
            pads = {}
            cell_area_px = {}
            center_of_mass = {}
            cropped_center_of_mass = {}
            cell_meshes = {}
            cell_polygons = {}
            
            for cell in range(1, mask_array.max()+1):
                cell_id = self.experiment+'_'+self.position_string+'_'+str(cell)
                if cell_id not in bad_cells:
                    cell_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]))
                    cell_mask[mask_array==cell]=1
                    cell_mask = ndimage.morphology.binary_fill_holes(cell_mask).astype(int)
                    dilated_cell_mask = scipy.ndimage.morphology.binary_dilation(cell_mask, iterations=1)
                    y_mask_coords, x_mask_coords = np.where(cell_mask==1)
                    minx,miny,maxx,maxy = x_mask_coords.min(), y_mask_coords.min(), x_mask_coords.max(), y_mask_coords.max()
                    # remove the masks at the edge of the sensor
                    if maxy < mask_array.shape[0]-pad and maxx < mask_array.shape[1]-pad and miny > pad and minx > pad:
                        cell_masks[cell_id] = cell_mask
                        cropped_cell_masks[cell_id] = cell_mask[(miny-pad):(maxy+pad), (minx-pad):(maxx+pad)]
    #                    plt.imshow(cropped_cell_masks[cell_id])
    #                    plt.show()
                        pads[cell_id] = (minx-pad, miny-pad, maxx+pad, maxy+pad)
                        cell_area_px[cell_id] = np.nonzero(cropped_cell_masks[cell_id])[0].shape[0]
                        center_of_mass[cell_id] = ndimage.measurements.center_of_mass(cell_masks[cell_id])
                        cropped_center_of_mass[cell_id] = ndimage.measurements.center_of_mass(cropped_cell_masks[cell_id])
                        cell_mesh_y, cell_mesh_x = zip(*skimage.measure.find_contours(dilated_cell_mask, level=0.5)[0])
                        cell_polygons[cell_id] = Polygon(tuple(zip(cell_mesh_x,cell_mesh_y)))
                        cell_meshes[cell_id] = [cell_mesh_x, cell_mesh_y]
            return cell_masks, cropped_cell_masks, pads, cell_area_px, center_of_mass, cropped_center_of_mass, cell_meshes, cell_polygons
                
        # Get the images as numpy arrays from the nd2 files
        if len(snapshots_paths) > 1:
            # Get the images from the nd2 files
            phase_before_snapshots = ndtwo.nd2_to_array(snapshots_paths[0])
            # snapshots_iteration_axis = phase_before_snapshots[0]
            scale = phase_before_snapshots[6]
            sensor = phase_before_snapshots[7]
            snapshot_channels = ['phase_before']
            phase_before_arrays = phase_before_snapshots[2]
            self.phase_image = phase_before_arrays
            self.phase_before_metadata = phase_before_snapshots[1]
            
            phase_after_snapshots = ndtwo.nd2_to_array(snapshots_paths[1])
            # phase_after_iteration_axis = phase_after_snapshots[0]
            snapshot_channels.append('phase_after')
            phase_after_arrays = phase_after_snapshots[2]
            self.phase_image_after = phase_after_arrays
            self.phase_after_metadata = phase_after_snapshots[1]
                
            if len(snapshots_paths) == 3:
                signal_snapshots = ndtwo.nd2_to_array(snapshots_paths[2])
                # signal_iteration_axis = signal_snapshots[0]
                snapshot_channels.append('cell_marker')
                signal_arrays = signal_snapshots[2]
                self.signal_image = signal_arrays
                self.signal_metadata = signal_snapshots[1]
            
        elif len(snapshots_paths) == 1:
            snapshots = ndtwo.nd2_to_array(snapshots_paths[0])
            scale = snapshots[6]
            sensor = snapshots[7]
            channels = snapshots[3]
            if len(channels) == 1:
                phase_before_arrays = snapshots[2][channels[0]]
                snapshot_channels = ['phase_before']
            elif len(channels)==2:
                phase_before_arrays = snapshots[2][channels[0]]
                phase_after_arrays = snapshots[2][channels[1]]
                snapshot_channels = ['phase_before', 'phase_after']
            elif len(channels)==3:
                phase_before_arrays = snapshots[2][channels[0]]
                phase_after_arrays = snapshots[2][channels[2]]
                signal_arrays = snapshots[2][channels[1]]
                snapshot_channels = ['phase_before', 'phase_after', 'cell_marker']
            self.phase_image = phase_before_arrays
            self.phase_image_after = phase_after_arrays
            self.signal_image = signal_arrays
            self.phase_before_metadata = snapshots[1]
            self.phase_after_metadata = snapshots[1]
            self.signal_metadata = snapshots[1]
        
        elif len(snapshots_paths) == 0:
            print('This experiment contains no snapshots, only particle stream axquisitions')
            
        # get the particle images from the stream acquisition
        stream_images = ndtwo.nd2_to_array(fast_time_lapse_path)
        particle_images = stream_images[2] # numpy array
        scale = stream_images[6]
        sensor = stream_images[7]
        self.particle_images = particle_images
        
        if position <= 8:
            position_string = 'XY0'+str(position+1)
        elif position > 8:
            position_string = 'XY'+str(position+1)
            
        self.fast_time_lapse_path = fast_time_lapse_path
        self.experiment = experiment
        self.position = position
        self.save_path = save_path
        self.position_string = position_string
        self.stream_metadata = stream_images[1]
        self.scale = scale  # μm/px scale
        self.sensor = sensor
        self.n_frames = stream_images[4]
#        self.interval = round(ND2Reader(fast_time_lapse_path).metadata['experiment']['loops'][0]['sampling_interval'], 1)
        # the interval output is deprecated
        self.interval = interval
        
        if os.path.isfile(unet_path):
            self.unet_path = unet_path
            self.cell_masks, self.cropped_cell_masks, self.pads, self.cell_area_pix, self.center_mass, self.cropped_center_mass, self.cell_meshes, self.cell_polygons = unet_to_python(unet_path, pad=5)
            self.cells = list(self.cell_masks.keys())
            print(len(self.cells), 'cells in total')
        else:
            print('No cell masks available.')
            
        if len(snapshots_paths) > 0:
            self.snapshots_paths = snapshots_paths
            self.snapshot_channels = snapshot_channels
        
    
    # CHECK THE UNET SEGMENTATION AND THE MEDIAL AXIS DEFINITION
    def show_unet_masks(self):
    # def show_oufti_meshes(self, channel_offsets):
        """
        This function can be used to plot the perimeter of the Unet masks after processing.
        The perimeter of the masks is plotted over the phase image.
        """

        print('printing the cell boundaries on the phase contrast image...')
        plt.figure(figsize=(20,20))
        plt.imshow(self.phase_image)
        for test_cell in self.cell_masks:
            plt.plot(self.cell_meshes[test_cell][0], self.cell_meshes[test_cell][1], color='white')
        plt.show()
    
    
    def check_cell_segmentation(self):
        """
        This function is used to check the cell meshes and cell segmentation.
        The cells that are annotated as badly segmented are stored in a list in the save_path
        """
        bad_cells = []
        
        for cell in self.cell_masks:
            plt.imshow(self.phase_image[self.pads[cell][1]:self.pads[cell][3], self.pads[cell][0]:self.pads[cell][2]])
            plt.plot(self.cell_meshes[cell][0]-self.pads[cell][0], self.cell_meshes[cell][1]-self.pads[cell][1], color='white')
            plt.show()
            
            loop = 0
            while loop == 0:
                cell_segmentation = str(input('Choose "g" for good cell and "b" for bad cell:'))
                cell_segmentation = cell_segmentation.lower()
                
                if cell_segmentation == '0' or cell_segmentation == 'b':
                    print('cell: '+cell+' is included in the list of bad segmentations')
                    print()
                    bad_cells.append(cell)
                    loop+=1
                
                elif cell_segmentation == '1' or cell_segmentation == 'g':
                    print('cell: '+cell+' is well segmented')
                    print()
                    loop+=1
                
                else:
                    print('wrong input, please try again')
                    
        with open(self.save_path+'/'+self.experiment+'_'+self.position_string+'_bad_cells', 'wb') as handle:
            pickle.dump(bad_cells, handle)
        
        return bad_cells

    
    def test_segmentation_parameters(self, frame, post_process=False):
        
        ps.check_particle_segmentation_parameters(self.particle_images[frame], self.phase_image, post_process)
        
        
    def cell_contains_particle(self, gaussian_particle_center, bkg_cor_image, channel_offset=(0,0), cell_show=False):
        """
        This function checks if the particle belongs to a segmented cell and returns the cell ID or np.nan otherwise.
        
        Parameters
        ----------
        gaussian_particle_center: (x,y) tuple of floats which correspond to the subpixel particle coordinates
        bkg_cor_image: 2D numpy array - the background corrected particle fluorescence image
        channel_offset: tuple of (x,y) floats - the offset between the phase and particle fluorescence images
                        In our images there is no offset since a triple-ex cube is used
                        This parameter is useful if the filter cubes are switched in the turret between channels
        cell_show: bool - choose True if you wish to see the particle in the segmented cell
                          otherwise choose False
        
        Returns
        -------
        The cell ID - string or np.nan if the particle does not belong in a segmented cell
        """
        # get the particle point object from the corrected gaussian center
        particle_point = Point(gaussian_particle_center)
    #--------------------------------- ASSIGN PARTICLES TO CELLS --------------------------#
        # check if the particle point is in a cell
        cells_containing_particle = []
        # check how deep the particle is in the cell
        for cell in self.cell_polygons:
            # dilate the polygon by two pixels
            dilated_polygon = self.cell_polygons[cell].buffer(1)
            # check if the cell polygon contains the particle center
            if dilated_polygon.contains(particle_point):
                cells_containing_particle.append(cell)
        # if the particle is assigned to one cell only include the cell ID
        if len(cells_containing_particle) == 1:
            cell_input = cells_containing_particle[0]
            if cell_show == True:
                print(cell_input)
                image = bkg_cor_image[(self.pads[cell_input][1]-channel_offset[1]):(self.pads[cell_input][3]-channel_offset[1]), (self.pads[cell_input][0]-channel_offset[0]):(self.pads[cell_input][2]-channel_offset[0])]
#                    image = np.array(phase_image)[pad3[2]:pad3[3], pad3[0]:pad3[1]]
                fig, ax = plt.subplots(figsize=(10, 5))
                circle3 = matplotlib.patches.Circle((gaussian_particle_center[0]-self.pads[cell_input][0],gaussian_particle_center[1]-self.pads[cell_input][1]), radius=3, color='red', fill=False, linewidth=1)
                plt.imshow(image)
                plt.plot(self.cell_meshes[cell_input][0]-self.pads[cell_input][0],self.cell_meshes[cell_input][1]-self.pads[cell_input][1],'w')
                ax.add_patch(circle3)
                plt.show()
                print('press ENTER to continue..')
                input()
        # else if the particle is not assigned to any cell 
        elif len(cells_containing_particle) == 0:
            cell_input = np.nan
        # else if a particle is assigned to more than one cells, pick the cell in which the particle is more deeply located
        elif len(cells_containing_particle) > 1:
            cell_input = np.nan
        
        return cell_input
    
    
    def getting_the_particles(self, log_adaptive_parameters, min_particle_size, max_particle_size, min_particle_aspect_ratio, post_processing_threshold, box_size, analysis_range, metric, operation,  gaussian_fit_show = False, cell_show=False, channel_offset=(0,0)):
        """
        This function is used to run the segmentation of the particles and track them.
        Gaussian distributions are fitted to the difraction limited particles to estimate the volume and the center of each particle. 
        
        Parameters
        ----------
        channel_offset: tuple of (x,y) floats - the 2D offset between the particle fluorescence and phase contrast images
        analysis_range: tuple of non-negaive integers: the first and the last frame to be analyzed
        log_adaptive_parameters: list of the particle segmentation parameters. See particle_segmentation function.
        min_particle_size: Non-negative integer: the minimum expected particle size in pixels. 
        max_particle_size: Positive integer: the maximum expected particle size in pixels.
        min_particle_aspect_ratio: float: the minimum expected particle aspect ration (minor/major axis length)
        post_processing_threshold: float or integer: the % threshold of brightest pixels to separate clustered spots.
        box_size: odd integer - the size of the box that is used to fit the 2D gaussian for the particle center estimation (5 or 11 suggested)
        metric: string, the column of the pandas dataframe which is used as a proxy for particle size. Choose 'gaussian volume', 'raw pixels' or 'smoothed pixels'
        operation: function, the mathematical operation applied to the bin_column as a particle size proxy. Choose 'mean', 'median', 'sum' or 'max'
    
        Returns
        -------
        A dataframe with the following fields:
            'experiment' - A string with the experiment ID
            'xy_position' - Non-negative integer showing the xy position of the experiment
            'cell' - a cell ID string  compatible with the oufti cell ID
            'frame' - the time frame in the analysis (Non-negative integer)
            'particle_center' - the particle center approximated during particle segmentation (particle mask centroid)
            'max_fluorescence' - the maxmimum fluorescence of the particle estimated from the particle mask
            'particle_brightest_pixels' - the 60% of the particle pixels around the particle centroid
            'smoothed_brightest_pixels' - the 60% brightest smoothed particle pixels (after fitting a 2D Gaussian) around the particle centroid
            'gaussian_center' - the center of the particle estimated from the peak of the 2D gaussian fitted
            'gaussian_amplitude' - the amplitude of the fitted 2D gaussian
            'gaussian_std' - the std of the fitted 2D gaussian in the x and y dimension (tuple)
            'gaussian_volume' - the volume of the fitted 2D gaussian given by this function (2 π A σx σy) - https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
            'gaussian_rotation' - the rotation of the fitted 2D gaussian
            'particle_fluorescence' - the proxy of particle size estimated using a mathematical operation (parameter: operation) on a given particle metric (parameter: metric)
            'average_background_fluorescence' - the average estimated background fluorescence
        
        This dataframe is also saved in the designated folder:
            self.save_path+'/'+self.experiment+'_'+self.position_string+'_particle_df'
        
        Raises
        -----
        ValueError if the box_size is not an odd integer (for fitting the 2D Gaussian)
                   if the metric input is not valid (for the estimation of the particle fluorescence)
                   if the operation input is not valid (for the estimation of the particle fluorescence)
        """
        
        if box_size%2 == 0:
            raise ValueError('The box_size parameter is even. Choose an odd integer.')
        if metric not in ['gaussian volume', 'raw pixels', 'smoothed pixels']:
            raise ValueError("Choose 'gaussian volume', 'raw pixels', or 'smoothed pixels' as a metric")
        if operation not in ['mean', 'median', 'sum', 'max']:
             raise ValueError("Choose 'mean', 'median', 'sum' or 'max' as the operation.")
        
        # particle_data = []
        pandas_columns = ['experiment', 'xy_position','cell', 'frame', 'particle_center', 
                          'max_fluorescence', 'particle_brightest_pixels', 'smoothed_brightest_pixels', 
                          'gaussian_center', 'gaussian_amplitude', 'gaussian_std', 'gaussian_volume', 
                          'gaussian_rotation', 'particle_fluorescence', 'average_background_fluorescence']
        if analysis_range[0] == 0:
            # Initiate a pandas dataframe were all the results will stored
            particle_df = pd.DataFrame(columns=pandas_columns)
            # resume the analysis from a previous frame if it exists, otherwise start a new analysis from a non-zero frame
        elif analysis_range[0] > 0:
            if os.path.exists(self.save_path+'/'+self.experiment+'_'+self.position_string+'_particles_df') == True:
                particle_df = pd.read_pickle(self.save_path+'/'+self.experiment+'_'+self.position_string+'_particles_df', compression='zip')
            else:
                particle_df = pd.DataFrame(columns=pandas_columns)
        # Iterate through the specified frames in the stream acquisition
        for fr in range(analysis_range[0], analysis_range[1]):
            print('frame '+str(fr+1)+' out of '+str(len(self.particle_images))+', position: '+self.position_string)
            # get the signal image
            particle_image = self.particle_images[fr]
            #--------- BACKGROUND SUBTRACTION ----------#
            background_subtraction = bkgf.back_sub(particle_image, bkgf.get_inverted_mask(self.phase_image))
            bkg_cor_image = background_subtraction[0]
#           bkg_cor[bkg_cor<0]=0 ## needed to convert negative pixels to zero but not required
            mean_bkg = background_subtraction[1]
            #--------- PARTICLE SEGMENTATION ----------#
            particle_segmentation_df = ps.particle_segmentation(bkg_cor_image, log_adaptive_parameters, min_particle_size, max_particle_size, min_particle_aspect_ratio, post_processing_threshold)
            particle_segmentation_df['experiment'] = self.experiment
            
            for index, row in particle_segmentation_df.iterrows():
                particle_center = row['centroid']
                # get the particle center and the 2D Gaussian parameters
                particle_center_stats = tgf.estimate_particle_center(bkg_cor_image, particle_center, box_size, gaussian_fit_show)
                if particle_center_stats != 'none':
                    gaussian_particle_center, brightest_raw_pixels, brightest_fitted_pixels, param = particle_center_stats
                    (height, y, x, width_y, width_x, rotation) = param
                    gaussian_vol = 2*np.pi*height*width_x*width_y
                    # get the particle fluorescence metric
                    particle_vol = tgf.get_particle_fluorescence(metric, operation, brightest_raw_pixels, brightest_fitted_pixels, gaussian_vol)
                    # get the cell ID of the associated particle
                    cell_input = self.cell_contains_particle(gaussian_particle_center, bkg_cor_image, channel_offset, cell_show)
                    # organize the data into a new pandas row
                    pre_data = [self.experiment, self.position, cell_input, fr, particle_center, 
                                brightest_raw_pixels.max(), brightest_raw_pixels, brightest_fitted_pixels, 
                                gaussian_particle_center, height, (width_x, width_y), gaussian_vol, rotation, particle_vol, mean_bkg]
                    pre_df = pd.DataFrame([pre_data], columns=particle_df.columns.tolist())
                    particle_df = pd.concat([particle_df, pre_df], ignore_index=True)
                    # particle_data.append(pre_data)

        # par_df = pd.DataFrame(np.array(particle_data).transpose(), columns=pandas_columns)
        # particle_df = pd.concat([particle_df, par_df])
        # particle_df = particle_df.reindex(np.arange(particle_df.shape[0]))
        
        with open(self.save_path+'/'+self.experiment+'_'+self.position_string+'_particles_df', 'wb') as handle:
            particle_df.to_pickle(path = handle, compression='zip', protocol = pickle.HIGHEST_PROTOCOL)
        
        return particle_df
    
    
    def run_particle_tracking(self, max_radius, memory, fluorescence_bandpass, fraction_length=0.1, merged=True, cell_connect=True):
        
        particle_df = pd.read_pickle(self.save_path+'/'+self.experiment+'_'+self.position_string+'_particles_df', compression='zip')
        # print(particle_df.describe)
        curated_df = ptm.tracking_the_particles(particle_df, max_radius, memory, fluorescence_bandpass, self.interval, fraction_length, merged, cell_connect)
       
        white = np.ones((self.sensor[1], self.sensor[0]), dtype=np.float)

        plt.figure(figsize=(40,40))
        plt.imshow(white, cmap='gray', vmin=0, vmax=1)
        for trajectory in curated_df['particle_trajectory_id'].unique():
            trajectory_df = curated_df[curated_df['particle_trajectory_id']==trajectory]
            print('trajectory:', trajectory,' , with',trajectory_df.shape[0], ' out of a total of',self.n_frames, ' frames')
    #            print('trajectory:', trajectory,' , with',trajectory_df.shape[0], ' out of a total of',test.n_frames, ' frames')
            plt.plot(trajectory_df['x'], trajectory_df['y'])
        plt.show()
        
        try:
            phase_drift = imd.estimate_phase_image_drift(self.phase_image, self.phase_image_after, precision=100)
            curated_df['phase_drift_x'] = phase_drift[0][1] # the x-coordiantes of the drift 
            curated_df['phase_drift_y'] = phase_drift[0][0] # the y-coordinates of the drift
            
        except AttributeError:
            print('No phase image was taken after stream acquisition and the xy drift could not be estimated...')
            curated_df['phase_drift_x'] = np.nan
            curated_df['phase_drift_y'] = np.nan
            
        bad_trajectories = []
        print('removing trajectories with duplicated frames...')
        
        # plot the trajectories
        white = np.ones((self.sensor[1], self.sensor[0]), dtype=np.float)
        plt.figure(figsize=(40,40))
        plt.imshow(white, cmap='gray', vmin=0, vmax=1)
        for trajectory in curated_df['particle_trajectory_id'].unique():
            trajectory_df = curated_df[curated_df['particle_trajectory_id']==trajectory]
            # if a trajectory has a duplicated frame it is removed
            if trajectory_df['frame'].duplicated().any() == True:
                bad_trajectories.append(trajectory)
            elif trajectory_df['frame'].duplicated().any() == False:
                plt.plot(trajectory_df['x'], trajectory_df['y'])
        try:
            for cell in self.cells:
                plt.plot(self.cell_meshes[cell][0], self.cell_meshes[cell][1], color='black', linewidth=1)
        except AttributeError:
            print('No cells were segmented in this experiment.')
    
        plt.show()
        
        curated_df = curated_df[~curated_df['particle_trajectory_id'].isin(bad_trajectories)]
        prefix = self.experiment + '_' + self.position_string + '_'  # generate experiment and position unique particle IDs
        curated_df['particle_trajectory_id'] =  prefix + curated_df['particle_trajectory_id']
        
        with open(self.save_path+'/'+self.experiment+'_'+self.position_string+'_tracked_particles_df', 'wb') as handle:
            curated_df.to_pickle(path = handle, compression='zip', protocol = pickle.HIGHEST_PROTOCOL)
    
        return curated_df

    
    def show_all_trajectories(self, cell_based=False, save='none'):
        """
        This functions shows a 2D map with all the cell meshes and the particle trajectories
        
        Parameters
        ----------
        cell_based: bool - True if particle tracking was cell_based. Otherwise use False.
        
        Returns
        -------
        A map of all the trajectories and cell meshes. Each trajectory is assigned a different color. 
        The trajectory IDs are also displayed. 
        """
        if cell_based == False:
            curated_df = pd.read_pickle(self.save_path+'/'+self.experiment+'_'+self.position_string+'_tracked_particles_df', compression='zip')
        elif cell_based == True:
            curated_df = pd.read_pickle(self.save_path+'/'+self.experiment+'_'+self.position_string+'_tracked_particles_in_cells_df', compression='zip')
            
        x_use = 'x'
        y_use = 'y'
            
        white = np.ones((self.sensor[1], self.sensor[0]), dtype=np.float)
        plt.figure(figsize=(40,40))
        plt.imshow(white, cmap='gray', vmin=0, vmax=1)
        for trajectory in curated_df['particle_trajectory_id'].unique():
            trajectory_df = curated_df[curated_df['particle_trajectory_id']==trajectory]
            plt.plot(trajectory_df[x_use], trajectory_df[y_use])
            plt.text(trajectory_df[x_use].min()-5, trajectory_df[y_use].min()-5, trajectory, fontsize=7)
        for cell in self.cells:
            plt.plot(self.cell_meshes[cell][0], self.cell_meshes[cell][1], color='black', linewidth=1)
        if os.path.isdir(save):
            plt.savefig(save+'/'+self.experiment+'_'+self.position_string+'_all_trajectories.jpeg')
        plt.show()
        
