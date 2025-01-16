# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:37:36 2025

@author: Alexandros Papagiannakis, HHMI at Stanford University, 2025
"""

import numpy as np

def removing_merging_trajectories(curated_df):
    """
    This function is used to remove those particle displacements that merge to a common position.
    The joint between the two trajectories into their future common trajectory is removed from the analysis;
    
    This function accepts the particle tracking dataframe as an input and returns the updated dataframe with the merged trajectories removed
    """
    # removing trajectories that merge to a common particle position
    print('Removing trajectories which merge to a common particle position...')
    clean_df = curated_df[~curated_df['particle_linkage'].isnull()]   # remove the nan values
    merged_trajectories = clean_df[clean_df['particle_linkage'].duplicated()]['particle_linkage'].values
    curated_df = curated_df[~curated_df['particle_linkage'].isin(merged_trajectories)]
    print(len(merged_trajectories), 'merged trajectories were removed.')
    
    return curated_df

def connecting_trajectories(curated_df, max_radius, cell, fluorescence_bandpass, n_frames):
    """
    This function can be used to merge trajectories that belong to the same cell and do not overlap.
    It is based on the hypothesis that if two trajectories belong to the same cell and do not overlap not even for 
    a single frame, then they are most possibly not belong to the same trajectory.
    This fucntion applies only in particles that belong to cell meshes.
    
    Parameters
    ----------
    curated_df: pandas dataframe returned after tracking
    max_radius: positive integer in pixels - the max distance between the last frame of the first
                and the first frame of the second trajectory to be linked
    cell: string - cell ID
    
    Returns
    -------
    The updated dataframe with the merged trajectories
    """
    
    print('connecting trajectories in cells')
    particles = list(curated_df[curated_df['cell']==cell]['particle_trajectory_id'].unique())
    number_of_particles = len(particles)
    
    if number_of_particles > 1: # check if the cell has more than 1 particles
        
        print('cell', cell, 'has', number_of_particles, 'detected particles.')
        
        while curated_df[curated_df['cell']==cell]['particle_trajectory_id'].unique().shape[0] > 1:  # keep connecting trajectories until there is only one left
#                    print(curated_df[curated_df['cell']==cell]['particle_trajectory_id'].unique().shape[0])
            cell_df = curated_df[curated_df['cell']==cell] # get the dataframe specific for the cell of interest
            
            particles = list(curated_df[curated_df['cell']==cell]['particle_trajectory_id'].unique())  # get a list of all the particle trajectories in the cell
            number_of_particles = len(particles) # get the number of particle trajectories in the cell
            
            particle_merging_dictionary = {} # initialize a dictionary with the connected trajectories as a key (traj_1, traj_2) and the length of the two trajectories combined as a value (integer)
            
            for par_1 in range(number_of_particles):  # iterate over the particles in the list 
                for par_2 in range(par_1+1, number_of_particles): # second iteration to make all possible combinations
                    
                    par_1_df = cell_df[cell_df['particle_trajectory_id']==particles[par_1]]
                    par_2_df = cell_df[cell_df['particle_trajectory_id']==particles[par_2]]
                    
                    particle_1_frames = par_1_df.frame.to_list() # get a list with the frames of the first trajectory in the pair
                    particle_2_frames = par_2_df.frame.to_list() # get a list with the frames of the second trajectory in the pair
                    particle_sum_frames = len(particle_1_frames)+len(particle_2_frames) # get a list with the combined length of the two trajectories
                    
                    if (bool(set(particle_1_frames) & set(particle_2_frames)) == False and particle_sum_frames <= n_frames): # if the frame ranges of the particles particles do not overlap 
                        # and the total frame covered by both is not longer than the number of frames in the stream acquisition
                        # estimate the distance between the last frame of the earlier trajectory and the first frame of the latter one
                        # this distance is named "closest_distance"
                        particle_1_max_frame = np.max(particle_1_frames) # get the max frame of par_1
                        particle_2_max_frame = np.max(particle_2_frames) # get the max frame of par 2
                        if particle_1_max_frame > particle_2_max_frame: # if the max frame of par 1 is bigger then this trajectory follows 
                            last_x = par_2_df[par_2_df.frame==particle_2_max_frame].x.values[0]
                            last_y = par_2_df[par_2_df.frame==particle_2_max_frame].y.values[0]
                            next_x = par_1_df[par_1_df.frame==np.min(par_1_df.frame)].x.values[0]
                            next_y = par_1_df[par_1_df.frame==np.min(par_1_df.frame)].y.values[0]
                            closest_distance = np.sqrt((next_x-last_x)**2 + (next_y-last_y)**2)
                        elif particle_1_max_frame < particle_2_max_frame: # if the max frame of par 2 is bigger then this trajectory follows 
                            last_x = par_1_df[par_1_df.frame==particle_1_max_frame].x.values[0]
                            last_y = par_1_df[par_1_df.frame==particle_1_max_frame].y.values[0]
                            next_x = par_2_df[par_2_df.frame==np.min(par_2_df.frame)].x.values[0]
                            next_y = par_2_df[par_2_df.frame==np.min(par_2_df.frame)].y.values[0]
                            closest_distance = np.sqrt((next_x-last_x)**2 + (next_y-last_y)**2)
                        elif particle_1_max_frame == particle_2_max_frame: # if the trajectories share a frame then they cannot be linked
                            # This is achieved by setting the closest_distance higher than the max_radius (safety condition)
                            closest_distance = max_radius + 10
                        
                        average_particle_1_intensity = par_1_df.particle_fluorescence.mean()
                        average_particle_2_intensity = par_2_df.particle_fluorescence.mean()
 
                        if closest_distance <= max_radius: # setting a max radius and fluorescence radius requirement 
                            if ((average_particle_1_intensity/average_particle_2_intensity) > fluorescence_bandpass[0] and (average_particle_1_intensity/average_particle_2_intensity) < fluorescence_bandpass[1]): # setting a relaxed max fluorescence intensity requirement
                                particle_merging_dictionary[(particles[par_1], particles[par_2])] = particle_sum_frames # the two particles will then be merged and thus are included in the merging dataframe
            
            if len(particle_merging_dictionary.keys()) == 0: # if no trajectories are to be merged, terminated the loop
                print('These are two particles and cannot be merged...')
                break
            
            elif len(particle_merging_dictionary.keys()) > 0: # if there are particle trajectories to be merged, first merge the two trajectories that give the longest trajectory combined
                max_key = max(particle_merging_dictionary, key=particle_merging_dictionary.get) 
                print(max_key, 'particles are merged...')
                # print(curated_df)
                curated_df.particle_trajectory_id.replace(max_key[1], max_key[0], inplace=True)
                # repeat the loop to look to other pairs after merging of the first pair.
    return curated_df


def removing_spurious_trajectories(curated_df, fraction_length, n_frames):
    """
    This fucntion can be used to remove spurious trajectories below a specified length.
    The user specifies the dataframe with the tracked particles and the trajectory minimum lenght
    as a fraction of the enire stream acquisition (self.n_frames).
    
    Parameters
    ----------
    curated_df: pandas dataframe which includes the tracked particles
    fraction_length: float - the trajectory length as a fraction of the entire stream acquisition (e.g. 0.1 for 10%)
    
    Returns
    -------
    The updated particle tracking dataframe with the spurious trajectories removed
    """
    
    print('removing trajectories which are shorter than', fraction_length*n_frames, 'frames...')
    min_length = fraction_length*n_frames
    # get the size of each trajectory length
    size_df = curated_df.groupby(['particle_trajectory_id']).size()
    # select the trajectory indexes that are above or equal to the specified threshold and generate a list with the partcle IDs
    long_trajectories = size_df[size_df.values>= min_length].index.tolist()
    short_trajectories = size_df[size_df.values< min_length].index.tolist()
    curated_df = curated_df[curated_df['particle_trajectory_id'].isin(long_trajectories)]
    
    print(len(short_trajectories), 'short trajectories were removed.')
    
    return curated_df


def tracking_the_particles(particle_df, max_radius, memory, fluorescence_bandpass, interval, fraction_length, merged=True, cell_connect=True):
    """
    This code is used to track the segmented particles. 
    It also removes the spurious trajectories, links short trajectories that belong to the same cell, removes particle trajectories that merge to the same position and correctes for the average x/y drift.
    It opens the particle segmentation dataframe saved from the 'getting_the_particles_function' 
    (self.save_path+'/'+self.experiment+'_'+self.position_string+'_particles_df')
    
    Particle tracking:
        particle trascking is performed based on the distance of the particles in the next frame (max_radius)
        ...the fluorescence of the particles in the next frame (fluorescence_bandpass)
        ...searching over a range of frames downstream if a linked particle is not found in the next frame (memory)
    Removing merging trajectories:
        if 'merged' is True, the frames when two or more different particles merge to the same future position are removed from the analysis
    Removing spurious trajectories:
        if 'fraction_length' is above zero, trajectories below a certain length are removed from the analysis. 
        The minimum length of a trajectory is selected by the user as the fraction of the entire stream acquisitio ('fraction_length')
    Connecting trajectories in cells:
        if 'cell_connect' is True, the particle that belong in the same cell, and their frame do not overlap not even for a single frame are connnected to a single trajectory. 
        First the longest trajectories are connected, and then shorter ones.
        The last frame of the first trajectory and the first frame of the second trajectory have to be within the specified max radius
            and their fluorescence should match within the fluorescence bandpass in order to be linked
    
    Parameters
    ----------
    max_radius: integer - the max radius within which the software is looking for the next particle position in pixels
    memory: integer - the number of frames that the algorithm waits for the particles to reappear if it does not find a nearest neighbor within the specified radius in the next frame.
    fluorescence_bandpass: tuple of floats - the minimum and maximum ratio of the fluorescence of the particle between its previous and next potential position. 
        This is used to ensure that each particle is linked to particles in downstream frames with similar fluorescence.
    fraction_length: float - the length of the minimum trajectory as a fraction of the entire stream acquisition.
        This is used to remove spurious trajectories below the specified length.
    merged: bool - if True the algorithm removes points were two trajectories merge into one
    cell_connect: bool - if True the algorithm links trajectories in the same cell that do not share not a single frame in time.
        This function requires a 'cell' column

    Returns
    -------
    The particle dataframe with four extra columns: 
        'particle_linkage' - tje linkage between indexes between subsequent frames in the same trajectory
        'particle_trajectory_id' - the unique IDs of the particle trajectories
        'phase_drift_x' - the subpixel drift of the phase images between and after stream acquisition in the x dimension (in pixels)
        'phase_drift_y' - the subpixel drift of the phase images between and after stream acquisition in the y dimension (in pixels)
            If a phase image is not captured after stream acquisition 'NaN' is returned
    
    Exception
    ---------
    If not phase image was acquired after stream acquisition the 
    'phase_drift_x' and 'phase_drift_y' are set to np.nan
    """
    # suggested parameters
#        max_radius = 20
#        memory = 15
#        fraction_length = 0.05
#        fluorescence_bandpass = (0.7,1.3)
#        merged=True
#        cell_connect=True
#        fraction_length = 0.1
    
    # get the x and y coordinates of the particle centers to estimate the 2D MSD
    particle_df['x'] = particle_df['gaussian_center'].apply(lambda x: x[0])
    particle_df['y'] = particle_df['gaussian_center'].apply(lambda x: x[1])
    # get the time using the time interval variable in msec
    particle_df['t'] = particle_df['frame']*interval
#        particle_df['t'] = particle_df['frame']*test.interval
          
    particle_linkage_dictionary = {}
    particle_trajectories_dictionary = {}
    
    # iterate through the particles regardless of their position
    for particle_index, particle_row in particle_df.iterrows():
        
        particle_x = particle_row['x']
        particle_y = particle_row['y']
        particle_fluor = particle_row['particle_fluorescence']
        particle_frame = particle_row['frame']
        particle_cell = particle_row['cell']   # include the cell in the tracking
        
        # as long as the particles belong to the frame before the last
        n_frames = particle_df.frame.max()+1
        if particle_frame < n_frames - 1:
        # if particle_frame < test.n_frames - 1:
            # create a temporary dataframe which does not include the particle selected during the iteration
            temporary_dataframe = particle_df.copy()
            temporary_dataframe = temporary_dataframe.drop(index=particle_index) # remove the investigated particle index to look for linked particles
            
            temporary_dataframe['distance'] = np.sqrt((temporary_dataframe['x']-particle_x)**2 + (temporary_dataframe['y']-particle_y)**2) # get the distance to all the particles
            temporary_dataframe['fluorescence_diff'] = (temporary_dataframe['particle_fluorescence']-particle_fluor).abs() # get the difference in fluorescence to all the particles
            temporary_dataframe['fluorescence_ratio'] = (temporary_dataframe['particle_fluorescence'] / particle_fluor) # get the fluorescence ratio between the investigated and the rest of the particles
            # temporary_dataframe['distance_fluorescence_metric'] = temporary_dataframe['fluorescence_diff'] + temporary_dataframe['distance']
            temporary_dataframe['distance_fluorescence_metric'] = temporary_dataframe['fluorescence_diff']/temporary_dataframe['fluorescence_diff'].mean() + 2*temporary_dataframe['distance']/temporary_dataframe['distance'].mean()
            # Increasing the contribution of the distance metric to the distance_fluorescence_metric
            # Also, the distance and fluorescence difference measures were normalized by the mean difference per feature.
            
            minimum_distance_df = temporary_dataframe[temporary_dataframe['distance']<= max_radius] # select those particles that are within the specified radius from the investigated particle
            next_frame = particle_frame + 1 # look in the next frames for the linked particle
            while (next_frame <= particle_frame + memory and next_frame < n_frames):    # the range of frames to look into is defined by the memory parameter
                minimum_distance_frame_df = minimum_distance_df[minimum_distance_df['frame']==next_frame]
                minimum_distance_frame_df = minimum_distance_frame_df[minimum_distance_frame_df['fluorescence_ratio'].between(*fluorescence_bandpass, inclusive='neither')] # select those particles that have similar fluorescence
                if  minimum_distance_frame_df.shape[0] == 1: # if there is only one particle with similar fluorescence
                    particle_linkage_dictionary[particle_index] = minimum_distance_frame_df.index.values[0]
                    linked_frame = next_frame
                    next_frame = particle_frame + memory + 1 # stop the while loop
                elif minimum_distance_frame_df.shape[0] > 1: # if there are more than one particles with similar fluorescence
                    if particle_cell != np.nan:
                        cell_df = minimum_distance_frame_df[minimum_distance_frame_df['cell']==particle_cell]
                        if cell_df.shape[0] > 0:
                            particle_linkage_dictionary[particle_index] = cell_df[cell_df['distance']==cell_df['distance'].min()].index.values[0]
                        elif cell_df.shape[0] == 0:
                            particle_linkage_dictionary[particle_index] = minimum_distance_frame_df[minimum_distance_frame_df['distance']==minimum_distance_frame_df['distance'].min()].index.values[0]
                    linked_frame = next_frame
                    next_frame = particle_frame + memory + 1 # stop the while loop
                elif minimum_distance_frame_df.shape[0] == 0:
                    next_frame += 1
            
            if particle_index not in particle_linkage_dictionary:  # if no particle was found to be linked
                particle_linkage_dictionary[particle_index] = np.nan # if the particle is not linked then the linkage dictionary value is np.nan
                print('particle index:', particle_index, ', in frame:', particle_frame,', was not linked to a particle')
                if particle_index not in particle_trajectories_dictionary: # if the particle index does not belong to a trajectory
                    particle_trajectories_dictionary[particle_index] = 'particle_'+str(particle_index) # and a new particle ID is assigned 
            
            elif particle_index in particle_linkage_dictionary: # if the particle was linked to a particle downstream
                print('particle index:', particle_index, ', in frame:', particle_frame,', linked to particle index:', particle_linkage_dictionary[particle_index], ', in frame:', linked_frame)
                if particle_index in particle_trajectories_dictionary: # if particle already belongs to a trajectory
                    particle_trajectories_dictionary[particle_linkage_dictionary[particle_index]] = particle_trajectories_dictionary[particle_index] # link the new particle in the same trajectory
                elif particle_index not in particle_trajectories_dictionary: # if particle does not belong to a trajectory
                    particle_trajectories_dictionary[particle_index] = 'particle_'+str(particle_index) # link the particle to a new trajectory ID
                    particle_trajectories_dictionary[particle_linkage_dictionary[particle_index]] = particle_trajectories_dictionary[particle_index] # and its linked particle to the same trajectory ID
        
        elif particle_frame == n_frames -1: 
            particle_linkage_dictionary[particle_index] = np.nan
            if particle_index not in particle_trajectories_dictionary:
                particle_trajectories_dictionary[particle_index] = 'particle_'+str(particle_index)
                # if the particle in the last frmame does not belong to a trajectory a new particle entry is created.
    
    particle_df['particle_linkage'] = particle_df.index.map(particle_linkage_dictionary)
    particle_df['particle_trajectory_id'] = particle_df.index.map(particle_trajectories_dictionary)
    
    curated_df = particle_df.copy()
    # creating a copy of the particle tracking dataframe
    
    if fraction_length > 0: # can be also performed before the "remobing merging trajectories" function
        # using the removing_spurious_trajectories function
        curated_df = removing_spurious_trajectories(curated_df, fraction_length, n_frames)
    
    if merged == True:
        # removing the merging trajectories
        curated_df = removing_merging_trajectories(curated_df)
    
    if cell_connect == True: 
        # using the connecting_trajectories function
        print('Merging particle trajectories that belong to the same cell and their frames do not overlap...') 
        cells = curated_df['cell'].unique()
        for cell in cells:
            if cell != np.nan:
                curated_df = connecting_trajectories(curated_df, max_radius, cell, fluorescence_bandpass, n_frames)
    
    return curated_df
    
