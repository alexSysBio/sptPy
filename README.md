Author: Alexandros Papagianakis, HHMI at Stanford University, 2025

<br> This repository includes functions for single particle segmentation and tracking. Thse functions are implemented via the particle_tracking class which contains a number of attributes that can be used to segment, track and visualize single particles in cells. This particle_tracking class is implemented directly on .nd2 microscopy files, separate for each phase_contrast or fluorescent channel. The cell segmentation labels correspond to a .tif image, where the background pixels have an integer value of 0, and each cell mask has a unique integer label equal or higher than 1.

<br> An example of using the particle_tracking class is provided in the particle_tracking_example.ipynb notebook.

<br> comming soon... a second class, which inherits from the particle_tracking class, will be used to project the segmented particle trajectories on relative cell coordinates.
