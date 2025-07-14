#sptPy

Author: Alexandros Papagianakis, HHMI at Stanford University, 2025

<br> This repository includes functions for single particle segmentation and tracking. Thse functions are implemented via the particle_tracking class which contains a number of attributes that can be used to segment, track and visualize single particles in cells. This particle_tracking class is implemented directly on .nd2 microscopy files, separate for each phase_contrast or fluorescent channel. The cell segmentation labels correspond to a .tif image, where the background pixels have an integer value of 0, and each cell mask has a unique integer label equal or higher than 1.

<br> An example of using the particle_tracking class is provided in the particle_tracking_example.ipynb notebook.

<br> comming soon... a second class, which inherits from the particle_tracking class, will be used to project the segmented particle trajectories on relative cell coordinates.

<br> The LoG/adaptive filter and the associated object segmentation method been used in the following paper:
<br> https://www.biorxiv.org/content/10.1101/2024.10.08.617237v2.full
<br> DNA/polysome phase separation and cell width confinement couple nucleoid segregation to cell growth in Escherichia coli
<br> Alexandros Papagiannakis, Qiwei Yu, Sander K. Govers, Wei-Hsiang Lin, Ned S. Wingreen, Christine Jacobs-Wagner
<br> doi: https://doi.org/10.1101/2024.10.08.617237
