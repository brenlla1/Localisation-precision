# Localisation-precision

This repository contains the the files necessary to assess the experimental localisation error in superresolution microscopy

# Requirements
This software was tested using `Python 3.7.4`, `numpy 1.21.5`, `scipy 1.3.1` and `skimage 0.15.0`


# Instructions

Run the program `fit_rois_mle_ls.py` and then run command:

`get_opt_params_ls_mle(file, offset=0, sensitivity=1, WINDOW=15, a=75, EM_gain=1)` 

Where `file` is a string with the full path of _.tif_ image file containing beads, `offset` is the offset of the camera, `sensitivity` is the camera sensitivity, `WINDOW` is the size of the ROI window side in pixels, `a` is the pixel size in nm and and `EM_gain` is the real camera gain, in case camera has no gain, set to 1.

For data analysis, ensure each folder only contains one _.tif_ image file
After the fitting is done, 3 _.npz_ files with the results are created in the same folder of the _.tif_ file
Place the Jupyter notebook `comparison-experimental-theoretical-precision.ipynb` in the same folder as the data and run it.











