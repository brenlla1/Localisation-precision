# Localisation-precision

This repository contains the the files necessary to assess the real experimental error in your superresolution  system

# Instructions

Run the program `fit_rois_mle_ls.py` and then run command:

`get_opt_params_ls_mle(file, offset=2, sensitivity=1, WINDOW=15, a=75, EM_gain=1)` 

Where `file` is a string, offset is the offset of the camera, `sesitivity `is the camera sensitivity, `WINDOW` is the size of the ROI window side in pixels, `a` is the pixel size in nm and and `EM_gain` is the real camera gain, in case camera has no gain set to 1.


