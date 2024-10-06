# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib as mpl
mpl.rc('image', cmap='afmhot')
mpl.rc('font', size=14)
mpl.rc('lines', linewidth=1.5)

np.set_printoptions(precision=2)

from skimage.feature import corner_peaks
from scipy.optimize import least_squares, minimize
from scipy.integrate import quad
import skimage.io as skio
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
   
def get_opt_params_ls_mle(file_name, file_num=None, offset=None, sensitivity=1, EM_gain=1, WINDOW=15, a=100):

    
    folder_name, _ = os.path.split(file_name)
    img_stack = skio.imread(file_name)   
    
    if offset==None:
        offset = img_stack.min()
        
    img_stack = img_stack - offset
    
    img_stack[img_stack<1] = 1 # ensure all data is positive because we have to take log 
    
    win_2 = WINDOW//2 
    
    pix = np.arange(-win_2, win_2+1) #defines the pixel region   
    sigma = 4
    
    n_frames = img_stack.shape[0]
    
    def detect_gauss_corr(imgs):
        img_mean = imgs.mean(axis=(1,2)) 
        img_std = np.std(imgs, axis=(1,2))
        width = (sigma/2)**0.5
        peaks = []
        for idx, img in enumerate(imgs):
            corr_img = gaussian_filter(imgs[idx], width )
            pks = corner_peaks(corr_img, min_distance=1,
                               threshold_abs=img_mean[idx]+img_std[idx]*3, exclude_border=WINDOW)
            peaks.append(pks)
        return peaks 
    
    peaks = detect_gauss_corr( img_stack[:10].mean(axis=0)[None, :, :] )
    peaks = peaks[0] # list with one element
     
    peak_distances = np.linalg.norm( peaks[:,None,:]-peaks[None,:,:], axis=-1 )
    idx_peaks_too_close = np.where( np.logical_and( peak_distances < WINDOW*1, peak_distances > 0) )[0]
    peaks = np.delete(peaks, idx_peaks_too_close, axis=0)
    n_peaks = peaks.shape[0]
    
       
    def get_rois_from_peaks(imgs, peaks):
        rois = [] #empty list
        for idx in peaks: # iterating over frames
            rois.append( imgs[:, idx[0]-win_2:idx[0]+win_2+1, idx[1]-win_2:idx[1]+win_2+1] ) 
        return rois
    
    
    rois = get_rois_from_peaks(img_stack, peaks)
    data_arr = np.stack( rois )   # shape (n_peaks, n_frames, WINDOW, WINDOW )
    
    data_arr = data_arr.astype(np.float32)
    data_arr = data_arr*sensitivity / EM_gain
    
    n_rois = n_peaks*n_frames
    data_arr_for_fit = data_arr.reshape(-1, WINDOW, WINDOW)
    
    
    # Functions for fit with
    def compute_gauss(params, pix):    
        x0, sigx, y0, sigy = params
        x_exponent = (pix - x0)**2/(2*sigx**2)
        y_exponent = (pix - y0)**2/(2*sigy**2)
        return np.outer( np.exp(-y_exponent), np.exp(-x_exponent) )             
    
    design_matrix = np.ones( (WINDOW**2, 2) )
    def compute_fit(params, data):
        """fit a 2D Gaussian to data"""    
        gauss = compute_gauss(params, pix) 
        design_matrix[:, 0] = gauss.ravel()
        coeff = nnls(design_matrix, data)[0]
        return design_matrix@coeff, coeff
    
    def compute_res(params, data):
        return data-compute_fit(params, data)[0]
     
    start = [0,4,0,4] 
    
    def fit_ls(start):
        # Get optimized params and coeffs
        opt_all, coeff_all = [], []
        for idx, roi in enumerate( data_arr_for_fit ):
            #max_position = np.argmax(roi)
            #y_max, x_max = np.unravel_index(max_position, roi.shape)
            opt = least_squares(compute_res, start, args=(roi.ravel(),), 
                                bounds=([-WINDOW+1, 0.1, -WINDOW+1, 0.1], WINDOW-1) )
            opt = opt.x
            opt_all.append( opt )
            coeff = compute_fit(opt, roi.ravel())[1]
            coeff_all.append(coeff)
            if idx%n_frames==0:
                start = [0,4,0,4]
            else:
                start = opt
        
        opt_all = np.array( opt_all )
        xy_shift = opt_all[:,[0, 2]] 
        coeff_all = np.array( coeff_all )
        
        # compute fits
        all_fits  = [ compute_fit(i,j.ravel())[0].reshape(WINDOW, WINDOW) for i,j in zip(opt_all, data_arr_for_fit) ] 
        fit_arr = np.array( all_fits )
        return opt_all, coeff_all, xy_shift, fit_arr
    
    
    def find_variance_formula(params, coeff):
        gaussian = compute_gauss(params, pix)
        N = coeff[0]*gaussian.sum()
        b2 = coeff[1]
        sigma_x = params[1]*a # convert sigma to nm
        sigma_y = params[3]*a
        sigma_x_a_squared = sigma_x**2 + a**2/12 
        sigma_y_a_squared = sigma_y**2 + a**2/12
        
        variance_x = ( sigma_x_a_squared / N) * ( 16/9 + 8 * np.pi * sigma_x_a_squared * b2 / (N * a**2) )
        variance_y = ( sigma_y_a_squared / N) * ( 16/9 + 8 * np.pi * sigma_y_a_squared * b2 / (N * a**2) )

        return variance_x, variance_y
    
    opt_all_ls, coeff_all_ls, xy_shift_ls, fit_ls_arr = fit_ls(start)
    variance_formula = [find_variance_formula(i,j) for i,j in zip(opt_all_ls, coeff_all_ls)]
    variance_x_ls, variance_y_ls = list( zip(*variance_formula) )
    variance_ls_arr = np.stack([variance_x_ls, variance_y_ls])
    
    file_LS_name = r'LS-fit.npz'
    np.savez(os.path.join(folder_name, file_LS_name), params_ls=opt_all_ls,
             coeff_ls=coeff_all_ls, all_fits_ls=fit_ls_arr, variance_ls=variance_ls_arr)
    
    
    def fit_mle(start):
        # Functions to do the fit with MLE
        def datafun(params, data):
            exp_vals = compute_fit(params, data)[0]
            return np.sum(exp_vals) - np.sum(data * np.log(exp_vals))
          
        opt_all, coeff_all = [], []   
        for start, roi in zip(opt_all_ls, data_arr_for_fit ):
            result = minimize( datafun, start, options={'maxiter': 100000}, args=roi.ravel(), 
                              bounds=( (-WINDOW+1, WINDOW-1), (0.1, WINDOW), (-WINDOW+1, WINDOW-1), (0.1, WINDOW) ) )
            opt_all.append( result.x )
            coeff = compute_fit(result.x, roi.ravel())[1]
            coeff_all.append(coeff)
    
            
            
        opt_all = np.array( opt_all )
        xy_shift = opt_all[:,[0, 2]] 
        coeff_all = np.array( coeff_all )    
            
        all_fits  = [ compute_fit(i,j.ravel())[0].reshape(WINDOW, WINDOW) for i,j in zip(opt_all, data_arr_for_fit) ] 
        fit_arr = np.array( all_fits )
        
        return opt_all, coeff_all, xy_shift, fit_arr
    
    
    opt_all_mle, coeff_all_mle, xy_shift_mle, fit_mle_arr = fit_mle(start)
    
    # calculate variance MLE
    def find_variance_mle(params, coeff):
        gaussian = compute_gauss(params, pix)
        N = coeff[0]*gaussian.sum()
        b2 = coeff[1]
        
        sigma_x = params[1]*a # convert sigma to nm
        sigma_y = params[3]*a
        sigma_x_a_squared = sigma_x**2 + a**2/12 
        sigma_y_a_squared = sigma_y**2 + a**2/12
 
        Fx = lambda t: np.log(t) / (1 + (N * a**2 * t / (2 * np.pi * sigma_x_a_squared * b2)))
        Fy = lambda t: np.log(t) / (1 + (N * a**2 * t / (2 * np.pi * sigma_y_a_squared * b2)))

        integral_x, _ = quad(Fx, 0, 1)        
        variance_x = sigma_x_a_squared / N * (1 + integral_x)**-1
        integral_y, _ = quad(Fy, 0, 1)        
        variance_y = sigma_y_a_squared / N * (1 + integral_y)**-1
        
        return variance_x, variance_y
      
    variance_mle = [find_variance_mle(i,j) for i,j in zip(opt_all_mle, coeff_all_mle)]
    variance_x_mle, variance_y_mle = list( zip(*variance_mle) )
    variance_mle = np.stack([variance_x_mle, variance_y_mle])
    
    # save the optimized parameters, coeffs, and fits
    file_MLE_name = r'MLE-fit.npz' 
    np.savez(os.path.join(folder_name, file_MLE_name), params_mle=opt_all_mle,
             coeff_mle=coeff_all_mle, all_fits_mle=fit_mle_arr, variance_mle=variance_mle)
       
    file_rois_name = r'rois.npz'
    np.savez(os.path.join(folder_name, file_rois_name), data_arr=data_arr, peaks=peaks )
    
    
    
    
    
    
    
    
    
    
