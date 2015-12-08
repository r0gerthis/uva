# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#           NAME: week1.py
#         AUTHOR: Stratis Gavves
#  LAST MODIFIED: 18/03/10
#    DESCRIPTION: TODO
#
#------------------------------------------------------------------------------
import numpy as np
import urllib
import os
import sys
import math
import scipy.signal as sgn
import matplotlib.pyplot as plt
from scipy import misc

def extractColorHistogram( im ):
    # PRE [DO NOT TOUCH]
    histo = []
    
    # WRITE YOUR CODE HERE
    im_r = im[:, :, 0].flatten()
    histo_r = np.bincount(im_r, None, 256)
    im_g = im[:, :, 1].flatten()
    histo_g = np.bincount(im_g, None, 256)
    im_b = im[:, :, 2].flatten()
    histo_b = np.bincount(im_b, None, 256)
    
    # RETURN [DO NOT TOUCH]
    histo = np.concatenate([histo_r, histo_g, histo_b])
    return histo

def computeVectorDistance( vec1, vec2, dist_type ):
    # PRE [DO NOT TOUCH]
    dist = 0

    # WRITE YOUR CODE HERE
    
    # normalize histograms
    # l2 normalisation for l2/eucl
    #if dist_type == 'euclidean' or dist_type == 'l2': 
        #vec1 = vec1/np.sqrt(sum(vec1**2))
        #vec2 = vec2/np.sqrt(sum(vec2**2))
        
    # l1 normalisation for intersect, hellinger & chi2
    #if dist_type == 'intersect' or dist_type == 'l1' or dist_type == 'hellinger' or dist_type == 'chi2':
        #vec1 = vec1/sum(vec1)
        #vec2 = vec2/sum(vec2)
    
    # smaller
    if dist_type == 'euclidean':
        dist = sum((vec1-vec2)**2)
        
    # larger
    elif dist_type == 'l2':
        dist = sum(vec1*vec2)
    
    # larger
    elif dist_type == 'intersect' or dist_type == 'l1':
        dist = sum(np.minimum(vec1, vec2))
        
    # smaller
    elif dist_type == 'chi2':
        dist = sum(((vec1-vec2)**2)/(vec1+vec2))

    # larger
    elif dist_type == 'hellinger':
        dist = sum(np.sqrt(vec1)*np.sqrt(vec2))

    # RETURN [DO NOT TOUCH]
    return dist
    
def computeImageDistances( images, dist_type ):
    # added dist_type
    # PRE [DO NOT TOUCH]
    imdists = []
    
    # WRITE YOUR CODE HERE
    histo = []
    for img in images:
        histo.append(extractColorHistogram(plt.imread(img)))
    
    # setup matrix for comparison outcomes. 0,0 = img1,img1, 0,1=img1,img2
    imdists = np.zeros((len(histo), len(histo)))
    # setup check matrix to keep track of comparisons already done. 
    # set to 1 if done - makes code more efficient & faster
    imcheck = np.zeros((len(histo), len(histo)))

    # to keep track where we are in the matrix
    x = -1
    y = -1
    
    for imd1 in histo:
        x += 1
        y = -1
        for imd2 in histo:
            imdiststorage = 0
            y += 1
            # check if we haven't compared these images before
            if imcheck[x,y] == 0 and imcheck[y,x] == 0:
                # mark as checked in the check matrix
                imcheck[x,y] = 1
                imcheck[y,x] = 1
                # get distance and store in var to make code faster
                imdiststorage = computeVectorDistance(imd1, imd2, dist_type)
                # store distance in matrix
                imdists[x,y] = imdiststorage
                imdists[y,x] = imdiststorage
                
    # RETURN [DO NOT TOUCH]
    return imdists
    
def rankImages( imdists, query_id, dist_type ):
    # PRE [DO NOT TOUCH]
    ranking = []

    # WRITE YOUR CODE HERE
    related_img = []
    related_img = imdists[query_id,:]
    
    # smaller, order asc
    if dist_type == 'euclidean':
        ranking = np.argsort(related_img)
        
    # larger, order desc
    elif dist_type == 'l2':
        ranking = np.argsort(-related_img)
    
    # larger, order desc
    elif dist_type == 'intersect' or dist_type == 'l1':
        ranking = np.argsort(-related_img)
        
    # smaller, order asc
    elif dist_type == 'chi2':
        ranking = np.argsort(related_img)

    # larger, order desc
    elif dist_type == 'hellinger':
        ranking = np.argsort(-related_img)
    
    
    # RETURN [DO NOT TOUCH]
    return ranking

def get_gaussian_filter(sigma):
    # PRE [DO NOT TOUCH]
    sigma = float(sigma)
    G = []
    
    # WRITE YOUR CODE HERE FOR DEFINING THE HALF SIZE OF THE FILTER
    half_size = 3*sigma
    #
    x = np.arange(-half_size, half_size + 1)

    # WRITE YOUR CODE HERE
    
    G = ( 1/ (sigma*np.sqrt(2*math.pi))) * np.exp(-( (x**2) / (2*sigma**2) ) )
                        
    # RETURN [DO NOT TOUCH]
    G = G / sum(G) # It is important to normalize with the total sum of G
    return G
    
def get_gaussian_der_filter(sigma, order):
    # PRE [DO NOT TOUCH]
    sigma = float(sigma)
    dG = []
    
    # WRITE YOUR CODE HERE
    # half_size = ...
    #
    half_size = 3*sigma
    x = np.arange(-half_size, half_size + 1)
    
    if order == 1:
        # WRITE YOUR CODE HERE
        dG = (x/(sigma**2)*get_gaussian_filter(sigma))
    # elif order == 2:
        # WRITE YOUR CODE HERE
        # dG = ...

    # RETURN [DO NOT TOUCH]
    return dG

def gradmag(im_dr, im_dc):
    # PRE [DO NOT TOUCH]
    im_dmag = []

    # WRITE YOUR CODE HERE
    im_dmag = np.sqrt(im_dr**2 + im_dc**2)
    #

    # RETURN [DO NOT TOUCH]
    return im_dmag    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [ALREADY IMPLEMENTED. DO NOT TOUCH]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def apply_filter(im, myfilter, dim):
    H, W = im.shape
    if dim == 'col':
        im_filt = sgn.convolve(im.flatten(), myfilter, 'same')
        im_filt = np.reshape(im_filt, [H, W])
    elif dim == 'row':
        im_filt = sgn.convolve(im.T.flatten(), myfilter, 'same')
        im_filt = np.reshape(im_filt, [W, H]).T
    
    return im_filt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [ALREADY IMPLEMENTED. DO NOT TOUCH]    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def apply_gaussian_conv(im, G):
    im_gfilt = apply_filter(im, G, 'col')
    im_gfilt = apply_filter(im_gfilt, G, 'row')
    
    return im_gfilt


        
        
    
