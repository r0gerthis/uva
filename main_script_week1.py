# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import scipy.signal as sgn
from PIL import Image
from scipy.ndimage import filters

import sys
sys.path.insert(0, '../')
import tools
import week1
import math
reload(week1)
# PREFERENCES FOR DISPLAYING ARRAYS. FEEL FREE TO CHANGE THE VALUES TO YOUR LIKING
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step A. Download images [Already implemented]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step B. Basic image operations [Already implemented]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# B.1 Read image
im = array(imread('../../data/objects/flower/1.jpg'))
print im
# B.2 Show image
imshow(im)
axis('off')

# B.3 Get image size
H, W, C = im.shape    # H for height, W for width, C for number of color channels
print 'Height: ' + str(H) + ', Width: ' + str(W) + ', Channels: ' + str(C)

# B.4 Access image pixel
print im[1000, 600, 0]    # Single value in the 2 color dimension. Remember, numbering start from 0 (thus 1 means "2")
print im[1000, 600]       # Vector of RGB values in all 3 color dimensions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step C. Compute image histograms [You should implement]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Compute color histogram from channel R
# C.1 Vectorize first the array
im_r = im[:, :, 0].flatten()

# C.2 Compute histogram from channel R using the bincount command, as indicated in the handout
histo_r = np.bincount(im_r, None, 256)

# C.3 Compute now the histograms from the other channels, that is G and B
im_g = im[:, :, 1].flatten()
histo_g = np.bincount(im_g, None, 256)
im_b = im[:, :, 2].flatten()
histo_b = np.bincount(im_b, None, 256)

# C.4 Concatenate histograms from R, G, B one below the other into a single histogram
histo = np.concatenate([histo_r, histo_g, histo_b])

print histo


histo_opdr = week1.extractColorHistogram(im)
print histo_opdr
matplotlib.pyplot.bar(range(0, len(histo_opdr)), histo_opdr, 0.8, None, None, edgecolor='red')


######
# C.5 PUT YOUR CODE INTO THE FUNCTION extractColorHistogram( im ) IN week1.py
######

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step D. Compute distances between vectors [You should implement]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
reload(week1)
# D.1 Open images and extract their RGB histograms
im1 = imread('../../data/objects/flower/1.jpg')
histo1 = week1.extractColorHistogram(im1)
print im1[:,:,0].flatten()
im2 = imread('../../data/objects/flower/3.jpg')
histo2 = week1.extractColorHistogram(im2)

# D.2 Compute euclidean distance: d=Σ(x-y)^2 
# Note: the ***smaller*** the value, the more similar the histograms
dist_euc = 0.0
dist_euc = np.sqrt(week1.computeVectorDistance(histo1, histo2, 'euclidean'))
print dist_euc
for i in range(len(histo1)):
    dist_euc += pow((histo1[i]-histo2[i]), 2)

print dist_euc

# D.3 Compute histogram intersection distance: d=Σmin(x, y)
# Note: the ***larger*** the value, the more similar the histograms
# dist_hi = ...# WRITE YOUR CODE HERE
dist_hi = 0
for i in range(len(histo1)):
    dist_hi += min(histo1[i], histo2[i])

print dist_hi
# D.4 Compute chi-2 similarity: d= Σ(x-y)^2 / (x+y)
# Note: the ***smaller*** the value, the more similar the histograms
# dist_chi2 = ...# WRITE YOUR CODE HERE

dist_chi2 = 0

for i in range(len(histo1)):
    dist_chi2 += (pow(histo1[i] - histo2[i], 2) / (histo1[i]+histo2[i]))

print dist_chi2
# D.5 Compute hellinger distance: d= Σsqrt(x*y)
# Note: the ***larger*** the value, the more similar the histograms
# dist_hell = ...# WRITE YOUR CODE HERE
dist_hell  = 0

for i in range(len(histo1)):
    dist_hell += sqrt(histo1[i]*histo2[i])
        
print dist_hell
######
# D.6 PUT YOUR CODE INTO THE FUNCTION computeVectorDistance( vec1, vec2, dist_type ) IN week1.py
######

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step E. Rank images [You should implement]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# E.1 Compute histograms for all images in the dataset
impaths = tools.getImagePathsFromObjectsDataset('flower') # [ALREADY IMPLEMENTED]
impaths = ['../../data/objects/flower/1.jpg', '../../data/objects/flower/2.jpg', '../../data/objects/flower/3.jpg', '../../data/objects/flower/4.jpg', '../../data/objects/flower/5.jpg', '../../data/objects/flower/6.jpg']
print impaths
# histo = ...# WRITE YOUR CODE HERE
histo = []
for imh in impaths:
    print "extr histogram frm: %s" %  imh
    histo.append(week1.extractColorHistogram(imread(imh)))
# E.2 Compute distances between all images in the dataset
# imdists = ... # WRITE YOUR CODE HERE

# setup matrix for comparison outcomes. 0,0 = img1,img1, 0,1=img1,img2
imdists = np.zeros((len(histo), len(histo)))
# setup check matrix to keep track of comparisons already done. set to 1 if done
imcheck = np.zeros((len(histo), len(histo)))
imdiststorage = 0
x = -1

for imd1 in histo:
    x += 1
    y = -1
    for imd2 in histo:
        imdiststorage = 0
        y += 1
        if imcheck[x,y] == 0 and imcheck[y,x] == 0:
            # new image!
            imcheck[x,y] = 1
            imcheck[y,x] = 1
            print 'com img - new ', x ,y
            imdiststorage = week1.computeVectorDistance(imd1, imd2, 'l2')
            imdists[x,y] = imdiststorage
            imdists[y,x] = imdiststorage





# E.3 Given an image, rank all other images
query_id = rnd.randint(0, 59) # get a random image for a query
# sorted_id = ... # Here you should sort the images according to how distant they are
query_id = 0
related_img = imdists[query_id,:]

sorted_id = np.argsort(related_img)

reload(week1)

vec1 = array([0,0,3,0])
vec2 = array([0,30,100,0])

vec1 = vec1/np.sqrt(sum(vec1**2))
vec2 = vec2/np.sqrt(sum(vec2**2))

print vec1, vec2


query_id = 4
imdists = week1.computeImageDistances(impaths, 'hellinger')
sorted_id = week1.rankImages(imdists, query_id, 'hellinger')
# E.4 Showing results. First image is the query, the rest are the top-5 most similar images [ALREADY IMPLEMENTED]
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)
im = imread(impaths[query_id])
ax.imshow(im)
ax.axis('off')
ax.set_title('Query image')

for i in np.arange(1, 1+5):
    ax = fig.add_subplot(2, 3, i+1)
    im = imread(impaths[sorted_id[i-1]]) # The 0th image is the query itself
    ax.imshow(im)
    ax.axis('off')
    ax.set_title(impaths[sorted_id[i-1]])

######
# E.5 PUT YOUR CODE INTO THE FUNCTIONS computeImageDistances( images )
#     AND rankImages( imdists, query_id ) IN week1.py
######

# F. Gaussian blurring using gaussian filter for convolution
reload(week1)
# F.1 Open an image
im = array(Image.open('../../data/objects/flower/1.jpg').convert('L'))
imshow(im, cmap='gray') # To show as grayscale image

# F.2 Compute gaussian filter
sigma = -10.0
G = week1.get_gaussian_filter(sigma) # WRITE YOUR CODE HERE

# F.3 Apply gaussian convolution filter to the image. See the result. Compare with Python functionality
im_gf = week1.apply_gaussian_conv(im, G) # [ALREADY IMPLEMENTED, YOU ONLY NEED TO INPUT YOUR GAUSSIAN FILTER G]
im_gf2 = filters.gaussian_filter(im, sigma) # The result using Python functionality

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(im_gf, cmap='gray')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(im_gf2, cmap='gray')

# F.4 Compute first order gaussian derivative filter in one dimension, row or column
reload(week1)
dG = week1.get_gaussian_der_filter(10, 1)

# Apply first on the row dimension
im_drow = week1.apply_filter(im, dG, 'row') # [ALREADY IMPLEMENTED, YOU ONLY NEED TO INPUT YOUR GAUSSIAN DERIVATIVE dG YOU JUST IMPLEMENTED]
# Apply then on the column dimension
im_dcol = week1.apply_filter(im, dG, 'col') # [ALREADY IMPLEMENTED, YOU ONLY NEED TO INPUT YOUR GAUSSIAN DERIVATIVE dG YOU JUST IMPLEMENTED]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.imshow(im_drow, cmap='gray')
ax = fig.add_subplot(1, 2, 2)
ax.imshow(im_dcol, cmap='gray')

# F.6 Compute the magnitude and the orientation of the gradients of an image
im_dmag = week1.gradmag(im_drow, im_dcol)

fig = plt.figure()
imshow(im_dmag, cmap='gray')

######
# F.6.1 PUT YOUR CODE INTO THE FUNCTIONS get_gaussian_filter(sigma),
#       get_gaussian_der_filter(sigma, order) AND gradmag(im_drow, im_dcol) IN week1.py
######

# F.7 Apply gaussian filters on impulse image. HERE YOU JUST NEED TO USE THE CODE
#     YOU HAVE ALREADY IMPLEMENTED
reload(week1)
# F.7.1 Create impulse image
imp = np.zeros([15, 15])
imp[6, 6] = 1
imshow(imp, cmap='gray')

H = 150
imp = np.zeros([H, H])
imp[round(H/2), round(H/2)] = 1
imshow(imp, cmap='gray')

# F.7.1 Compute gaussian filters
sigma = 10.0
G = week1.get_gaussian_filter(sigma) # BY NOW YOU SHOULD HAVE THIS FUNCTION IMPLEMENTED

fig = plt.figure()
plt.plot(G)
fig.suptitle('My gaussian filter') # HERE YOU SHOULD GET A BELL CURVE

# F.7.2 Apply gaussian filters
imp_gfilt = week1.apply_gaussian_conv(imp, G) # [ALREADY IMPLEMENTED, ADDED HERE ONLY FOR VISUALIZATION PURPOSES]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imshow(imp_gfilt, cmap='gray')
ax.set_title('Gaussian convolution: my implementation')
ax = fig.add_subplot(1, 2, 2)
imshow(tools.gf_2d(sigma, H), cmap='gray')
ax.set_title('Gaussian Kernel already provided')

# F.7.3 Apply first order derivative gradient filter
dG = week1.get_gaussian_der_filter(sigma, 1) # BY NOW YOU SHOULD HAVE THIS FUNCTION IMPLEMENTED
imp_drow = week1.apply_filter(imp, dG, 'row') # [ALREADY IMPLEMENTED]
imp_dcol = week1.apply_filter(imp, dG, 'col') # [ALREADY IMPLEMENTED]

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imshow(imp_drow, cmap='gray')
ax = fig.add_subplot(1, 2, 2)
imshow(imp_dcol, cmap='gray')

