# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PRE A : GO TO MAIN DIR FOR WEEK2, THAT IS WHERE MAIN_SCRIPT_WEEK2.PY IS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import sys
import os

from scipy import ndimage
from pylab import *
from numpy import *
from collections import defaultdict

import homography
import warp

import week2
sys.path.insert(0, '../')
import tools
import platform
import week1

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PRE B : SETTING PRINTING OPTIONS [FEEL FREE TO CHANGE THE VALUES TO YOUR LIKING]
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# STARTING THE ASSIGNMENT
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##############################################################################
#### PART 2. SIFT FEATURES
##############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP A: READ AN IMAGE AND EXTRACT SIFT FEATURES FROM IT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
reload(week2)
impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

print frames1
print sift1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP B: PLOT SIFT FEATURES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
figure(1)
subplot(1, 2, 1)
im1 = Image.open(impath1)
week2.plot_features(im1, frames1, False, 'r')
subplot(1, 2, 2)
week2.plot_features(im1, frames1, True, 'r') # [YOU NEED TO FINISH THE IMPLEMENTATION FOR THIS]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP C: PLOT SIFT MATCHES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impath1 = '../../data/oxford_scaled/all_souls_000075.jpg'
frames1, sift1 = week2.compute_sift(impath1)

impath2 = '../../data/oxford_scaled/all_souls_000076.jpg'
frames2, sift2 = week2.compute_sift(impath2)

figure(1)
subplot(1, 2, 1)
im1 = Image.open(impath1)
week2.plot_features(im1, frames1, False, 'r')
subplot(1, 2, 2)
im2 = Image.open(impath2)
week2.plot_features(im2, frames2, False, 'r')


# STEP C.1 NORMALIZE SIFT VECTORS
DEFINE_AXIS=0
sift1 = tools.normalizeL2(sift1, DEFINE_AXIS)
sift2 = tools.normalizeL2(sift2, DEFINE_AXIS)

threshold = 1.1
# setup our array of matches
matches = np.zeros(len(sift1))

sift2transp = sift2.T

# loop through all sift1 features
for i in range(len(sift1)):
    # make sure vars are empty before looping. canopy rules
    dist_sift = []
    dist_sift_sort = []l

    # compute distance for each sift1 vector with all sift2 vectors and keep track of distances in dist_sift     
    dist_sift = dot(sift1[i], sift2transp)

    # sort all found distances to get the 2 best matches (0 & 1)
    dist_sift_sort = np.argsort(dist_sift)
    
    # since the matches are the highest ones, they're at the end of the sorted array
    dist_frst = (len(dist_sift)-1)
    dist_scnd = (len(dist_sift)-2)
    
    # apply Lowe's match criterion
    if (dist_sift[dist_sift_sort[dist_frst]] / dist_sift[dist_sift_sort[dist_scnd]]) > threshold:
        # seems we've found a match for feature sift1[i]
        # adding the sift2 index to our match array
        matches[i] = dist_sift_sort[dist_frst]
    else:
        # no match for sift1[i] in sift2.
        # so no kudos for this one
        matches[i] = -1

# STEP C.2 COMPUTE DISTANCES BETWEEN ALL VECTORS IN SIFT1 AND SIFT2.

# PUT YOUR CODE IN week2.match_sift, SO THAT IT CAN BE CALLED AS
#dist_thresh = ...
#matches = week2.match_sift(sift1, sift2, dist_thres) # [YOU NEED TO IMPLEMENT THIS]

# NOW YOU ARE ABLE TO PLOT THE MATCHES AND GET SOMETHING LIKE IN FIG.1 OF THE HANDOUT
week2.plot_matches(im1, im2, frames1, frames2, matches) # [ALREADY IMPLEMENTED]

# !!! EXPERIMENT FOR DIFFERENT VALUES OF DIST_THRESH AND CONTINUE WITH THAT ONE.
# !!! REPORT THIS VALUE IN YOUR REPORT
# Rog - Result for assignment in C2
impath1 = '../../data/oxford_scaled/all_souls_000075.jpg'
frames1, sift1 = week2.compute_sift(impath1)
impath2 = '../../data/oxford_scaled/all_souls_000076.jpg'
frames2, sift2 = week2.compute_sift(impath2)

im1 = Image.open(impath2)
im2 = Image.open(impath2)

matches = week2.match_sift(sift1, sift2, 1.15)
week2.plot_matches(im1, im2, frames1, frames2, matches)

##############################################################################
#### PART 3. SIFT FEATURES ARE INVARIANT TO
##############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP D: ROTATION (REPEAT FOR 15, 30, 45, 60, 75, 90)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

im1 = Image.open(impath1)

for deg in np.arange(0, 90+15, 15):
    im_rot = im1.rotate(np.int(deg))
    im_rot.save('temp_rot.jpg')
    frames_rot, sift_rot = week2.compute_sift('temp_rot.jpg')
    matches = week2.match_sift(sift1, sift_rot, 1.15) # [YOU NEED TO IMPLEMENT THIS]
    week2.plot_matches(im1, im_rot, frames1, frames_rot, matches) # [ALREADY IMPLEMENTED]
    title('Rotating ' + str(deg) + ': Number of matches is ' + str(sum(matches!=-1)))
    
# MAKE A PLOT HERE. ON THE X-AXIS THERE SHOULD BE THE ROTATION CHANGES. ON THE Y-AXIS
# THERE SHOULD BE THE NUMBER OF MATCHES. BASED ON THE PLOT TELL IN YOUR OPINION IF SIFT
# IS ROTATION INVARIANT, THAT IS IF THE NUMBER OF MATCHES REMAINS SIMILAR FOR ROTATION CHANGES

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP E: SCALE  (REPEAT FOR x0.2, x0.5, x0.8, x1.2, x1.5, x1.8)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

im1 = Image.open(impath1)
H, W, C = np.array(im1).shape

for alpha in np.arange(0.2, 2.0 + 0.2, 0.2):
    print alpha
    im_scl = im1.resize((int(alpha * W), int(alpha * H)))
    im_scl.save('temp_scl.jpg')
    frames_scl, sift_scl = week2.compute_sift('temp_scl.jpg')
    matches = week2.match_sift(sift1, sift_scl, 1.15) # [YOU NEED TO IMPLEMENT THIS]
    week2.plot_matches(im1, im_scl, frames1, frames_scl, matches) # [ALREADY IMPLEMENTED]
    title('Scaling x' + str(alpha) + ': Number of matches is ' + str(sum(matches!=-1)))
    
# MAKE A PLOT HERE. ON THE X-AXIS THERE SHOULD BE THE SCALE CHANGES. ON THE Y-AXIS
# THERE SHOULD BE THE NUMBER OF MATCHES. BASED ON THE PLOT TELL IN YOUR OPINION IF SIFT
# IS SCALE INVARIANT, THAT IS IF THE NUMBER OF MATCHES REMAINS SIMILAR FOR SCALE CHANGES

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP F: PERSPECTIVE (REPEAT FOR ALL PERSPECTIVE CHANGED IMAGES)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# '../../data/other/all_souls_000026_prsp1.jpg'
# '../../data/other/all_souls_000026_prsp2.jpg'
# '../../data/other/all_souls_000026_prsp3.jpg'
# '../../data/other/all_souls_000026_prsp4.jpg'

impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

im1 = Image.open(impath1)

for i in range(4):
    im_prsp = Image.open('../../data/other/all_souls_000026_prsp' + str(i+1) + '.jpg')
    frames_prsp, sift_prsp = week2.compute_sift('../../data/other/all_souls_000026_prsp' + str(i+1) + '.jpg')
    matches = week2.match_sift(sift1, sift_prsp, 1.15) # [YOU NEED TO IMPLEMENT THIS]
    week2.plot_matches(im1, im_prsp, frames1, frames_prsp, matches) # [ALREADY IMPLEMENTED]
    title('Pespective ' + str(i+1) + ': Number of matches is ' + str(sum(matches!=-1)))
    
# MAKE A PLOT HERE. ON THE X-AXIS THERE SHOULD BE THE PERSPECTIVE CHANGES. ON THE Y-AXIS
# THERE SHOULD BE THE NUMBER OF MATCHES. BASED ON THE PLOT TELL IN YOUR OPINION IF SIFT
# IS PERSPECTIVE INVARIANT, THAT IS IF THE NUMBER OF MATCHES REMAINS SIMILAR FOR PERSPECTIVE CHANGES

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP G: BRIGHTNESS (REPEAT FOR BRIGHTNESS CHANGES 0.5, 0.8, 1.2, 1.5, 1.8)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impath1 = '../../data/oxford_scaled/all_souls_000026.jpg'
frames1, sift1 = week2.compute_sift(impath1)

im1 = Image.open(impath1)

for beta in [0.5, 0.8, 1.2, 1.5, 1.8]:
    im_brt = ImageEnhance.Brightness(im1)
    im_brt = im_brt.enhance(beta)
    im_brt.save('temp_brt.jpg')
    frames_brt, sift_brt = week2.compute_sift('temp_brt.jpg')
    matches = week2.match_sift(sift1, sift_brt, 1.15) # [YOU SHOULD HAVE THIS IMPLEMENTED BY NOW]
    week2.plot_matches(im1, im_brt, frames1, frames_brt, matches) # [ALREADY IMPLEMENTED]
    title('Brightness x' + str(beta) + ': Number of matches is ' + str(sum(matches!=-1)))

# MAKE A PLOT HERE. ON THE X-AXIS THERE SHOULD BE THE BRIGHTNESS CHANGES. ON THE Y-AXIS
# THERE SHOULD BE THE NUMBER OF MATCHES. BASED ON THE PLOT TELL IN YOUR OPINION IF SIFT
# IS BRIGHTNESS INVARIANT, THAT IS IF THE NUMBER OF MATCHES REMAINS SIMILAR FOR BRIGHTNESS CHANGES

##############################################################################
#### PART 4. USING SIFT FOR GEOMETRY
##############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP H: FIND THE POINTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Open images
impath1 = '../../data/other/all_souls_000026.jpg'
im1 = Image.open(impath1)
frames1, sift1 = week2.compute_sift(impath1)

impath2 = '../../data/other/all_souls_000068.jpg'
im2 = Image.open(impath2)
frames2, sift2 = week2.compute_sift(impath2)

# Pre-computed homography matrix
H = np.array([[   0.554,    0.049,   72.393],
       [  -0.014,    0.642,  215.047],
       [  -0.   ,    0.   ,    1.   ]])

# VISUALIZING A POINT ON AN IMAGE
p = np.array([562, 52])
p_choice = []
p_choice.append([562, 52, 'ro', 'rd'])
p_choice.append([398,286, 'bo', 'bd'])
p_choice.append([138,261, 'go', 'gd'])
p_choice.append([498,405, 'yo', 'yd'])
p_choice.append([81,494, 'co', 'cd'])
p_choice.append([781,500, 'mo', 'md'])

figure(1)
imshow(im1)
axis('off')

for i in range(len(p_choice)):
    figure(1)
    plot(p_choice[i][0], p_choice[i][1], p_choice[i][2], markersize=10)
    
figure(2)
imshow(im2)
axis('off')
for i in range(len(p_choice)):
    p_calc = np.array([p_choice[i][0], p_choice[i][1]])
    newp = week2.project_point_via_homography(H, p_calc)
    plot(newp[0], newp[1], p_choice[i][3], markersize=10)



figure(1)
plot(p[0], p[1], 'ro', markersize=10)

# IN ORDER TO FIND THE LOCATION OF THE POINT TO THE NEW IMAGE, YOU NEED TO APPLY EQ.(5) FROM THE HANDOUT
newp = week2.project_point_via_homography(H, p) # [YOU NEED TO FINISH THE IMPLEMENTATION OF THIS FUNCTION]

# VISUALIZING THE SAME POINT, ESTIMATED ON THE SECOND IMAGE
figure(2)
imshow(im2)
plot(newp[0], newp[1], 'rd', markersize=10)
axis('off')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP I: COMPUTE HOMOGRAPHY BETWEEN 2 IMAGES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impath1 = '../../data/other/all_souls_000015.jpg'
im1 = Image.open(impath1)
frames1, sift1 = week2.compute_sift(impath1)

impath2 = '../../data/other/all_souls_000091_half.jpg'
im2 = Image.open(impath2)
frames2, sift2 = week2.compute_sift(impath2)

matches = week2.match_sift(sift1, sift2, 1.15) # BY NOW YOU SHOULD HAVE THIS ALREADY IMPLEMENTED
H, inliers = homography.estimate_homography_with_ransac(frames1, frames2, matches) # ALREADY PROVIDED

ix = matches != -1
# PLOT GOOD MATCHES
good_matches = -1 * np.ones(frames1.shape[0])
good_matches[inliers] = inliers
week2.plot_matches(im1, im2, frames1[ix], frames2[matches[ix]], good_matches)

# PLOT BAD MATCHES WITH A DIFFERENT COLOR
bad_matches = np.arange(frames1.shape[0])
bad_matches[inliers] = -1
week2.plot_matches(im1, im2, frames1[ix], frames2[matches[ix]], bad_matches, 'b', False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### STEP J: IMAGE RECOGNITION WITH GEOMETRIC VERIFICATION
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

files = os.listdir('../../data/oxford_scaled/')
query = 'all_souls_000040.jpg'
impath_qu = '../../data/oxford_scaled/' + query
frames_qu, sift_qu = week2.compute_sift(impath_qu)

# LOOP OVER ALL FILES AND FIND THE ONES THAT MATCH WELL GEOMETRICALLY WITH THE QUERY. FIND A GOOD VALUE FOR
# A MINIMUM NUMBER OF INLIERS, USING THE estimate_homography_with_ransac function from above

# Build imgdata array, so we can later go over this to be fasterrrrrr
imgdata = []
for i in range(len(files)):
    imgpath = '../../data/oxford_scaled/' + files[i]
    print 'processing: ', imgpath
    frames_tmp, sift_tmp = week2.compute_sift(imgpath)
    imgdata.append([i, frames_tmp, sift_tmp])

# After building imgdata, do this double-time!
matches = []
recognized = []
min_inliers = 0
print 'Query image: ', impath_qu
for i in range(len(imgdata)):
    matches = week2.match_sift(sift_qu, imgdata[i][2], 1.15)
    try:
        H, inliers = homography.estimate_homography_with_ransac(frames_qu, imgdata[i][1], matches) 
        if len(inliers) >= min_inliers:
            # jeuj!
            print ' -> ' + str(files[i]) + ' more than ' + str(min_inliers) + ' inliers:', len(inliers)            
            recognized.append(files[i])
    except:
        pass
print 'Done'
