# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PRE A : GO TO MAIN DIR FOR WEEK2, THAT IS WHERE MAIN_SCRIPT_WEEK2.PY IS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy.cluster.vq as cluster
import matplotlib.cm as cmx
from scipy import ndimage
from scipy import misc
from collections import defaultdict
import pickle
import math
import random
import sys
import os

import week3
sys.path.insert(0, '../')
import tools

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 1. KMEANS

# GENERATE RANDOM DATA
x, labels = week3.generate_2d_data()

#week3.plot_2d_data(x, labels, None, None)

# PART 1. STEP 0. PICK RANDOM CENTERS
K = 10
means = np.array(random.sample(x, K))
week3.plot_2d_data(x, None, None, means)

# PART 1. STEP 1. CALCULATE DISTANCE FROM DATA TO CENTERS
dist = np.zeros([K, x.shape[0]])
for i in np.arange(0, K):
    for j in np.arange(0, x.shape[0]):
        dist[i,j] = sum((means[i]-x[j])**2)

# PART 1. STEP 2. FIND WHAT IS THE CLOSEST CENTER PER POINT
closest = np.argmin(dist, 0)
week3.plot_2d_data(x, None, closest, means)

# PART 1. STEP 3. UPDATE CENTERS
for i in np.arange(0, K):
    mean_x = 0.0
    mean_y = 0.0
    count = 0
    for j in np.arange(0, x.shape[0]):
        if closest[j] == i:
            count += 1
            mean_x += x[j,0]
            mean_y += x[j,1]
    means[i, 0] = mean_x / count
    means[i, 1] = mean_y / count
  
week3.plot_2d_data(x, None, closest, means)

## mykmeans implementation
x, labels = week3.generate_2d_data()

week3.mykmeans(x, 3)

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 2. COLOR BASED IMAGE SEGMENTATION

im = Image.open('../../data/coral.jpg')
imshow(im)
im = np.array(im)
im_flat = np.reshape(im, [im.shape[0] * im.shape[1], im.shape[2]])

N = 10000
im_flat_random = np.array(random.sample(im_flat, N))

K = 100
[codebook, dummy] = cluster.kmeans(im_flat_random, K)     # RUN SCIPY KMEANS
[indexes, dummy] = cluster.vq(im_flat, codebook)   # VECTOR QUANTIZE PIXELS TO COLOR CENTERS

im_vq = codebook[indexes]
im_vq = np.reshape(im_vq, (im.shape))
im_vq = Image.fromarray(im_vq, 'RGB')

figure
subplot(1, 2, 1)
imshow(im)
subplot(1, 2, 2)
imshow(im_vq)
title('K=' + str(K))

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 3. k-MEANS AND BAG-OF-WORDS

codebook = week3.load_codebook('../../data/codebook_100.pkl')
K = codebook.shape[0]
colors = week3.get_colors(K)

files = os.listdir('../../data/oxford_scaled/')

# PART 3. STEP 1. VISUALIZE WORDS ON IMAGES
word_patches = defaultdict(list)
files_random = random.sample(files, 5)

f = 'all_souls_000057.jpg'
impath = '../../data/oxford_scaled/' + f
frames, sift = week3.compute_sift(impath)       # COMPUTE SIFT
[indexes, dummy] = cluster.vq(sift, codebook)   # VECTOR QUANTIZE SIFT TO WORDS

word_patches = week3.show_words_on_image(impath, 100, frames, sift, indexes, colors, word_patches)    # VISUALIZE WORDS

# PART 4. BAG-OF-WORDS IMAGE REPRESENTATION
# USE THE np.bincount COUNTING THE INDEXES TO COMPUTE THE BAG-OF-WORDS REPRESENTATION,
bow = np.bincount(indexes)
matplotlib.pyplot.bar(range(0, len(bow)), bow, 0.8, None, None)

# PART 3. STEP 2. PLOT COLORBAR
week3.get_colorbar(colors)

# PART 3. STEP 3. PLOT WORD CONTENTS
k = 0
WN = len(word_patches[k])
figure()
suptitle('Word ' + str(k))
for i in range(WN):
    subplot(int(math.ceil(sqrt(WN))), int(math.ceil(sqrt(WN))), i+1)
    imshow(Image.fromarray(word_patches[k][i], 'RGB'))
    axis('off')

# PART 4. BAG-OF-WORDS IMAGE REPRESENTATION
# USE THE np.bincount COUNTING THE INDEXES TO COMPUTE THE BAG-OF-WORDS REPRESENTATION,
bow = np.bincount(indexes)
matplotlib.pyplot.bar(range(0, len(bow)), bow, 0.8, None, None)

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PART 5. PERFORM RETRIEVAL WITH THE BAG-OF-WORDS MODEL

# PART 5. STEP 1. LOAD BAG-OF-WORDS VECTORS FROM ../../data/bow/codebook_100/ using the week3.load_bow function

# parameters for the code below...
CODEBOOK_DIR = '../../data/bow/codebook_10/'
DISTANCE = 'l2'
query_img = 'all_souls_000065.jpg' + '.pkl'
#query_img = 'radcliffe_camera_000390.jpg' + '.pkl'
#query_img = 'christ_church_000190.jpg' + '.pkl'

# used for testing:
#query_img = 'all_souls_000026.jpg' + '.pkl'

images = os.listdir('../../data/oxford_scaled/')
files = os.listdir(CODEBOOK_DIR)
bows = []
for i in range(len(files)):
    # get all bow vectors from the files 
    #bows.append(week3.load_bow('../../data/bow/codebook_50/' + files[i]))
    bows.append(week3.load_bow(CODEBOOK_DIR + files[i]))
    
# PART 5. STEP 2. COMPUTE DISTANCE MATRIX
# setup matrix for comparison outcomes. 0,0 = img1,img1, 0,1=img1,img2
bowdists = np.zeros((len(bows), len(bows)))
# setup check matrix to keep track of comparisons already done. set to 1 if done
bowcheck = np.zeros((len(bows), len(bows)))

# reset for canopy...
bowdiststorage = 0
x = -1

for bw1 in bows:
    x += 1
    y = -1
    bw1n = tools.normalizeL2(bw1) # for l2 distance
    #bw1n = tools.normalizeL1(bw1) # for intersect distance
    for bw2 in bows:
        bowdiststorage = 0
        y += 1
        if bowcheck[x,y] == 0 and bowcheck[y,x] == 0:
            # new bow distance
            bowcheck[x,y] = 1
            bowcheck[y,x] = 1
            bw2n = tools.normalizeL2(bw2) # for l2 distance
            #bw2n = tools.normalizeL1(bw2) # for intersect distance
            bowdiststorage = week3.computeVectorDistance(bw1n, bw2n, DISTANCE)
            bowdists[x,y] = bowdiststorage
            bowdists[y,x] = bowdiststorage

# PART 5. STEP 3. PERFORM RANKING SIMILAR TO WEEK 1 & 2 WITH QUERIES 'all_souls_000065.jpg', 'all_souls_0000XX.jpg', 'all_souls_0000XX.jpg'
# to find query_id we've to figure out the id in the array
for f in range(len(files)):
    if query_img == files[f]:
        query_id = f

ranking = week3.rankImages(bowdists, query_id, DISTANCE)

fig = plt.figure()
ax = fig.add_subplot(1, 11, 1)
im = imread('../../data/oxford_scaled/' +images[query_id])
ax.imshow(im)
ax.axis('off')
ax.set_title('Query')

for i in np.arange(1, 1+10):
    ax = fig.add_subplot(1, 11, i+1)
    im = imread('../../data/oxford_scaled/' + images[ranking[i-1]]) # The 0th image is the query itself
    ax.imshow(im)
    ax.axis('off')
    #ax.set_title(images[ranking[i-1]])
    ax.set_title('Rank #%s' % i)


# PART 5. STEP 4. COMPUTE THE PRECISION@5
files, labels, label_names = week3.get_oxford_filedata()
# ...
prec5 = week3.precision_at_N(0, gt_labels, ranking, 5)

# PART 5. STEP 4. IMPLEMENT & COMPUTE AVERAGE PRECISION




