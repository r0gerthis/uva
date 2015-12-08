import numpy as np
import math
import sys
import platform
from PIL import Image
from pylab import *
import os
from matplotlib import cm
import matplotlib.colors as cols
import pickle
sys.path.insert(0, '../')
import tools
import random

##############################################################################
## YOUR IMPLEMENTATIONS
##############################################################################

def mykmeans(x, K):
    max_iter = 20
    codebook = []
    
    # PART 1. STEP 0. PICK RANDOM CENTERS
    means = np.array(random.sample(x, K))
    
    for it in range(max_iter):
        # STEP 1. CALCULATE DISTANCE FROM DATA TO CENTERS
        dist = np.zeros([K, x.shape[0]])        
        for i in np.arange(0, K):
            for j in np.arange(0, x.shape[0]):
                dist[i,j] = sum((means[i]-x[j])**2)
                
        # STEP 2. FIND WHAT IS THE CLOSEST CENTER PER POINT
        closest = np.argmin(dist, 0)
        
        # STEP 3. UPDATE CENTERS
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
            
        # STEP 3. INCLUDE PERHAPS TERMINATION CRITERIA ????
    
    # ...
    plot_2d_data(x, None, closest, means)
    return codebook
    
def computeVectorDistance( vec1, vec2, dist_type ):
    dist = 0
    
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
    return dist
    
def rankImages( imdists, query_id, dist_type ):
    ranking = []
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

##############################################################################
### ALREADY IMPLEMENTED, DO NOT TOUCH
##############################################################################

def get_oxford_filedata():
    text_file = open('../../data/oxford_labels.txt', 'r')
    lines = text_file.readlines()
    filenames = []
    labels = []
    label_names = []
    for l in lines:
        temp = l.split()
        filenames.append(temp[0])
        labels.append(int(temp[1]))
        label_names.append(temp[2])
                
    return filenames, labels, label_names

def load_codebook(path):
    f = open(path, 'rb')
    codebook = pickle.load(f)
    f.close()
    return codebook

def load_bow(path):
    f = open(path, 'rb')
    bow = pickle.load(f)
    f.close()
    return bow

def save_bow(path, bow):
    f = open(path, 'wb')
    pickle.dump(bow, f)
    f.close()

def generate_2d_data():
    np.random.seed(0)
    x1 = np.random.rand(30, 2)
    x1[:, 0] = 2 * x1[:, 0] + 3
    x1[:, 1] = 4 * x1[:, 1] + 7
    
    x2 = np.random.rand(30, 2)
    x2[:, 0] = 5 * x2[:, 0] + 3
    x2[:, 1] = 4 * x2[:, 1] + 2
    
    x3 = np.random.rand(30, 2)
    x3[:, 0] = 5 * x3[:, 0] + 7
    x3[:, 1] = 5 * x3[:, 1] + 5
    
    x = np.vstack([x1, x2, x3])
    labels = np.hstack([0 * np.ones(30), 1 * np.ones(30), 2 * np.ones(30)])    
    return x, labels
    
def plot_2d_data(x, labels, assignments, centers):
    colors = cm.Set1(np.linspace(0, 1, 10))
    
    figure    
    if labels is not None:
        for i, c in zip(set(labels), colors):
            i = int(i)
            plot(x[labels == i, 0], x[labels == i, 1], 's', markeredgecolor=c)
    else:
        plot(x[:, 0], x[:, 1], 'ks')
    
    if centers is not None:
        for i, c in zip(np.arange(centers.shape[0]), colors):
            i = int(i)
            plot(centers[i, 0], centers[i, 1], 'x', markeredgecolor=c, markersize=20, markeredgewidth=3)
            
    if assignments is not None:
        for i, c in zip(set(assignments), colors):
            i = int(i)
            plot(x[assignments == i, 0], x[assignments == i, 1], 'o', markeredgecolor=c, markersize=12, markerfacecolor='none', markeredgewidth=2)

def get_colors(K):
    colors = np.random.rand(K, 3)
    return colors
    
def get_colorbar(colors):
    K = colors.shape[0]
    figure(figsize=(3.5, 8))
    cmap = plt.cm.jet
    cmap = cmap.from_list('Custom cmap', colors, K)
    bounds = np.linspace(0,K,K+1)
    norm = mpl.colors.BoundaryNorm(bounds, K)
    cb = matplotlib.colorbar.ColorbarBase(gca(), cmap=cmap, norm=norm, spacing='proportional', ticks=bounds[0:K+1:10], boundaries=bounds, format='%1i')
    title('Colorbar for visual words')

def show_words_on_image(impath, K, frames, sift, indexes, colors, word_patches):
    figure()
    im = Image.open(impath)
    imshow(im)
    axis('off')
    for k in range(K):        
        ix = indexes == k
        plot(frames[ix, 0], frames[ix, 1], marker='o', markeredgecolor=colors[k, :], markerfacecolor=colors[k, :], markersize=4, linewidth=0.0)
        
        indexes_range = [i for i,x in enumerate(indexes) if x == k]
        #indexes_range = indexes_range[0 : max(len(indexes_range, 10))]
        indexes_range = indexes_range[0 : min(len(indexes_range), 10)]
        
        for i in indexes_range:
            r = frames[i, 1]
            c = frames[i, 0]
            pix = frames[i, 2]
            
            from_row = max(0, r-pix-1)
            to_row = min(size(im, 1), r+pix+1)
            from_col = max(0, c-pix-1)
            to_col = min(size(im, 0), c+pix+1)
            
            patch = np.array(im)[from_row:to_row, from_col:to_col, :]
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                continue
            word_patches[k].append(patch)
            
    return word_patches

def precision_at_N(query_id, gt_labels, ranking, N):
    gt_labels_N = [ gt_labels[i] for i in ranking]
    gt_labels_N = np.array(gt_labels_N[0 : N])
    gt_labels_query = gt_labels[query_id]
    
    prec = float(sum(gt_labels_N == gt_labels_query)) / float(N)
    return prec
    return prec

def compute_sift(impath, edge_thresh = 10, peak_thresh = 5):
    params = '--edge-thresh ' + str(edge_thresh) + ' --peak-thresh ' + str(peak_thresh)

    im1 = Image.open(impath).convert('L')
    filpat1, filnam1, filext1 = tools.fileparts(impath)
    temp_im1 = 'tmp_' + filnam1 + '.pgm'
    im1.save(temp_im1)
    
    import struct
    is_64bit = struct.calcsize('P') * 8 == 64
    if platform.system() == 'Windows' and is_64bit == True:
        sift_exec = '..\\..\\external\\vlfeat-0.9.17\\bin\\win64\\sift.exe'
        command = sift_exec + ' \"' + os.getcwd() + '\\' + temp_im1 + '\" --output \"' + os.getcwd() + '\\' + filnam1 + '.sift.output' + '\" ' + params
    elif platform.system() == 'Windows' and is_64bit == False:
        sift_exec = '..\\..\\external\\vlfeat-0.9.17\\bin\\win32\\sift.exe'
        command = sift_exec + ' \"' + os.getcwd() + '\\' + temp_im1 + '\" --output \"' + os.getcwd() + '\\' + filnam1 + '.sift.output' + '\" ' + params
    elif platform.system() == 'Linux':
        sift_exec = '..//..//external//vlfeat-0.9.17//bin//glnxa64//sift'
        command = sift_exec + ' \"' + os.getcwd() + '//' + temp_im1 + '\" --output \"' + os.getcwd() + '//' + filnam1 + '.sift.output' + '\" ' + params
    elif platform.system() == 'Darwin':
        sift_exec = '..//..//external//vlfeat-0.9.17//bin//maci64//sift'
        command = sift_exec + ' \"' + os.getcwd() + '//' + temp_im1 + '\" --output \"' + os.getcwd() + '//' + filnam1 + '.sift.output' + '\" ' + params
        
    os.system(command)
    frames, sift = read_sift_from_file(filnam1 + '.sift.output')
    os.remove(temp_im1)
    os.remove(filnam1 + '.sift.output')
    return frames, sift
    
def read_sift_from_file(sift_path):
    f = np.loadtxt(sift_path)
    return f[:, :4], f[:, 4:]