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
import fnmatch
import subprocess
from sklearn import svm, datasets


sys.path.insert(0, '../')
import tools

#from week3
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
    

def generate_toy_potential_classifiers(data, labels):
    C = 1
    svc = svm.SVC(kernel='linear', C=C).fit(data, labels)
    sv_id = svc.support_
    alpha_id = svc.dual_coef_
    w = svc.coef_[0]
    w0 = w[0]
    w1 = w[1]
    b = svc.intercept_
    svm_w = np.array([[w0 + 1.5 * randn(), w1 + 1.5 * randn()], 
               [w0 + 0.5 * randn(), w1 + 0.5 * randn()],  
               [w0 + 0.0 * randn(), w1 + 0.0 * randn()], 
               [w0 + 1.5 * randn(), w1 + 1.5 * randn()]])
    svm_b = [[b + 0.5 * randn()],
         [b + 1.5 * randn()],
         [b + 0.0 * randn()],
         [b + 1.5 * randn()]]    
    return svm_w, svm_b

def generate_ring_data():
    rads1 = np.random.rand(500)
    thetas1 = 2 * math.pi * np.random.rand(500)
    x1 = rads1 * [np.cos(thetas1), np.sin(thetas1)]
    y1 = 1 * np.ones(x1.shape[1])
    rads2 = 1.05 + np.random.rand(500)
    thetas2 = 2 * math.pi * np.random.rand(500)
    x2 = rads2 * [np.cos(thetas2), np.sin(thetas2)]
    y2 = -1 * np.ones(x2.shape[1])    
    data = np.concatenate((x1.T, x2.T), axis=0)
    labels = np.concatenate((y1, y2), axis=0)
    
    return data, labels

def generate_toy_data():
    x1 = np.random.rand(100, 2)
    x2 = 0.6 + np.random.rand(100, 2)
    data = np.concatenate((x1, x2), axis=0)
    labels = np.concatenate((np.ones(100), -1 * np.ones(100)), axis=0)
    return data, labels


def get_files_recursively(dir_path, extension='*'):
    matches = []
    for root, dirnames, filenames in os.walk(dir_path):
        for filename in fnmatch.filter(filenames, extension):
            matches.append(os.path.join(root, filename))
            
    return matches

def get_objects_filedata():
    files = get_files_recursively('../../data/objects_scaled/','*.jpg')
    files = sorted(files)
    label_names = [f.split('../../data/objects_scaled/')[1] for f in files]
    if platform.system() == 'Windows':
        label_names = [l.split('\\')[0] for l in label_names]
    else:
        label_names = [l.split('/')[0] for l in label_names]
                
    unique_labels = sorted(set(label_names))
    
    labels = [list(unique_labels).index(l) for l in label_names]
    labels = np.array(labels) + 1
    
    testset = np.arange(0, len(files), 6)
    trainset = np.arange(0, len(files))
    trainset = np.setdiff1d(trainset, testset)
    
    return files, labels, label_names, unique_labels, trainset, testset

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
        
    proc = subprocess.Popen(command,  creationflags=subprocess.SW_HIDE, shell=True)
    proc.wait()
    #os.system(command)
    frames, sift = read_sift_from_file(filnam1 + '.sift.output')
    os.remove(temp_im1)
    os.remove(filnam1 + '.sift.output')
    return frames, sift
    
def read_sift_from_file(sift_path):
    f = np.loadtxt(sift_path)
    return f[:, :4], f[:, 4:]

def load_bow(path):
    f = open(path, 'rb')
    bow = pickle.load(f)
    f.close()
    return bow

def save_bow(path, bow):
    f = open(path, 'wb')
    pickle.dump(bow, f)
    f.close()

