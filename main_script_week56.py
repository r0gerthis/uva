import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as clr
#import scipy as sc
#import scipy.cluster.vq as cluster
import random
import os
import matplotlib.cm as cmx
import pickle
from collections import defaultdict
import math
sys.path.insert(0, '../')
import tools
import math
import pylab as pl
from sklearn import svm, datasets
#import operator
import week56

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

####

files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

C = 100
bow = np.zeros([len(files), C])
cnt = -1
for impath in files:
    cnt = cnt + 1
    print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(C) + '/' + filnam2 + '/' + filnam + '.pkl')

# Q1: IMPLEMENT HERE kNN CLASSIFIER. 
# YOU CAN USE CODE FROM PREVIOUS WEEK

K = 9 
query_id = 366# (366, goal), (150, bicycle), (84, beach ), (450, mountain)
bow_qid = tools.normalizeL1(bow[query_id, :])
dist = []

for i in range(len(trainset)):
    bow_tmp = tools.normalizeL1(bow[trainset[i], :])
    dist.append(sum(np.minimum(bow_qid, bow_tmp)))
    
#ranking = np.argsort(dist[query_id, :])
ranking = np.argsort(dist)
ranking = ranking[::-1]
nearest_labels = labels[trainset[ranking[0 : K]]]

# VISUALIZE RESULTS
figure
subplot(2, 6, 1)
imshow(Image.open(files[query_id]))
#title('Query')
title('Predicted label: '+ unique_labels[argmax(bincount(nearest_labels))-1])
axis('off')

for cnt in range(K):
    subplot(2, 6, cnt+2)
    imshow(Image.open(files[trainset[ranking[cnt]]]))
    title(unique_labels[nearest_labels[cnt]-1])
    axis('off')


# Q2: USE DIFFERENT STRATEGY
# weighing where first label is more important than the 2nd.
w = array([0.19, 0.17, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.04])

K = 9 
query_id = 366# (366, goal), (150, bicycle), (84, beach ), (450, mountain)
bow_qid = tools.normalizeL1(bow[query_id, :])
dist = []

for i in range(len(trainset)):
    bow_tmp = tools.normalizeL1(bow[trainset[i], :])
    dist.append(sum(np.minimum(bow_qid, bow_tmp)))

ranking = np.argsort(dist)
ranking = ranking[::-1]
nearest_labels = labels[trainset[ranking[0 : K]]]

# VISUALIZE RESULTS
figure
subplot(2, 6, 1)
imshow(Image.open(files[query_id]))
#title('Query')
title('Predicted label: '+ unique_labels[argmax(bincount(nearest_labels,w))-1])
axis('off')

for cnt in range(K):
    subplot(2, 6, cnt+2)
    imshow(Image.open(files[trainset[ranking[cnt]]]))
    title(unique_labels[nearest_labels[cnt]-1])
    axis('off')



# Q3: For K = 9, COMPUTE THE CLASS ACCURACY FOR THE TESTSET
K = 15

# get the distances
dist = np.zeros([len(testset), len(trainset)])
for i in range(len(testset)):
    bow_tst = tools.normalizeL1(bow[testset[i], :])
    print testset[i]
    for j in range(len(trainset)):
        bow_trn = tools.normalizeL1(bow[trainset[j], :])
        dist[i,j] = sum(np.minimum(bow_tst, bow_trn))
        
# lbl test set with kNN
test_labels = []
for i in range(len(testset)):
    ranking = np.argsort(dist[i, :])
    ranking = ranking[::-1]
    nearest_labels = labels[trainset[ranking[0 : K]]]
    if i == 0:
        print i, nearest_labels
        print bincount(nearest_labels)
        print argmax(bincount(nearest_labels))
    test_labels.append(argmax(bincount(nearest_labels)))

classAcc = np.zeros(len(unique_labels))
tp = 0
fn = 0
for c in range(len(unique_labels)):
    tp = 0
    fn = 0
    ilbl = c + 1
    for t in range(len(testset)):
        if labels[testset[t]] == ilbl:
            if test_labels[t] == labels[testset[t]]:
                tp += 1
            else:
                fn += 1
    classAcc[c] = tp / (tp + fn + 0.0)

print classAcc
print np.mean(classAcc)

# REPORT THE CLASS ACC *PER CLASS* and the MEAN
# THE MEAN SHOULD BE (CLOSE TO): 0.31

# ---------------------------------------------------------------
#    RESTART CANOPY AND EXEC ALL OF BELOW FOR THIS ONE>.. w00t
# ---------------------------------------------------------------

# Get data
files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()
K = [1, 3, 5, 7, 9, 15]
# Get BOW
C = 100
bow = np.zeros([len(files), C])
cnt = -1
for impath in files:
    cnt = cnt + 1
    print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(C) + '/' + filnam2 + '/' + filnam + '.pkl')

# Use random indices and three groups of (almost) the same size
np.random.shuffle(trainset)
inx1 = trainset[0:166]
inx2 = trainset[167:332]
inx3 = trainset[333::]

perfK = np.zeros([len(K),3])

for ki in range(len(K)):
    k = K[ki]
    #print k
                    
    # - LOOP OVER DIFFERENT COMBINATIONS OF inx1,inx2,inx3
    for t in range(3):
        if t == 0: t_train = np.concatenate((inx1, inx2)); t_val = inx3; print '### FIRST SET ###';
        elif t == 1: t_train = np.concatenate((inx2, inx3)); t_val = inx1;  print '### SECOND SET ###';
        else: t_train = np.concatenate((inx1, inx3)); t_val = inx2; print '### THIRD SET ###';
    
        # - MEASURE THE MEAN CLASSIFICATION ACCURACY FOR ALL IMAGES IN THE VALIDATION PART    

        # Get distances
        distTable = np.zeros([len(t_val), len(t_train)])
        for i in range(len(t_val)):
            bowTst = tools.normalizeL1(bow[t_val[i], :])
            for j in range(len(t_train)):
                bowTrn = tools.normalizeL1(bow[t_train[j], :])
                distTable[i,j] = sum(np.minimum(bowTst, bowTrn))
        
        # Learn labels with kNN
        rankTable = np.argsort(distTable,1)
        rankTable = rankTable[:,::-1]
        learnedLabels = []
        for tlbl in range(len(t_val)):
            nearLbls = labels[t_train[rankTable[tlbl, 0:k]]]
            learnedLabels.append(argmax(bincount(nearLbls)))
            
        # Get mean class-accuracy
        classAcc = np.zeros(len(unique_labels))
        for c in range(len(unique_labels)):
            tp = 0
            fn = 0
            iLbl = c+1
            for tacc in range(len(t_val)):
                if labels[t_val[tacc]] == iLbl:
                    if learnedLabels[tacc] == labels[t_val[tacc]]:
                        tp += 1
                    else:
                        fn += 1
            classAcc[c] = tp / (tp + fn + 0.0)
            #print 'Class acc for K=' + str(k) + ' with label ' + str(unique_labels[c]) + ' is ' + str(classAcc[c])
        print 'Mean class acc for K=' + str(k) + ' is: ' + str(mean(classAcc))
        perfK[ki, t] = mean(classAcc)
        
# - PICK THE BEST K AS THE VALUE OF K THAT WORKS BEST ON AVERAGE FOR ALL POSSIBLE
mPerfK = mean(perfK,1)
Kbest  = K[np.argmax(mPerfK)]
print 'Best k is: ', Kbest


# PART 3. SVM ON TOY DATA
data, labels = week56.generate_toy_data()
svm_w, svm_b = week56.generate_toy_potential_classifiers(data,labels)

# Q5: CLASSIFY ACCORDING TO THE 4 DIFFERENT CLASSIFIERS AND VISUALIZE THE RESULTS

#pred= np.sign(inner(svm_w[2],data)+svm_b[2])
pred = np.sign(dot(svm_w[2],data.T) + svm_b[2])
pred = append(pred,0)

figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
plt.plot(data[pred==1, 0], data[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data[pred==-1, 0], data[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)


# Q6: USE HERE SVC function from sklearn to run a linear svm
# THEN USE THE PREDICT FUNCTION TO PREDICT THE LABEL FOR THE SAME DATA
svc = svm.SVC(C=1.0, kernel='linear').fit(data, labels)
svc.coef_
svc.intercept_
pred=(svc.predict(data))

# PART 4. SVM ON RING DATA
data, labels = week56.generate_ring_data()

figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')

# Q7: USE LINEAR SVM AS BEFORE, VISUALIZE RESULTS and DRAW PREFERRED CLASSIFICATION LINE IN FIGURE
svc = svm.SVC(C=1.0, kernel='linear').fit(data, labels)
svc.coef_
svc.intercept_
pred=(svc.predict(data))

figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
plt.plot(data[pred==1, 0], data[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data[pred==-1, 0], data[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)

# Q8: (report only) 



C = 1.0  # SVM regularization parameter
# Q9: TRANSFORM DATA TO POLAR COORDINATES FIRST
data, labels = week56.generate_ring_data()

x = data[:, 0]
y = data[:, 1]

rad = numpy.sqrt(x**2+y**2)
ang = numpy.arctan2(y,x)
print rad

data_polar = np.array([rad, ang])
data_polar = data_polar.T

svc = svm.SVC(C=1.0, kernel='linear').fit(data, labels)
svc.coef_
svc.intercept_
pred=(svc.predict(data_polar))

figure()
plt.scatter(data_polar[labels==1, 0], data_polar[labels==1, 1], facecolor='r')
plt.scatter(data_polar[labels==-1, 0], data_polar[labels==-1, 1], facecolor='g')
plt.plot(data_polar[pred==1, 0], data_polar[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data_polar[pred==-1, 0], data_polar[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
 
# PLOT POLAR DATA

data2 = np.vstack((rad, ang))
data2 = data2.T

# Q10: USE THE LINEAR SVM AS BEFORE (BUT ON DATA 2)
svc = svm.SVC(C=1.0, kernel='linear').fit(data_polar, labels)
svc.coef_
svc.intercept_
pred=(svc.predict(data_polar))

# PLOT THE RESULTS IN ORIGINAL DATA
figure()
plt.scatter(data[labels==1, 0], data[labels==1, 1], facecolor='r')
plt.scatter(data[labels==-1, 0], data[labels==-1, 1], facecolor='g')
plt.plot(data[pred==1, 0], data[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data[pred==-1, 0], data[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)


# PLOT POLAR DATA
figure()
plt.scatter(data_polar[labels==1, 0], data_polar[labels==1, 1], facecolor='r')
plt.scatter(data_polar[labels==-1, 0], data_polar[labels==-1, 1], facecolor='g')
plt.plot(data_polar[pred==1, 0], data_polar[pred==1, 1], marker='o', markersize=10, markeredgecolor='r', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)
plt.plot(data_polar[pred==-1, 0], data_polar[pred==-1, 1], marker='o', markersize=10, markeredgecolor='g', markerfacecolor='none', linestyle='none', markeredgewidth=2.0)

# Results
unique_labels = [-1, 1]


classAcc = np.zeros(len(unique_labels))
cnt= 0
for c in unique_labels: #range(len(unique_labels)):
    # Find the true positives, that is the number of images for which pred == labelstest and labelstest == c
    tp = sum(operator.and_(pred == labels, labels == c) == True)
    # Find the false negatives, that is the number of images for which pred != labelstest and labelstest == c
    fn = sum(operator.and_(pred != labels, labels == c) == True)
    #allofthem = sum(labeltest - 1 == c)
    classAcc[cnt] = tp / (tp + fn + 0.0)
    cnt += 1


print classAcc

# PART 5. LOAD BAG-OF-WORDS FOR THE OBJECT IMAGES AND RUN SVM CLASSIFIER FOR THE OBJECTS

files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

K = 4000
bow = np.zeros([len(files), K])
cnt = -1
for impath in files:
    cnt = cnt + 1
    print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(K) + '/' + filnam2 + '/' + filnam + '.pkl')

shuffleArr = np.arange(bow.shape[0])
random.shuffle(shuffleArr)
shuffleBow = bow[shuffleArr, :]
shuffleLbl = labels[shuffleArr]

bow = shuffleBow
labels = shuffleLbl

# Q11: USE linear SVM, perform CROSS VALIDATION ON C = (.1,1,10,100), evaluate using MEAN CLASS ACCURACY
cArr = [0.1, 1, 10, 100, 1000]
inx1 = bow[0:200]
inx2 = bow[201:300]
inx3 = bow[301::]
lblx1 = labels[0:200]
lblx2 = labels[201:300]
lblx3 = labels[301::]

perfC = np.zeros([len(cArr),3])
cntr = -1
for C in cArr:
    cntr += 1
    for t in range(3):
        if t == 0: 
            t_train = np.concatenate((inx1, inx2)); 
            t_lbltr = np.concatenate((lblx1, lblx2));
            t_val = inx3; 
            t_lblva = lblx3
        elif t == 1: 
            t_train = np.concatenate((inx2, inx3)); 
            t_lbltr = np.concatenate((lblx2, lblx3));
            t_val = inx1; 
            t_lblva = lblx1 
        else: 
            t_train = np.concatenate((inx1, inx3)); 
            t_lbltr = np.concatenate((lblx1, lblx3));
            t_val = inx2; 
            t_lblva = lblx2
    
        svc = svm.SVC(C, kernel='linear').fit(t_train, t_lbltr)
        svc.coef_
        svc.intercept_
        pred=(svc.predict(t_val))
            
        classAcc = np.zeros(len(unique_labels))
        for c in range(len(unique_labels)):
            tp = 0
            fn = 0
            iLbl = c+1
            for tacc in range(len(t_val)):
                if t_lblva[tacc] == iLbl:
                    if pred[tacc] == t_lblva[tacc]:
                        tp += 1
                    else:
                        fn += 1
            if (tp+fn) > 0: classAcc[c] = tp / (tp + fn + 0.0)
            else: classAcc[c] = 0
            #print 'Class acc for C=' + str(C) + ' with label ' + str(unique_labels[c]) + ' is ' + str(classAcc[c])
        perfC[cntr, t] = mean(classAcc)
    print 'Mean classification accuracy for K=' + str(K) + ' & C=' + str(C) + ' is: ' + str(mean(perfC[cntr]))

# Q12: Visualize the best performing SVM, what are good classes, bad classes, examples of images etc

# Q13: Compare SVM with k-NN

files, labels, label_names, unique_labels, trainset, testset = week56.get_objects_filedata()

C = 4000
bow = np.zeros([len(files), C])
cnt = -1
for impath in files:
    cnt = cnt + 1
    print str(cnt) + '/' + str(len(files)) + '): ' + impath
    filpat, filnam, filext = tools.fileparts(impath)
    filpat2, filnam2, filext2 = tools.fileparts(filpat)
    bow[cnt, :] = week56.load_bow('../../data/bow_objects/codebook_' + str(C) + '/' + filnam2 + '/' + filnam + '.pkl')

K = 15

# get the distances
dist = np.zeros([len(testset), len(trainset)])
for i in range(len(testset)):
    bow_tst = tools.normalizeL1(bow[testset[i], :])
    #print testset[i]
    for j in range(len(trainset)):
        bow_trn = tools.normalizeL1(bow[trainset[j], :])
        dist[i,j] = sum(np.minimum(bow_tst, bow_trn))
        
# lbl test set with kNN
test_labels = []
for i in range(len(testset)):
    ranking = np.argsort(dist[i, :])
    ranking = ranking[::-1]
    nearest_labels = labels[trainset[ranking[0 : K]]]
    test_labels.append(argmax(bincount(nearest_labels)))

classAcc = np.zeros(len(unique_labels))
tp = 0
fn = 0
for c in range(len(unique_labels)):
    tp = 0
    fn = 0
    ilbl = c + 1
    for t in range(len(testset)):
        if labels[testset[t]] == ilbl:
            if test_labels[t] == labels[testset[t]]:
                tp += 1
            else:
                fn += 1
    classAcc[c] = tp / (tp + fn + 0.0)

print classAcc
print np.mean(classAcc)