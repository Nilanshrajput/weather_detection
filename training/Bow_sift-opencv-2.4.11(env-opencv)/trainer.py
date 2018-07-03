import numpy as np
import cv2
from skimage import color
from matplotlib import pyplot as plt
from skimage.feature import hog as hg
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, accuracy_score

from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

import glob
import pandas as pd


from os import listdir
from os.path import isfile, join

# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")


def sifthistogram(des_list, k, image_paths, count):

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        a = np.asarray(descriptor)
        print a.shape
        if a.shape == ():
            continue
        c, d = a.shape

        print d
        descriptors = np.vstack((descriptors, descriptor))

        print descriptors.shape
        print descriptor.shape
    print descriptors
    # Perform k-means clustering
    size = len(image_paths) - count
    voc, variance = kmeans(descriptors, k, 1)

    # Calculate the histogram of features
    im_features = np.zeros((size, k), "float32")
    for i in xrange(size):
        words, distance = vq(des_list[i][1], voc)
        print i
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * size + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    return im_features


'''
cloudy

'''


def data(kmean, path):

    pathfront = join('C:\Users\Abhi\dev\weather_detectcnn\Bow_sift-opencv-2.4.11(env-opencv)', path)

    count = 0
    mypath = 'F:\weather_deter\Image\cl500'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    k = 0

    count = 0
    des_list1 = []
    for n in range(0, len(onlyfiles)):
        im = cv2.imread(join(mypath, onlyfiles[n]))
        dim = (400, 300)
        im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        a = np.asarray(des)
        print a.shape
        if a.shape == ():
            print ("sssssssssssssssssssss")
            print n
            print onlyfiles[n]
            count += 1
            continue
        des_list1.append((join(mypath, onlyfiles[n]), des))

        k = k + 1
        print(k)

    a = sifthistogram(des_list1, kmean, onlyfiles, count)
    print a.shape
    print a
    df1 = pd.DataFrame(a)
    print df1
    c = k
    dfcoludy = df1
    dfcoludy.to_csv(join(pathfront, 'dfcoludy.csv'))

    '''
    Foggy
    '''
    count = 0
    mypath = 'F:\weather_deter\Image\Fg500'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    des_list2 = []
    for n in range(0, len(onlyfiles)):
        im = cv2.imread(join(mypath, onlyfiles[n]))
        dim = (400, 300)
        im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        a = np.asarray(des)
        print a.shape
        if a.shape == ():
            print ("sssssssssssssssssssss")
            print n
            print onlyfiles[n]
            count += 1
            continue
        des_list2.append((join(mypath, onlyfiles[n]), des))

        k = k + 1
        print(k)

    a = sifthistogram(des_list2, kmean, onlyfiles, count)
    print a.shape
    print a
    df2 = pd.DataFrame(a)

    print df2
    dffoggy = df2
    dffoggy.to_csv(join(pathfront, 'dfoggy.csv'))
    c = k

    '''
    Rainy
    '''

    mypath = 'F:\weather_deter\Image\Rn500'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    des_list3 = []
    count = 0
    for n in range(0, len(onlyfiles)):
        im = cv2.imread(join(mypath, onlyfiles[n]))
        dim = (400, 300)
        im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        a = np.asarray(des)
        print a.shape
        if a.shape == ():
            print ("sssssssssssssssssssss")
            print n
            print onlyfiles[n]
            count += 1
            continue
        des_list3.append((join(mypath, onlyfiles[n]), des))

        k = k + 1
        print(k)

    a = sifthistogram(des_list3, kmean, onlyfiles, count)
    print a.shape
    print a
    df3 = pd.DataFrame(a)
    dfrainy = df3
    dfrainy.to_csv(join(pathfront, 'dfrainy.csv'))

    '''
    snowy
    '''
    count = 0
    mypath = 'F:\weather_deter\Image\sn500'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    des_list4 = []
    for n in range(0, len(onlyfiles)):
        im = cv2.imread(join(mypath, onlyfiles[n]))
        dim = (400, 300)
        im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        a = np.asarray(des)
        print a.shape
        if a.shape == ():
            print ("sssssssssssssssssssss")
            print n
            print onlyfiles[n]
            count += 1
            continue
        des_list4.append((join(mypath, onlyfiles[n]), des))

        k = k + 1
        print(k)

    a = sifthistogram(des_list4, kmean, onlyfiles, count)
    print a.shape
    print a
    df4 = pd.DataFrame(a)

    dfsnowy = df4
    c = dfsnowy.to_csv(join(pathfront, 'dfsnowy.csv'))
    c = k

    mypath = 'F:\weather_deter\Image\sunny500'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    count = 0
    des_list5 = []
    for n in range(0, len(onlyfiles)):
        im = cv2.imread(join(mypath, onlyfiles[n]))
        dim = (400, 300)
        im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        a = np.asarray(des)
        print a.shape
        if a.shape == ():
            print ("sssssssssssssssssssss")
            print n
            print onlyfiles[n]
            count += 1
            continue
        des_list5.append((join(mypath, onlyfiles[n]), des))

        k = k + 1
        print(k)

    a = sifthistogram(des_list5, kmean, onlyfiles, count)
    print a.shape
    print a
    df5 = pd.DataFrame(a)

    dfsunny = df5

    dfsunny.to_csv(join(pathfront, 'dfsunny.csv'))
    c = k

    a, b = df1.shape
    label1 = np.ones([a, 1]) * 0
    a, b = df2.shape
    label2 = np.ones([a, 1]) * 1
    a, b = df3.shape
    label3 = np.ones([a, 1]) * 2
    a, b = df4.shape
    label4 = np.ones([a, 1]) * 3
    a, b = df5.shape
    label5 = np.ones([a, 1]) * 4

    df = pd.DataFrame(df1)
    label = pd.DataFrame(label1)

    df = pd.concat([df, df2], axis=0, ignore_index=True)
    label = pd.concat([label, pd.DataFrame(label2)], axis=0, ignore_index=True)
    df = pd.concat([df, df3], axis=0, ignore_index=True, sort=False)
    label = pd.concat([label, pd.DataFrame(label3)], axis=0, ignore_index=True)

    df = pd.concat([df, df4], axis=0, ignore_index=True, sort=False)
    label = pd.concat([label, pd.DataFrame(label4)], axis=0, ignore_index=True)

    df = pd.concat([df, df5], axis=0, ignore_index=True, sort=False)
    label = pd.concat([label, pd.DataFrame(label5)], axis=0, ignore_index=True)

    trainData = df.values
    responses = label.values.ravel()

    df.to_csv(join(pathfront, 'Feature_lbp.csv'))
    label.to_csv(join(pathfront, 'label_lbp_500.csv'))

    X_train, X_test, y_train, y_test = train_test_split(trainData, responses, test_size=0.2, random_state=42)
    clf = LinearSVC(random_state=0)
    clf.fit(X_train, y_train)

    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print (scores)

    y_pred = clf.predict(X_test)

    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
    print('\n')
    print(classification_report(y_test, y_pred))

    thefile = open('test.txt', 'w')
    for item in exceptons:
        thefile.write("%s\n" % item)


data(16, 'k16')
data(32, 'k32')
data(64, 'k64')
data(128, 'k128')
