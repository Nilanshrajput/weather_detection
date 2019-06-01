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
import Descriptors as des
import glob
import pandas as pd


def hog(img):

    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20),)
    hist = hog.compute(img, winStride, padding, locations)

    return hist


from os import listdir
from os.path import isfile, join

'''
cloudy

'''
count = 0
mypath = 'D:\dev\weather_detectcnn\Image\cl500'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

dim = (400, 300)
im = cv2.imread(join(mypath, onlyfiles[0]))

matrix = des.autoCorrelogram(im)
a = np.asarray(matrix)
a = np.reshape(a, (-1, 64))
df = pd.DataFrame(a)
training_set = []
training_labels = []

k = 0
training_labels = []
training_labels.append(0)
exceptons = []
e = 0
for n in range(1, len(onlyfiles)):
    im = cv2.imread(join(mypath, onlyfiles[n]))
    matrix = des.autoCorrelogram(im)
    a = np.asarray(matrix)
    if a.shape != (1, 64):
        exceptons.append(join(mypath, onlyfiles[n]))

        e = e + 1
        continue

    df2 = pd.DataFrame(a)
    df = pd.concat([df, df2], axis=0, ignore_index=True)
    k = k + 1
    print(k)
    training_labels.append(0)


'''
foggy
'''
mypath = 'F:\weather_deter\Image\Foggy_im'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for n in range(0, len(onlyfiles)):
    im = cv2.imread(join(mypath, onlyfiles[n]))

    matrix = des.autoCorrelogram(im)
    a = np.asarray(matrix)
    if a.shape != (1, 64):
        exceptons.append(join(mypath, onlyfiles[n]))

        e = e + 1
        continue

    df2 = pd.DataFrame(a)
    df = pd.concat([df, df2], axis=0, ignore_index=True)
    k = k + 1
    print(k)
    training_labels.append(1)

'''
Rainy
'''

mypath = 'F:\weather_deter\Image\Rain_im'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for n in range(0, len(onlyfiles)):
    im = cv2.imread(join(mypath, onlyfiles[n]))

    matrix = des.autoCorrelogram(im)
    a = np.asarray(matrix)
    if a.shape != (1, 64):
        exceptons.append(join(mypath, onlyfiles[n]))

        e = e + 1
        continue

    df2 = pd.DataFrame(a)
    df = pd.concat([df, df2], axis=0, ignore_index=True)
    k = k + 1
    print(k)
    training_labels.append(2)


'''
snowy
'''

mypath = 'F:\weather_deter\Image\snow_im'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for n in range(0, len(onlyfiles)):
    im = cv2.imread(join(mypath, onlyfiles[n]))

    matrix = des.autoCorrelogram(im)
    a = np.asarray(matrix)
    if a.shape != (1, 64):
        exceptons.append(join(mypath, onlyfiles[n]))

        e = e + 1
        continue
    k = k + 1
    df2 = pd.DataFrame(a)
    df = pd.concat([df, df2], axis=0, ignore_index=True)
    print(k)
    training_labels.append(3)

'''
sunny
'''

mypath = 'C:\Users\Abhi\dev\weather_detectcnn\sunny_im'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for n in range(0, len(onlyfiles)):
    im = cv2.imread(join(mypath, onlyfiles[n]))

    matrix = des.autoCorrelogram(im)
    a = np.asarray(matrix)
    if a.shape != (1, 64):
        exceptons.append(join(mypath, onlyfiles[n]))
        e = e + 1
        continue
    df2 = pd.DataFrame(a)
    df = pd.concat([df, df2], axis=0, ignore_index=True)
    k = k + 1
    print(k)
    training_labels.append(4)


trainData = df.values
responses = np.float32(training_labels)
df3 = pd.DataFrame(responses)

df.to_csv('C:\Users\Abhi\dev\weather_detectcnn\Tempc.csv')
df3.to_csv('C:\Users\Abhi\dev\weather_detectcnn\labecl.csv')


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

'''
ppc = 16
hog_images = []
hog_features = []
for image in images:
    fd, hog_image = hg(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2', visualize=True)

    hog_images.append(hog_image)
    hog_features.append(fd)
'''
