import numpy as np
import cv2
import sys


#
# Helper functions
#

def thresholded(center, pixels):
    """
    Compare the center pixel to its 8 neighbors
    """
    ret = []
    for a in pixels:
        if a >= center:
            ret.append(1)
        else:
            ret.append(0)
    return ret


def getNeighborhood(img, idx, idy, default=0):
    """
    Given the center position to find the position of its neighbors.
    """
    try:
        return img[idx, idy]
    except IndexError:
        return default


def unique(a):
    """
    remove duplicates from input list
    """
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


def isValid(X, Y, point):
    """
    Check if point is a valid pixel
    """
    if point[0] < 0 or point[0] >= X:
        return False
    if point[1] < 0 or point[1] >= Y:
        return False
    return True


def getNeighbors(X, Y, x, y, dist):
    """
    Find pixel neighbors according to various distances
    """
    cn1 = (x + dist, y + dist)
    cn2 = (x + dist, y)
    cn3 = (x + dist, y - dist)
    cn4 = (x, y - dist)
    cn5 = (x - dist, y - dist)
    cn6 = (x - dist, y)
    cn7 = (x - dist, y + dist)
    cn8 = (x, y + dist)

    points = (cn1, cn2, cn3, cn4, cn5, cn6, cn7, cn8)
    Cn = []

    for i in points:
        if isValid(X, Y, i):
            Cn.append(i)

    return Cn


def correlogram(photo, Cm, K):
    """
    Get auto correlogram
    """
    X, Y, t = photo.shape

    colorsPercent = []

    # for k in K:

    k = K
    # print "k: ", k
    countColor = 0

    color = []
    for i in Cm:
        color.append(0)

    for x in range(0, X, int(round(X / 10))):
        for y in range(0, Y, int(round(Y / 10))):
            Ci = photo[x][y]
            Cn = getNeighbors(X, Y, x, y, k)
            for j in Cn:
                Cj = photo[j[0]][j[1]]

                for m in range(len(Cm)):
                    if np.array_equal(Cm[m], Ci) and np.array_equal(Cm[m], Cj):
                        countColor = countColor + 1
                        color[m] = color[m] + 1

    for i in range(len(color)):
        color[i] = float(color[i]) / countColor

    colorsPercent.append(color)

    return colorsPercent


def Resize(src, width, height):
    """
    Resize the img to given width and height
    """
    new = cv2.resize(src, (width, height))
    return new


#
# Image Descriptor functions
#

def autoCorrelogram(img):
    """
    The functions for computing color correlogram.
    To improve the performance, we consider to utilize
    color quantization to reduce image into 64 colors.
    So the K value of k-means should be 64.

    img:
     The numpy ndarray that describe an image in 3 channels.
    """
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 64
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # according to "Image Indexing Using Color Correlograms" paper
   # K = [i for i in range(1, 9, 2)]
    K = 4

    colors64 = unique(np.array(res))

    result = correlogram(res2, colors64, K)
    return result
