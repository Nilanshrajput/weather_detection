import pandas as pd
import numpy as np
import numpy as np
from PIL import Image
import cv2


df1 = pd.read_csv('dfcloudy.csv')
df2 = pd.read_csv('dfcloudy_lbp.csv')
df = pd.concat([df1.iloc[0:, 1:], df2.iloc[0:, 1:]], axis=1, sort=False, ignore_index=True)
print df
df.to_csv('C:\Users\Abhi\dev\weather_detectcnn\Bow_sift-opencv-2.4.11(env-opencv)\lbp_bowsift_cld.csv')
print df.values


def get_dark_channel(I, w):
    """Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = I.shape
    padded = np.pad(I, ((w / 2, w / 2), (w / 2, w / 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(I, darkch, p):
    """Get the atmosphere light in the (RGB) image data.
    Parameters
    -----------
    I:      the M * N * 3 RGB image data ([0, L-1]) as numpy array
    darkch: the dark channel prior of the image as an M * N numpy array
    p:      percentage of pixels for estimating the atmosphere light
    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    # reference CVPR09, 4.4
    M, N = darkch.shape
    flatI = I.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:M * N * p]  # find top M * N * p indexes
    print 'atmosphere light region:', [(i / N, i % N) for i in searchidx]

    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)


I = cv2.imread('2.jpg')
w = 15
d = get_dark_channel(I, w)


print d
print d.shape
