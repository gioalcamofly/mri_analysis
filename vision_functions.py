import cv2

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import copy

def getMaxConvex(cnts):

    if len(cnts) == 0:
        return 0

    maxIndex = 0
    for i in range(len(cnts)):
        if cv2.contourArea(cv2.convexHull(cnts[maxIndex])) < cv2.contourArea(cv2.convexHull(cnts[i])):
            maxIndex = i

    return cnts[maxIndex]

def getBiggestArea(cnts):

    topIndex = 0
    for i in range(len(cnts)):
        if cv2.contourArea(cnts[topIndex]) < cv2.contourArea(cnts[i]):
            topIndex = i

    return cnts[topIndex]

def getTopmost(cnt):

    return tuple(cnt[cnt[:, :, 1].argmax()][0])

def getCentroid(cnt):

    # cnt = getBiggerArea(cnts)

    M = cv2.moments(cnt)

    # calculate x,y coordinate of center

    if M["m00"] != 0:
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
    else:
        x, y = 0, 0

    return (x, y)


def getTotalArea(cnts):
    sum = 0

    for i in range(len(cnts)):
        sum = sum + cv2.contourArea(cnts[i])

    return sum

def getRightmost(cnt):

    return tuple(cnt[cnt[:, :, 0].argmin()][0])

def getLeftmost(cnt):

    return tuple(cnt[cnt[:, :, 0].argmax()][0])

def getExtremes(cnt):

    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmin()][0])

    return (leftmost, rightmost, topmost, bottommost)

def show_img(img, i):
    fig, axes = plt.subplots(1, 1, figsize=[12, 4])
    axes.imshow(img, cmap="gray", origin="lower")
    # plt.show()
    fig.savefig('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/out/img' + str(i) + '.png')
    plt.close(fig)


def isTooHigh(cnt, y):

    high_thresh = -20
    centroid = getCentroid(cnt)

    if centroid[1] > y + high_thresh:
        return True
    else:
        return False