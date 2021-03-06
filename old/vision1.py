import cv2

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import copy

count_left = []
count_right = []
prev_area = 0
prev_top = 0
top = 0


def getBiggerArea(cnts):

    topIndex = 0
    for i in range(len(cnts)):
        if cv2.contourArea(cnts[topIndex]) < cv2.contourArea(cnts[i]):
            topIndex = i

    return cnts[topIndex]

def getTopmost(cnts):

    topmosts = []
    topIndex = 0


    for i in range (len(cnts)):
        cnt = copy.deepcopy(cnts[i])
        topmost = tuple(cnt[cnt[:, :, 1].argmax()][0])
        topmosts.append(topmost[1])

    for i in range(len(topmosts)):
        if topmosts[topIndex] < topmosts[i]:
            topIndex = i

    return cnts[topIndex]

def getCentroid(cnts):

    cnt = getBiggerArea(cnts)

    M = cv2.moments(cnt)

    # calculate x,y coordinate of center

    if M["m00"] != 0:
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
    else:
        x, y = 0, 0

    return (x, y)

#
# def drawLine(slice, cnt, mult, slope):
#
#     global prev_top
#     global top
#
#     # slope = 6
#     length = 50
#     topmost = tuple(cnt[cnt[:, :, 1].argmax()][0])
#
#     if mult == 1:
#         prev_top = cnt
#         top = topmost[1]
#
#     if mult == 1:
#         slice = cv2.line(slice, (topmost[0], topmost[1]), (topmost[0] + length, topmost[1] - slope * mult), (0, 0, 0), 2)
#     else:
#         slice = cv2.line(slice, (topmost[0] - length, topmost[1] + slope * mult), (topmost[0], topmost[1]), (0, 0, 0), 2)
#
#     return slice

def drawLine(slice, point, mult, slope):

    global prev_top
    global top

    length = 50
    up = 8

    if mult == 1:
        prev_top = point
        top = point[1]

    # slice = cv2.line(slice, (point[0] - length, point[1] + up + slope * mult), (point[0] + length, point[1] + up - slope * mult), (0, 0, 0), 2)
    slice = cv2.line(slice, (point[0], point[1] + up), (point[0] + length, point[1] + up - slope * mult), (0, 0, 0), 2)

    return slice

def getTotalArea(cnts):
    sum = 0

    for i in range(len(cnts)):
        sum = sum + cv2.contourArea(cnts[i])

    return sum

def getExtremes(cnt):

    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmin()][0])

    return (leftmost, rightmost, topmost, bottommost)

def show_img(img):
    fig, axes = plt.subplots(1, 1, figsize=[25, 7])
    axes.imshow(img, cmap="gray", origin="lower")
    plt.show()

def drawConvexDefects(cnt):
    if cv2.isContourConvex(cnt):
        return None

    hull = cv2.convexHull(cnt, returnPoints= False)
    defects = cv2.convexityDefects(cnt, hull)


    for i in range(defects.shape[0]):
        s, e, f, d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(slice_tot, start, end, [125, 255, 76], 2)
        cv2.circle(slice_tot, far, 1, [255, 0, 0], -1)

def checkLobe(cnt, hierarchy, slice):

    global count_left
    global count_right
    y, x = slice.shape

    leftmost, rightmost, topmost, bottommost = getExtremes(cnt)

    len_thresh = x/7
    high_thresh = 0


    #Temporal lobe shouldn't be higher than half of the image (y axis)
    if topmost[1] > (y/2 + high_thresh):
        return False

    #Temporal lobe would be an external contour, so it shouldn't have parent or child
    # if hierarchy[2] != -1 or hierarchy[3] != -1:
    #     print hierarchy
    #     return False

    # Temporal lobe should be at left or right of the image (not in the middle)
    if ((x / 2 - len_thresh) < leftmost[0]) and (rightmost[0] < (x / 2 + len_thresh)):
        return False
    elif ((x / 2 - len_thresh) >= leftmost[0]):
        count_left.append(cnt)
    elif (rightmost[0] > (x / 2 + len_thresh)):
        count_right.append(cnt)

    return True

#Load data

i3t = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3T.img')
i3tgm = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3TGM.img')
i3twm = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3TWM.img')
i3tcsf = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3TCSF.img')
i3t_data = i3t.get_fdata()
i3tgm_data = i3tgm.get_fdata()
i3twm_data = i3twm.get_fdata()
i3tcsf_data = i3tcsf.get_fdata()


#Percentages of the brain where the temporal lobe should be

tl_start = 0.75 #Approx at 25% (down-up) the TL starts to appear
tl_end = 0.60 #Approx at 40% (down-up) the TL starts to merge with the rest of the brain

tl_start_slice = int(i3t_data.shape[1] * tl_start)
tl_end_slice = int(i3t_data.shape[1] * tl_end)
slice_tl = ([])
slice_norm = ([])

for i in range (i3t_data.shape[1] - 1, 0, -1):
    if tl_end_slice < i < tl_start_slice:
        slice_tl.append([i3t_data[:, i, :].T, i3tgm_data[:, i, :].T, i3twm_data[:, i, :].T, i3tcsf_data[:, i, :].T])
    else:
        slice_norm.append([i3t_data[:, i, :].T, i3tgm_data[:, i, :].T, i3twm_data[:, i, :].T, i3tcsf_data[:, i, :].T])


#Get slices
# cut = 294

# slice_tot = i3t_data[:, cut, :].T
# slice_gm = i3tgm_data[:, cut, :].T
# slice_wm = i3twm_data[:, cut, :].T
# slice_csf = i3tcsf_data[:, cut, :].T

for i in range(12, len(slice_tl)):

    count_right = []
    count_left = []

    slice_tot = slice_tl[i][0]
    slice_gm = slice_tl[i][1]
    slice_wm = slice_tl[i][2]
    slice_csf = slice_tl[i][3]



    #Convert to datatype understandable by OpenCV

    slice_tot = (slice_tot/256).astype(np.uint8)

    slice_tot = cv2.normalize(slice_tot, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    slice_gm = cv2.normalize(slice_gm, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    slice_wm = cv2.normalize(slice_wm, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    slice_csf = cv2.normalize(slice_csf, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

    slice_gm = (slice_gm/256).astype(np.uint8)
    slice_wm = (slice_wm/256).astype(np.uint8)
    slice_csf = (slice_csf/256).astype(np.uint8)

    # test = slice_tl[i][1]
    # show_img(test)

    #Transform images to binary images (Thresholding)

    ret, thresh_gm = cv2.threshold(slice_gm, 10, 255, cv2.THRESH_OTSU)
    ret, thresh_wm = cv2.threshold(slice_wm, 127, 255, cv2.THRESH_OTSU)
    ret, thresh_csf = cv2.threshold(slice_csf, 127, 255, cv2.THRESH_OTSU)

    # thresh_gm = cv2.medianBlur(thresh_gm, 3)
    # show_img(thresh_gm)
    slice_gm, contours_gm, hierarchy_gm = cv2.findContours(thresh_gm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slice_wm, contours_wm, hierarchy_wm = cv2.findContours(thresh_wm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_csf, contours_csf, hierarchy_csf = cv2.findContours(thresh_csf, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    # slice_tot = cv2.cvtColor(slice_tot, cv2.COLOR_GRAY2BGR)
    slice_tot = cv2.cvtColor(slice_tot, cv2.COLOR_GRAY2BGR)
    # slice_tmp = slice_tot[:, :, :]
    slice_tmp = copy.deepcopy(slice_tot)
    min_thresh = 0

    for i in range(len(contours_gm)):
        if cv2.contourArea(contours_gm[i]) > min_thresh:
            if checkLobe(contours_gm[i], hierarchy_gm[0][i], slice_gm):
                cv2.drawContours(slice_tmp, contours_gm, i, (0,255,255), 1)
            else:
                cv2.drawContours(slice_tmp, contours_gm, i, (0, 255, 0), 1)
            # drawConvexDefects(contours_gm[i])

    # if len(count_left) > 1:
    #     slice_tmp = drawLine(slice_tmp, getTopmost(count_left), 1)
    #
    # if len(count_right) > 1:
    #     slice_tmp = drawLine(slice_tmp, getTopmost(count_right), -1)

    # print ("Prev area = " + str(prev_area))
    # print ("Total area = " + str(getTotalArea(count_left)))
    # print ("Total - prev = " + str(getTotalArea(count_left) - prev_area))
    slope = 5
    while (getTotalArea(count_left) - prev_area) < (-50):
        #TL hasn't been correctly detected
        print 'entro'
        show_img(slice_gm)
        slice_gm_tmp = copy.deepcopy(slice_gm)
        slice_gm_tmp = drawLine(slice_gm_tmp, prev_top, 1, slope)
        slope = slope + 1
        show_img(slice_gm_tmp)
        ret, thresh_gm = cv2.threshold(slice_gm_tmp, 10, 255, cv2.THRESH_OTSU)
        slice_gm_tmp, contours_gm, hierarchy_gm = cv2.findContours(thresh_gm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        slice_tmp = copy.deepcopy(slice_tot)
        count_left = []
        for i in range(len(contours_gm)):
            if checkLobe(contours_gm[i], hierarchy_gm[0][i], slice_gm_tmp):
                cv2.drawContours(slice_tmp, contours_gm, i, (0, 255, 255), 1)
            else:
                cv2.drawContours(slice_tmp, contours_gm, i, (0, 255, 0), 1)

    if len(count_left) > 0:
        # slice_tmp = drawLine(slice_tmp, getTopmost(count_left), 1, 6)
        slice_tmp = drawLine(slice_tmp, getCentroid(count_left), 1, 6)
    if len(count_right) > 0:
        # slice_tmp = drawLine(slice_tmp, getTopmost(count_right), -1, 6)
        slice_tmp = drawLine(slice_tmp, getCentroid(count_right), -1, 6)
    prev_area = getTotalArea(count_left)

    # slice_tot = drawLine(slice_tot, count_right, -1)
    # if len(count_left) > 1:
    #     left_area = getArea(count_left)
    # if len(count_right) > 1:
    #     right_area = getArea(count_right)
    #     print ("right area = " + str(right_area))

    for i in range(len(contours_wm)):
        if cv2.contourArea(contours_wm[i]) > min_thresh:
            if checkLobe(contours_wm[i], hierarchy_wm[0][i], slice_wm):
                cv2.drawContours(slice_tmp, contours_wm, i, (255,255,0), 1)
            else:
                cv2.drawContours(slice_tmp, contours_wm, i, (255, 0, 0), 1)

    # for i in range(len(contours_csf)):
    #     if cv2.contourArea(contours_csf[i]) > min_thresh:
    #         cv2.drawContours(slice_tot, contours_csf, i, (0, 0, 255), 1)


    #Show the image with Matplotlib

    print ("Count left = " + str(len(count_left)))
    print ("Count right = " + str(len(count_right)))

    slice_tot = slice_tmp

    fig, axes = plt.subplots(1, 1, figsize=[25, 7])
    axes.imshow(slice_tot, cmap="gray", origin="lower")
    # axes.imshow(slice_gm, origin="lower")
    plt.show()