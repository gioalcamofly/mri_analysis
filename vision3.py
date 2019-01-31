import cv2

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import copy
import vision_functions as vis
import math

count_left = []
count_right = []
gm_no_tl = []
prev_area = 0
prev_convex_area = []
prev_top = 0
prev_topmost = 0
started = False
last_try = False
detecting = True

def drawFrame(contours, hierarchy, slice_ori, slice_dest, color, c_type):

    global gm_no_tl

    for i in range(len(contours)):
        if detecting and checkLobe(contours[i], hierarchy[0][i], slice_ori) and c_type < 2:
            cv2.drawContours(slice_dest, contours, i, (255 * c_type, 255, 255 * pow((c_type - 1), 2)), 1)
        else:
            if c_type == 0:
                gm_no_tl.append(contours[i])
            cv2.drawContours(slice_dest, contours, i, color, 1)

    return slice_dest



def processFrame(slice):
    ret, thresh = cv2.threshold(slice, 127, 255, cv2.THRESH_OTSU)
    slice, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return slice, contours, hierarchy



def drawLine(slice, point, mult, slope, length, up):

    global prev_top
    global top

    if mult == 1:
        # prev_top = point
        top = point[1]

    slice = cv2.line(slice, (point[0], point[1] + up), (point[0] + length, point[1] + up - slope * mult), (0, 0, 0), 2)

    return slice



def checkLobe(cnt, hierarchy, slice):

    global count_left
    global count_right
    global prev_convex_area

    y, x = slice.shape

    leftmost, rightmost, topmost, bottommost = vis.getExtremes(cnt)

    len_thresh = x/7
    high_thresh = 0

    #Temporal Lobe shouldn't be inside biggest blob
    if started:
        dist = cv2.pointPolygonTest(prev_convex_area, vis.getCentroid(cnt), False)
        if dist >= 0:
            # print ("Area = " + str(cv2.contourArea(cnt)) + " regla = 1")
            return False

    #Temporal lobe shouldn't be higher than half of the image (y axis)
    # if topmost[1] > (y/2 + high_thresh):
    #     return False
    if vis.isTooHigh(cnt, y/2):
        # print ("Area = " + str(cv2.contourArea(cnt)) + " regla = 2")
        return False

    #Temporal lobe would be an external contour, so it shouldn't have parent or child
    # if hierarchy[3] != -1:
    #     return False

    # Temporal lobe should be at left or right of the image (not in the middle)
    if ((x / 2 - len_thresh) < leftmost[0]) and (rightmost[0] < (x / 2 + len_thresh)):
        # print ("Area = " + str(cv2.contourArea(cnt)) + " regla = 3")
        return False
    elif ((x / 2 - len_thresh) >= leftmost[0]):
        count_left.append(cnt)
    elif (rightmost[0] > (x / 2 + len_thresh)):
        count_right.append(cnt)

    return True

def errorLoop(slope, up, length):

    global count_left
    global prev_area
    global slice_tot
    global last_try
    global slice_tmp
    global gm_no_tl

    difference = (vis.getTotalArea(count_left) - prev_area)

    print ("Difference = " + str(difference))
    # slice_tmp = copy.deepcopy(slice_tot)

    while (difference < (-prev_area/3) or (vis.getTotalArea(count_left) < 1000 and prev_area > 1000)):

        if last_try:
            return []

        slice_gm_tmp = copy.deepcopy(slice_gm)

        if slope < 20:
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_top, 1, slope, 50, 8)
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_top, 1, slope, 50, 8)
            slope = slope + 1

        elif up <= 5 and length <= 33:
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, 25, up)
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, -25, up)
            up = up + 1

        elif length <= 33:
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, length, 0)
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, -length, 0)
            length = length + 1

        else:
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, length, 0)
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, -length, 0)
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_top, 1, 5, 50, 8)

            last_try = True

        # vis.show_img(slice_gm_tmp)

        slice_gm_tmp, contours_tmp, hierarchy_tmp = processFrame(slice_gm_tmp)

        slice_tmp = copy.deepcopy(slice_tot)

        count_left = []
        gm_no_tl = []

        slice_tmp = drawFrame(contours_tmp, hierarchy_tmp, slice_gm_tmp, slice_tmp, (0, 255, 0), 0)
        print ("len " + str(len(count_left)))

        difference = (vis.getTotalArea(count_left) - prev_area)
        print ("Difference = " + str(difference))


    return slice_tmp

def fillPrevs():

    global prev_top
    global prev_topmost
    global prev_area
    global prev_convex_area
    global count_left
    global count_right
    global gm_no_tl
    global slice_tmp


    if len(count_left) > 0:
        prev_top = vis.getCentroid(vis.getBiggerArea(count_left))
        slice_tmp = drawLine(slice_tmp, prev_top, 1, 6, 50, 8)
        prev_topmost = vis.getTopmost(vis.getBiggerArea(count_left))


    prev_area = vis.getTotalArea(count_left)
    prev_convex_area = vis.getMaxConvex(gm_no_tl)




########################################################################################################
########################################################################################################

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
slice_tl = []
slice_norm = []



#Load data in arrays

for i in range (i3t_data.shape[1] - 1, 0, -1):
    if tl_end_slice < i < tl_start_slice:
        slice_tl.append([i3t_data[:, i, :].T, i3tgm_data[:, i, :].T, i3twm_data[:, i, :].T, i3tcsf_data[:, i, :].T])
    else:
        slice_norm.append([i3t_data[:, i, :].T, i3tgm_data[:, i, :].T, i3twm_data[:, i, :].T, i3tcsf_data[:, i, :].T])


#Loop through all the slices

for i in range(len(slice_tl)):

    #Cleaning arrays

    count_right = []
    count_left = []
    gm_no_tl = []

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



    #Processing slices

    slice_gm, contours_gm, hierarchy_gm = processFrame(slice_gm)
    slice_wm, contours_wm, hierarchy_wm = processFrame(slice_wm)
    slice_csf, contours_csf, hierarchy_csf = processFrame(slice_csf)


    slice_tot = cv2.cvtColor(slice_tot, cv2.COLOR_GRAY2BGR)


    slice_tmp = copy.deepcopy(slice_tot)


    slice_tmp = drawFrame(contours_gm, hierarchy_gm, slice_gm, slice_tmp, (0, 255, 0), 0)

    if detecting:
        print len(slice_tmp)
        slice_tmp = errorLoop(5, 0, 25)
        print len(slice_tmp)

    if len(slice_tmp) == 0:
        detecting = False
        slice_tmp = copy.deepcopy(slice_tot)
        slice_tmp = drawFrame(contours_gm, hierarchy_gm, slice_gm, slice_tmp, (0, 255, 0), 0)


    fillPrevs()

    started = True

    slice_tmp = drawFrame(contours_wm, hierarchy_wm, slice_wm, slice_tmp, (255, 0, 0), 1)

    slice_tmp = drawFrame(contours_csf, hierarchy_csf, slice_csf, slice_tmp, (0, 0, 255), 2)


    slice_tot = slice_tmp

    fig, axes = plt.subplots(1, 1, figsize=[25, 7])
    axes.imshow(slice_tot, cmap="gray", origin="lower")
    plt.show()