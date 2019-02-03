import cv2

import nibabel as nib
import numpy as np
import random
import matplotlib.pyplot as plt


#Load data

i3t = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3T.img')
i3tgm = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3TGM.img')
i3twm = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3TWM.img')
i3tcsf = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3TCSF.img')
i3t_data = i3t.get_fdata()
i3tgm_data = i3tgm.get_fdata()
i3twm_data = i3twm.get_fdata()
i3tcsf_data = i3tcsf.get_fdata()

#Get slices
cut = 280

slice_tot = i3t_data[:, cut, :].T
slice_gm = i3tgm_data[:, cut, :].T
slice_wm = i3twm_data[:, cut, :].T
slice_csf = i3tcsf_data[:, cut, :].T


#Convert to datatype understandable by OpenCV

slice_tot = (slice_tot/256).astype(np.uint8)

slice_tot = cv2.normalize(slice_tot, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
slice_gm = cv2.normalize(slice_gm, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
slice_wm = cv2.normalize(slice_wm, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
slice_csf = cv2.normalize(slice_csf, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

slice_gm = (slice_gm/256).astype(np.uint8)
slice_wm = (slice_wm/256).astype(np.uint8)
slice_csf = (slice_csf/256).astype(np.uint8)

#
# cv2.imshow('sin filtros', slice_gm)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#Conver to color and overlay the three slices of the different fluids

# slice_gm = cv2.cvtColor(slice_gm, cv2.COLOR_GRAY2BGR)
# slice_gm[:,:, 2] = 255
# slice_wm = cv2.cvtColor(slice_wm, cv2.COLOR_GRAY2BGR)
# slice_wm[:,:, 1] = 255
# slice_csf = cv2.cvtColor(slice_csf, cv2.COLOR_GRAY2BGR)
# slice_csf[:, :, 0] = 255
# slice_prueba = cv2.add(slice_gm, slice_csf)
# slice_prueba = cv2.add(slice_prueba, slice_wm)

#Image filtering

#Filtro normal
#slice_gm = cv2.filter2D(slice_gm, -1, np.ones((5, 5), np.float32)/25)

#Filtro de media
#slice_gm = cv2.blur(slice_gm, (5, 5))

#Filtro gaussiano
# slice_gm = cv2.GaussianBlur(slice_gm, (3, 3), 0)

#Filtro de mediana
#slice_gm = cv2.medianBlur(slice_gm, 3)

#Filtro bilateral
# slice_gm = cv2.bilateralFilter(slice_gm, 9, 100, 100)



#Transformaciones morfologicas

#Closing
#mask = np.ones((3,3), np.uint8)
# mask = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,5))
# slice_gm = cv2.morphologyEx(slice_gm, cv2.MORPH_ERODE, mask)

#Dilation + erosion
# slice_gm = cv2.erode(slice_gm, mask, iterations = 1)
# slice_gm = cv2.dilate(slice_gm, mask, iterations =1)
# mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))



#Transform images to binary images (Thresholding)



# kernel = np.ones((3,3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
#
# sure_bg = cv2.dilate(opening, kernel, iterations = 3)
#
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
#
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg, sure_fg)
#
# ret, markers = cv2.connectedComponents(sure_fg)
# markers = markers + 1
# markers[unknown == 255] = 0
# markers = cv2.watershed(slice_gm, markers)
#
# slice_gm[markers == -1] = [255, 0, 0]

ret, thresh_gm = cv2.threshold(slice_gm, 127, 255, cv2.THRESH_OTSU)
ret, thresh_wm = cv2.threshold(slice_wm, 127, 255, cv2.THRESH_OTSU)
ret, thresh_csf = cv2.threshold(slice_csf, 127, 255, cv2.THRESH_OTSU)

slice_gm, contours_gm, hierarchy_gm = cv2.findContours(thresh_gm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
slice_wm, contours_wm, hierarchy_wm = cv2.findContours(thresh_wm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_csf, contours_csf, hierarchy_csf = cv2.findContours(thresh_csf, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# print hierarchy
# print len(contours)
slice_tot = cv2.cvtColor(slice_tot, cv2.COLOR_GRAY2BGR)
# slice_gm = cv2.cvtColor(slice_gm, cv2.COLOR_GRAY2BGR)
# slice_wm = cv2.cvtColor(slice_wm, cv2.COLOR_GRAY2BGR)
# slice_csf = cv2.cvtColor(slice_csf, cv2.COLOR_GRAY2BGR)
print slice_tot.shape

min_thresh = 0

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
    y, x = slice.shape
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmax()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmin()][0])

    len_thresh = x/7
    high_thresh = 10


    if ((x/2 - len_thresh) < leftmost[0]) and (rightmost[0] < (x/2 + len_thresh)):
        return False

    #Temporal lobe should be on the same half of the image
    # if (rightmost[0] > x/2) and (leftmost[0] < x/2):
    #     return False

    #Temporal lobe shouldn't be higher than half of the image (y axis)
    if topmost[1] > (y/2 - high_thresh):
        return False

    #Temporal lobe would be an external contour, so it shouldn't have parent or child
    if hierarchy[2] != -1 or hierarchy[3] != -1:
        return False

    print (y/2 - high_thresh)
    print bottommost
    # print ("borde inferior = " + str((x/2 - len_thresh)))
    # print ("borde superior = " + str((x/2 + len_thresh)))
    # print hierarchy[3]
    # print ("x = " + str(x))
    # print ("y = " + str(y))
    # print ("leftmost = " + str(leftmost))
    # print ("rightmost = " + str(rightmost))
    # print ("topmost = " + str(topmost))
    # print ("topmost (y) = " + str(topmost[1]))
    return True

for i in range(len(contours_gm)):
    # color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    #cv2.drawContours(slice_tot, contours, i, color, cv2.FILLED)
    if cv2.contourArea(contours_gm[i]) > min_thresh:
        if checkLobe(contours_gm[i], hierarchy_gm[0][i], slice_gm):
            cv2.drawContours(slice_tot, contours_gm, i, (0,255,255), 1)
            # cv2.drawContours(slice_tot, [cv2.convexHull(contours_gm[i])], -1, (123, 25, 200), 1)
        else:
            cv2.drawContours(slice_tot, contours_gm, i, (0, 255, 0), 1)
            # cv2.drawContours(slice_tot, [cv2.convexHull(contours_gm[i])], -1, (123, 25, 200), 1)
        drawConvexDefects(contours_gm[i])
        # cv2.drawContours(slice_gm, contours_gm, i, (0, 255, 0), 2)
        # cv2.drawContours(slice_tot, contours_gm, i, color, 2)
        # print cv2.contourArea(contours_gm[i])

for i in range(len(contours_wm)):
    if cv2.contourArea(contours_wm[i]) > min_thresh:
        if checkLobe(contours_wm[i], hierarchy_wm[0][i], slice_wm):
            cv2.drawContours(slice_tot, contours_wm, i, (255,255,0), 1)
        else:
            cv2.drawContours(slice_tot, contours_wm, i, (255, 0, 0), 1)

# for i in range(len(contours_csf)):
#     if cv2.contourArea(contours_csf[i]) > min_thresh:
#         cv2.drawContours(slice_tot, contours_csf, i, (0, 0, 255), 1)

# cv2.imshow('prueba', slice_tot)
# # cv2.imshow('prueba', slice_gm)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Show the image rotated

# rows, cols, channels = slice_prueba.shape
# M = cv2.getRotationMatrix2D((cols/2, rows/2),180,1)
# slice_prueba = cv2.warpAffine(slice_prueba, M, (cols, rows))
# cv2.imshow('prueba', slice_prueba)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Show the image with Matplotlib

fig, axes = plt.subplots(1, 1, figsize=[25, 7])
axes.imshow(slice_tot, cmap="gray", origin="lower")
# axes.imshow(slice_gm, origin="lower")
plt.show()