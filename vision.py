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
cut = 250

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

# fig, axes = plt.subplots(1, 1, figsize=[25, 7])
# axes.imshow(slice_gm, cmap="gray", origin="lower")
# plt.show()

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
# mask = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
# slice_gm = cv2.morphologyEx(slice_gm, cv2.MORPH_CLOSE, mask)

#Dilation + erosion
# slice_gm = cv2.dilate(slice_gm, mask, iterations = 1)
# mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# slice_gm = cv2.erode(slice_gm, mask, iterations = 1)


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

slice_gm, contours_gm, hierarchy_gm = cv2.findContours(thresh_gm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
slice_wm, contours_wm, hierarchy_wm = cv2.findContours(thresh_wm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_csf, contours_csf, hierarchy_csf = cv2.findContours(thresh_csf, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# print hierarchy
# print len(contours)
slice_tot = cv2.cvtColor(slice_tot, cv2.COLOR_GRAY2BGR)
# slice_gm = cv2.cvtColor(slice_gm, cv2.COLOR_GRAY2BGR)
# slice_wm = cv2.cvtColor(slice_wm, cv2.COLOR_GRAY2BGR)
# slice_csf = cv2.cvtColor(slice_csf, cv2.COLOR_GRAY2BGR)

min_thresh = 0

for i in range(len(contours_gm)):
    # color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    #cv2.drawContours(slice_tot, contours, i, color, cv2.FILLED)
    if cv2.contourArea(contours_gm[i]) > min_thresh:
        #cv2.drawContours(slice_gm, contours_gm, i, (0, 255, 0), 2)
        cv2.drawContours(slice_tot, contours_gm, i, (0,255,0), 2)
        # print cv2.contourArea(contours_gm[i])

for i in range(len(contours_wm)):
    if cv2.contourArea(contours_wm[i]) > min_thresh:
        cv2.drawContours(slice_tot, contours_wm, i, (255, 0, 0), 2)

for i in range(len(contours_csf)):
    if cv2.contourArea(contours_csf[i]) > min_thresh:
        cv2.drawContours(slice_tot, contours_csf, i, (0, 0, 255), 2)

cv2.imshow('prueba', slice_tot)
# cv2.imshow('prueba', slice_gm)
cv2.waitKey(0)
cv2.destroyAllWindows()


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
plt.show()