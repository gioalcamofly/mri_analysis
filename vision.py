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

slice_tot = i3t_data[:, 150, :].T
slice_gm = i3tgm_data[:, 150, :].T
slice_wm = i3twm_data[:, 150, :].T
slice_csf = i3tcsf_data[:, 150, :].T


#Convert to datatype understandable by OpenCV

slice_tot = (slice_tot/256).astype(np.uint8)

slice_tot = cv2.normalize(slice_tot, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
slice_gm = cv2.normalize(slice_gm, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
slice_wm = cv2.normalize(slice_wm, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
slice_csf = cv2.normalize(slice_csf, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

slice_gm = (slice_gm/256).astype(np.uint8)
slice_wm = (slice_wm/256).astype(np.uint8)
slice_csf = (slice_csf/256).astype(np.uint8)

#Conver to color and overlay the three slices of the different fluids

# slice_gm = cv2.cvtColor(slice_gm, cv2.COLOR_GRAY2BGR)
# slice_gm[:,:, 2] = 255
# slice_wm = cv2.cvtColor(slice_wm, cv2.COLOR_GRAY2BGR)
# slice_wm[:,:, 1] = 255
# slice_csf = cv2.cvtColor(slice_csf, cv2.COLOR_GRAY2BGR)
# slice_csf[:, :, 0] = 255
# slice_prueba = cv2.add(slice_gm, slice_csf)
# slice_prueba = cv2.add(slice_prueba, slice_wm)


#Transform images to binary images (Thresholding)

ret, thresh = cv2.threshold(slice_gm, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print len(contours)
slice_gm = cv2.cvtColor(slice_gm, cv2.COLOR_GRAY2BGR)
for i in range(len(contours)):
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    cv2.drawContours(slice_gm, contours, i, color, 3)
    print cv2.contourArea(contours[i])

cv2.imshow('prueba', slice_gm)
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

#fig, axes = plt.subplots(1, 1, figsize=[18, 3])
#axes.imshow(slice_tot, cmap="gray", origin="lower")
#plt.show()