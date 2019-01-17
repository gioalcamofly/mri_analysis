import nibabel as nib
import cv2
from nibabel.affines import apply_affine
import numpy as np
i3t = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3T.img')
prueba = cv2.imread('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3T.img')
i3tgm = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3TGM.img')
i3twm = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3TWM.img')
i3tcsf = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3TCSF.img')
i3t_data = i3t.get_fdata()
i3tgm_data = i3tgm.get_fdata()
i3twm_data = i3twm.get_fdata()
i3tcsf_data = i3tcsf.get_fdata()

import matplotlib.pyplot as plt

#def show_slices(slices):
#    fig, axes = plt.subplots(1, 6, figsize=[18, 3])
#    for i, slice in enumerate(slices):
#        axes[i].imshow(slice.T, cmap="gray", origin="lower")
#        axes[i].set_xticks([])
#        axes[i].set_yticks([])
#    fig.subplots_adjust(wspace=0, hspace=0)

#def show_slices(slices):
#    fig, axes = plt.subplots(2, 16, figsize=[18, 4])
#    for i, slice in enumerate(slices):
#        for j, frame in enumerate(slice):
#            axes[i, j].imshow(frame.T, cmap="gray", origin="lower")
#    fig.subplots_adjust(wspace=0, hspace=0.1)

slice = 0
slice_1 = []
slice_2 = []
slice_3 = []
slice_4 = []
cv2.imshow('prueba', i3t_data[:, :, 50])
for _ in range(16):
    slice_1.append(i3t_data[:, :, slice])
    slice_2.append(i3tgm_data[:, :, slice])
    slice_3.append(i3twm_data[:, :, slice])
    slice_4.append(i3tcsf_data[:, :, slice])
    slice += 10

#show_slices([slice_1, slice_2, slice_3, slice_4])
    #show_slices([slice_3, slice_4, slice_5])

    #show_slices([slice_6, slice_7, slice_8])

    #show_slices([slice_9, slice_10, slice_12])



plt.suptitle("PRUEBA")
plt.show()