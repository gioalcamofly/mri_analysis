import cv2

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
i3t = nib.load('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/I3TGM.img')
nine = cv2.imread('/home/giovanni/Documentos/UNED/vision_artificial/memoria_0/I3T/9.png')
print i3t.shape
print cv2.__version__
print i3t.affine.shape
i3t_data = i3t.get_fdata()

data = i3t_data[:, :, 25]

data = data.astype(np.uint16)
data = cv2.normalize(data, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

print data.dtype
print data.size
print data[81, 132]

cv2.imshow('prueba', data)
cv2.waitKey(0)
cv2.destroyAllWindows()



fig, axes = plt.subplots(1, 1, figsize=[18, 3])
axes.imshow(data, cmap="gray", origin="lower")
plt.show()