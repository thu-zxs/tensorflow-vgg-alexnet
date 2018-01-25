import numpy as np
import cv2

x = np.loadtxt('test.csv', dtype=np.str, delimiter=',')
data = x[1:, :].astype(np.uint8)

tmp = data[0:5, :].reshape((-1, 28, 28))[:,:,:,np.newaxis]
print tmp.shape
cv2.imwrite('test.png', tmp[4,:])
