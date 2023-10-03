import cv2
import numpy as np

img = cv2.imread('./assets/image.jpg')

# hor= np.hstack((img,img))
# ver=np.vstack((img,img))
# cv2.imshow("hor",hor)
# cv2.imshow("ver",ver)

cv2.waitKey(0)
cv2.destroyAllWindows()