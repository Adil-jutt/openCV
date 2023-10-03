import cv2
import numpy as np

img = cv2.imread('./assets/football.jpg')

width,height= 150,375
pts1= np.float32([[31,92],[158,90],[5,463],[155,465]])
pts2= np.float32([[0,0],[width,0],[0,height],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)

imgOut = cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow("output",imgOut)
cv2.waitKey(0)
cv2.destroyAllWindows()