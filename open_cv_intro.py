import cv2
import numpy as np

#============================================================================================

#img = cv2.imread('./images/image.jpg') #(path, -1=color, 0=grayscale, 1=not transparent background)
# img2 = cv2.resize(img,(800,800))
#img3 = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
# cv2.imshow('Image',img)
# cv2.waitKey(0) #wait until key is pressed
# cv2.imshow('Image',img2)
# cv2.waitKey(0) #wait until key is pressed
# cv2.imshow('Image',img3)
# cv2.waitKey(0) #wait until key is pressed
# cv2.destroyAllWindows()
#cv2.imwrite('./images/new_img.jpg',img3)

#import random
#img = cv2.imread('./images/image.jpg') #(path, -1=color, 0=grayscale, 1=not transparent background)
#for i in range(100):
    #for j in range(img.shape[1]):
        #img[i][j]=[random.randint(0,1),random.randint(0,255),random.randint(0,255)]

#replacing part of image
#temp_img= img[0:50,50:150]
#img[50:100,0:100] = temp_img

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#===============================================================================


# cap = cv2.VideoCapture('.\\assets\\Emotions_detection.mp4') #0 to integrate web cam

# while True:
#     ret, frame = cap.read() #video frame
#     width = int(cap.get(3)) # 3 is width property
#     height = int(cap.get(4)) #4 is height property

#     image = np.zeros(frame.shape,np.uint8)
#     small_frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
#     image[:height//2, :width//2]= cv2.rotate(small_frame,cv2.ROTATE_180)
#     image[height//2:, :width//2]= small_frame
#     image[:height//2, width//2:]=  cv2.rotate(small_frame,cv2.ROTATE_180)
#     image[height//2:, width//2:]= small_frame


#     cv2.imshow('frame',image)
#     if cv2.waitKey(1)==ord('q'): #convert into ASCII
#         break

# cap.release()
# cv2.destroyAllWindows()

#=============================================================================================

# cap = cv2.VideoCapture('.\\assets\\Emotions_detection.mp4')
# while True:
#     ret,frame= cap.read()
#     width=int(cap.get(3))
#     height=int(cap.get(4))

#     img = cv2.line(frame,(0,0),(width,height),(255,0,0),10) #(frame, starting point, ending point, color, thickness)
#     img = cv2.line(frame,(0,height),(width,0),(135,200,170),10) #(frame, starting point, ending point, color, thickness)
#     img= cv2.rectangle(img,(100,100),(200,200),(100,100,100),-1)
#     img= cv2.circle(img,(400,400),50,(0,0,255),-1) # (img, middle point, radius,color, -1(fill))
#     font=cv2.FONT_HERSHEY_PLAIN
#     img = cv2.putText(img, 'Hello world', (200, 400), font,4,(255,0,0),5, cv2.LINE_AA) #(img, text, textPosition bottom left, font, font scale, color,thickness, type)

#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#=====================================================================

# cap = cv2.VideoCapture('.\\assets\\Emotions_detection.mp4')

# while True:
#     ret, frame = cap.read()
#     width = int(cap.get(3))
#     height = int(cap.get(4))

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_blue= np.array([90,50,50])
#     upper_blue=np.array([130,255,255])
    
#     mask = cv2.inRange(hsv,lower_blue,upper_blue)
#     result = cv2.bitwise_and(frame,frame,mask=mask)
#     cv2.imshow('frame',result)
#     if(cv2.waitKey(1)== ord('q')):
#         break

# cap.release()
# cv2.destroyAllWindows()

# BGR_COLOR = np.array([[[255,0,0]]])
# x = cv2.cvtColor([[[255,0,0]]], cv2.COLOR_BGR2HSV)
# x[0][0]

#=============================================================

# img = cv2.imread('.\\assets\\chess.png')
# img = cv2.resize(img,(0,0),fx=2,fy=2)
# gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(gray,100,0.01,10) #(img, numbers of corners, sharpness, eucilidean)corner detection
# corners = np.int0(corners) #float to int
# for corner in corners:
#     x,y= corner.ravel() # corner will be flatten [[100,44]]-> [100,44]
#     cv2.circle(img,(x,y), 5,(255,0,0),-1)

# for i in range(len(corners)):
#     for j in range(i+1,len(corners)):
#         corner1=tuple(corners[i][0]) #ravel() 
#         corner2=tuple(corners[j][0])
#         color=tuple(map(lambda x: int(x),np.random.randint(0,255,size=3))) #lambda function converting every element of array into python integers
#         cv2.line(img,corner1,corner2,color,1 )

# cv2.imshow('frame',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#===================================================================================



