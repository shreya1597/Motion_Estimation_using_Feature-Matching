import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

start_time = time.time()



img = cv2.imread("E:\Sem2 HRW\Computer Vision\PROJECT\dog.jpg",0)
#cv2.imshow("Image",img)
#cv2.waitKey(0)
# Initiate STAR detector
orb = cv2.ORB_create()


# find the keypoints with ORB
kp = orb.detect(img,None)
print("Number of keypoints Detected:", len(kp), "\n")

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
print("--- %s seconds ---" % (time.time() - start_time))

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=0)
plt.imshow(img2),plt.show()
