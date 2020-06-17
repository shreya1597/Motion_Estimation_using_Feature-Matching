# Imports
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import os
import pickle

# Open and convert a input
# image from BGR to GRAYSCALE
image = cv.imread(filename = '/home/shreya/Sem2/CV/dog.jpg',
                   flags = cv.IMREAD_GRAYSCALE)

# BRISK is a feature detector and descriptor

# Initiate BRISK detector
BRISK = cv.BRISK_create()

# Find the keypoints with BRISK
keypoints = BRISK.detect(image, None)

# Print number of keypoints detected
print("Number of keypoints Detected:", len(keypoints), "\n")

# Save Keypoints to a file

index = []

for point in keypoints:
    temp = (point.pt,
            point.size,
            point.angle,
            point.response,
            point.octave, 
            point.class_id)
    
    index.append(temp)
    
# File name
filename = "/home/shreya/Sem2/CV/Feature_Matching/Output/BRISK-keypoints.txt"

# Delete a file if it exists
if os.path.exists(filename):
    os.remove(filename)

# Open a file
file = open(filename, "wb")

# Write 
file.write(pickle.dumps(index))

# Close a file
file.close()

# Compute the descriptors with BRISK
keypoints, descriptors = BRISK.compute(image, keypoints)

# Print the descriptor size in bytes
print("Size of Descriptor:", BRISK.descriptorSize(), "\n")

# Print the descriptor type
print("Type of Descriptor:", BRISK.descriptorType(), "\n")

# Print the default norm type
print("Default Norm Type:", BRISK.defaultNorm(), "\n")

# Print shape of descriptor
print("Shape of Descriptor:", descriptors.shape, "\n")

# Draw only 50 keypoints on input image
image = cv.drawKeypoints(image = image,
                         keypoints = keypoints[:50],
                         outImage = None,
                         flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Plot input image

# Turn interactive plotting off
plt.ioff()

# Create a new figure
plt.figure()
plt.axis('off')
plt.imshow(image)
plt.show()

plt.imsave(fname = '/home/shreya/Sem2/CV/Feature_Matching/Figures/feature-detection-BRISK.png',
           arr = image,
           dpi = 300)

# Close it
plt.close()