import cv2
import numpy 
# This code we discuss the property of image and the cvtcolor how it converts a 2d img black and white to 3d image 
# And we also discuss shape property discussed in obisidan 

img = numpy.zeros((5, 3), dtype=numpy.uint8)
print(img)
print ("--------------------------------")
img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(img2)
print ("--------------------------------")
print(img.shape)