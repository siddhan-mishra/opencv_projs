import cv2
print("OpenCV version:", cv2.__version__)

# Test SIFT (from xfeatures2d)
sift = cv2.SIFT_create()
print("SIFT loaded successfully!")
