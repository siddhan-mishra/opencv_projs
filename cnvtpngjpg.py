import cv2
import sys

image = cv2.imread('cnvtpic.png')
if image is None:
    print('Failed to read image from file')
    sys.exit(1)

success = cv2.imwrite('cnvtpic.jpg', image)
if not success:
    print('Failed to write image to file')
    sys.exit(1)

# Using imread to read our image and then imwrite to convert our read png to jpg file check in the file 