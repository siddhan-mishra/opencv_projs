import cv2

img= cv2.imread('pic.png',cv2.IMREAD_GRAYSCALE)
cv2.imwrite('pic2.png', img)
print(img)
#greyscalemode

 

