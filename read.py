import cv2 as cv
#reading image
# img=cv.imread(r'Photos\foodss.png')
# cv.imshow('Food',img)
# cv.waitKey(0)
# Reading videos
capture=cv.VideoCapture(0)
while True:
    isTrue, frame=capture.read()
    cv.imshow('Video',frame)
    if cv.waitKey(2) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()