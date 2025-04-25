import cv2 as cv
img=cv.imread(r'Photos\foodss.png')
cv.imshow('Food',img)

#rescaling
def rescaleFrame(frame,scale=0.75):
    #image,videos and live videos
    width=int(frame.shape[1]*0.2)
    height=int(frame.shape[0]*0.3)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
resized_photo=rescaleFrame(img)
cv.imshow('Resized photo',resized_photo)
cv.waitKey(0)
#Reading videos
# capture=cv.VideoCapture(r'C:\Users\ship2\OneDrive\Pictures\WhatsApp Video\VID-20191026-WA0000.mp4')
# while True:
#     isTrue, frame=capture.read()

#     frame_resized=rescaleFrame(frame)

#     cv.imshow('Video',frame)
#     cv.imshow('Video Resized',frame_resized)
    
#     if cv.waitKey(2) & 0xFF==ord('d'):
#         break
# capture.release()
# cv.destroyAllWindows()