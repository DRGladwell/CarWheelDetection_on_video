from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars((ap.parse_args()))

imgWidth = 800
# load the image, convert it to grayscale, blur it
# slightly, the find the edges
image = cv2.imread(args["image"])
ratio = image.shape[0] / imgWidth
resized = imutils.resize(image, imgWidth)
resizedFinal = resized.copy()
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
thresh = cv2.threshold(blurred, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
edged = cv2.Canny(thresh, 40, 60 )
cv2.imshow("thresh", thresh)
cv2.imshow("edged", edged)
cv2.waitKey(0)


# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
innerRim = []
innerRimPos = ()
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # loop over the sorted contours
    for c in cnts[:5]:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            cv2.drawContours(resizedFinal, c, -1, (0, 0, 255), 7)
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))
            innerRimPos = (cX,cY)
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            cv2.circle(resizedFinal,(cX,cY),5,(0,0,255),5)
            # show the output image
            innerRim = c
            cv2.imshow("drawContours", resizedFinal)
            cv2.waitKey(0)
            break



# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh
roi = resized.copy()
roi = cv2.bitwise_and(roi, roi, mask=mask)

# get the size of new region of interest
(x, y, w, h) = cv2.boundingRect(innerRim)

# we find the bounding box of the wheel
left = int(innerRimPos[0]-w/2)
right = int(innerRimPos[0]+w/2)
bottom = int(innerRimPos[1]+h/2)
top = int(innerRimPos[1]-h/2)

# to use for reating contour of region of interest
# cv2.rectangle(output, (left, top), (right, bottom), (0, 0, 255), 2)

# we consider the inside of the bounding box to be the region of interest
roi = roi[top:bottom, left:right]

cv2.imshow("roi", roi)
cv2.waitKey(0)

# We will now work on the new Region Of Interest
# We aim to find a maximum of 5 bolts and display the number of bolts

gray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blurred2 = cv2.GaussianBlur(gray2, (7, 7), 0)
thresh2 = cv2.threshold(blurred2, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

edged2 = cv2.Canny(thresh2, 40, 60 )
cv2.imshow("newThresh", thresh2)
cv2.imshow("newEdged", edged2)
cv2.waitKey(0)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnt2 = cv2.findContours(edged2.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnt2 = imutils.grab_contours(cnt2)
innerRim = []
innerRimPos = ()
if len(cnt2) > 0:
    # sort the contours according to their size in
    # descending order
    cnt2 = sorted(cnt2, key=cv2.contourArea, reverse=True)
    # loop over the sorted contours
    for c in cnt2[:20]:
        print("looking for car bolts")
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w <= 50 and h <=50 and w >=15 and h >= 15 and ar >= 0.9 and ar <= 1.1:
            print("we have found some small objects")
            cv2.drawContours(roi, c, -1, (0, 0, 255), 3)
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))
            #innerRimPos = (cX,cY)
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            cv2.circle(roi,(cX,cY),5,(0,0,255),3)
            #innerRim = c


            cv2.imshow("BOLTS!", roi)
            cv2.waitKey(0)
