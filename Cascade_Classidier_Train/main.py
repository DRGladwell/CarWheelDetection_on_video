import cv2 as cv
import numpy as np
import argparse
from imutils import paths
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True,help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# take in the test images
testImages = list(paths.list_images(args["images"]))

# load the trained model
cascade_wheel = cv.CascadeClassifier('model/cascade.xml')



def draw_rectangles(haystack_img, rectangles):
    print("in draw rectangles")
    # these colors are actually BGR
    line_color = (0, 255, 0)
    line_type = cv.LINE_4

    for (x, y, w, h) in rectangles:
        # determine the box positions
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        # draw the box
        cv.rectangle(haystack_img, top_left, bottom_right, line_color, lineType=line_type)

    return haystack_img

for img in testImages:

    # get an updated image of the game
    screenshot = cv.imread(img)
    #screenshot = imutils.resize(screenshot, width=300)

    # do object detection
    rectangles = cascade_wheel.detectMultiScale(screenshot)

    # draw the detection results onto the original image
    detection_image = draw_rectangles(screenshot, rectangles)

    # display the images
    cv.imshow('Matches', detection_image)
    cv.waitKey(0)
