from imutils.perspective import four_point_transform
from imutils import contours
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to input directory of images")
args = vars((ap.parse_args()))

imgWidth = 800
# grab the paths to the images
imagePaths = list(paths.list_images(args["images"]))

for image in imagePaths:
    result = []
    # load the image, convert it to grayscale, blur it
    # slightly, the find the edges
    image = cv2.imread(image)
    resized = imutils.resize(image, imgWidth)
    resizedFinal = resized.copy()
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # find all the 'black' shapes in the image
    thresh2 = cv2.threshold(blurred, 65, 255, cv2.THRESH_BINARY_INV)[1]
    thresh2 = cv2.dilate(thresh2, None, iterations=2)
    thresh2 = cv2.erode(thresh2, None, iterations=2)
    edged = cv2.Canny(thresh2, 30, 60 )
    #cv2.imshow("thresh", thresh2)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)
    result.append(cv2.cvtColor(edged,cv2.COLOR_GRAY2RGB))


    circlesFound = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 0.5, 40,
                  param1=60,
                  param2=20,
                  minRadius=100,
                  maxRadius=400)

    if circlesFound is not None:
        # of no circle is found this will return a none and attribute error will apear for .shape
        #print(circlesFound)
        #print(circlesFound.shape)
        circlesFound = np.uint16(np.around(circlesFound))

        # show the top 5 candidates, the results should be ordered by size?
        for i in circlesFound[0,:1]:
            cv2.circle(resized,(i[0],i[1]),i[2],(0,0,255),5)

        # construct the montages for the images
        result.append(resized)
        montages = build_montages(result, (gray.shape[1], gray.shape[0]), (2,1))
        for montage in montages:
            cv2.imshow("where are the circles?", montage)
            cv2.waitKey(0)
            cv2.destroyWindow("where are the circles?")
    else:
        print("[INFO] no reasonable wheel was found")