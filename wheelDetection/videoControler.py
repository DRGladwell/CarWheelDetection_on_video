import sys

sys.path.append('/home/david/PycharmProjects/DeepLearning/')

import cv2 as cv
import argparse
import imutils
from wheelDetection.fpsTracker import FPS
from imutils.video import WebcamVideoStream
import time
from drawBB import drawBoundingBox
import numpy as np

# from root directory cascadeClassifier
# python videoControler.py -v TestVideo/CarPark.mp4 -m models/cascadeWheelClassifier.xml -w 600
# drop the -v argument to run off webcam. add -r = 1 to record session.
# press q to end displaying of results

## Should rewite code with methods and sitch staments, note that the cascade classifier struggle in very low light and poor resolution

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-m", "--model", help="path to the cascade classifier model", required=True)
ap.add_argument("-r", "--recording", help="input 0 or higher number if recording", default=-1)
ap.add_argument("-w", "--width", help="integer value of width of processing widow.\
                                    Smaller will speed up algorithm but lower precision", default=300)
args = vars(ap.parse_args())


class findWheels:
    ### CREATE OUR OBJECTS FROM OTHER CLASSES
    # camera object
    camera = None

    ## change source to args["video"] when using video feed
    # load the video
    if args["video"] is None:
        camera = WebcamVideoStream(src=3).start()
        # if you can't find a USB webcam on linux type 'ls /dev/video*' with and without the camera plugged in to find the port number..
    else:
        camera = WebcamVideoStream(src=args["video"]).start()
    ## AttributeError: 'NoneType' object has no attribute 'shape'  , this occures when the path to the video is incorrect

    # load the trained model
    cascade_wheel = cv.CascadeClassifier(args["model"])

    # create a bounding box draw object. This also checks the boxes are actual wheels
    drawbb = drawBoundingBox()

    # create a time object to show fps and timer
    fps = FPS()

    # video recoder object, this may or may not be used
    out = None


    ### DEFINE VARIABLES

    # variable declaration on weather or not the video feed should be downsized for display
    mustResize = False
    # the width of the display window
    displayWidth = 1000

    # ratio of "original image width" to "display image width". Important when rescaling
    # "display image" back to "original image". These are different sizes to speed up algorithm.
    ratio2originalimg = 0.0

    # ratio of "display image width" to "worked on image width". Important when rescaling
    # "work on image" back to "display image". These are different sizes to speed up algorithm.
    ratio2workedimg = 0.0

    # minimum number of pixels for the detection to occure
    minDetctionSize = 100

    # used in BB prioritization
    original_videofeed_height = 0
    midpointOfFeed = 0

    ### DEFINE FUNCTIONS

    # rescale bounding boxes from worked image to either display image or original image
    def boundingBoxRescale(self, rectangles, ratio):
        rescaledBoundingBoxes = []
        for r in rectangles:
            boundingBox = []
            for coordinate in r:
                newCoordinate = int(round(coordinate * ratio))
                boundingBox.append(newCoordinate)
            rescaledBoundingBoxes.append(boundingBox)
        return rescaledBoundingBoxes

    def setVideoInputHeight(self, original_videofeed_height):
        self.original_videofeed_height = original_videofeed_height
        self.midpointOfFeed = int(round(original_videofeed_height / 2))

    def initialseVideoIntake(self):
        print("[INFO] camera is warming up")
        time.sleep(1.0)
        print("[INFO] camera has finished warming up")

        # grab a single frame to find size frame
        frame = self.camera.read()

        # check if the video feed is working
        if frame is None:
            print("################################")
            print("[WARNING] error has occurred")
            print("There is no video feed. \n In a webcam is being used please check that"
                  " the right WebcamVideoStream(src=2) is being used. Refer to line 39 of the code to change it. \n"
                  "In the case of a video being read from file, make sure the file path and name was correctly written")
            print("################################")
            return

        originalFrame = frame.copy()
        # check if video will fit on screen, if its width is bigger than displayWidth pixels, it's reduced to 1000 pixels
        if frame.shape[1] > self.displayWidth:
            frame = imutils.resize(frame, width=self.displayWidth)
            self.mustResize = True

        # keep reduction ratio to assist with upscaling later
        self.ratio2workedimg = frame.shape[1] / int(args["width"])

        # keep reduction ratio to assist with feeding the CNN and ORB feature detectors with highest possible resolution ROI
        self.ratio2originalimg = originalFrame.shape[1] / frame.shape[1]  # When checking a bounding box, use ratio to retrieve it's pixel contents from original img

        # important setup for drawbb obj, this is part of the BB prioritisation code. shape[0] is the frame height
        self.setVideoInputHeight(originalFrame.shape[0])

        if int(args["recording"]) >= 0:
            print("[INFO] video is now recording")
            # this is to record video
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            self.out = cv.VideoWriter('videoOutput.avi', fourcc, 15.0, (frame.shape[1], frame.shape[0]))

    # main body of code that loops through frames and launches other functions
    def runDetector(self):
        # Start off fps counter to measure speed of video.
        self.fps.start()
        while True:
            start = time.perf_counter()
            # grab the current frame
            originalFrame = self.camera.read()
            # check the frame isn't empty. If so code should stop running
            if originalFrame is None:
                print("[INFO] The code has finished running")
                return
            if self.mustResize == True:
                displayFrame = imutils.resize(originalFrame, width=1000)

            else:
                displayFrame = originalFrame    # the display and original frame are the same size.
                # resize to speed up process
            workFrame = imutils.resize(displayFrame, int(args["width"]))

            # detection of car wheel locations, high recall low precision/high true positives and high false positives
            rectangles = self.cascade_wheel.detectMultiScale(workFrame)

            # perform bounding box regression to prevent bounding boxes from overlapping
            non_max_rectangles = self.non_max_suppression_fast(rectangles, overlapThresh=0.2)

            # reorder BB to give priority to BB that are likely to be car wheels.
            # Important as their may not be time to check all of them, also it's necessary to only to run CNN on best BBs.
            rectangles_reordered = self.BBreorder(non_max_rectangles)

            # The rectangles have coordinates form the downscaled image.
            # The new bounding boxes will rescale according to the size of the original image.
            # Look at ratio variable to understand. This is necessary for ORB and CNN to work well.
            boundingBoxesOrigin = getThemWheels.boundingBoxRescale(rectangles_reordered,
                                                                   self.ratio2originalimg*self.ratio2workedimg)
            boundingBoxesDisplay = getThemWheels.boundingBoxRescale(rectangles_reordered,
                                                                    self.ratio2workedimg)
            # check bounding boxes is a wheel and add bounding boxes to display video
            displayFrame = self.drawbb.draw(originalFrame=originalFrame, rectanglesOriginal=boundingBoxesOrigin,
                                            displayFrame=displayFrame, rectanglesDisplay=boundingBoxesDisplay)


            # add fps to the image
            if self.fps._numFrames > 10:
                ts = "[INFO] elasped time: {:.2f}".format(self.fps.elapsed())
                ts += " and Approx. FPS: {:.2f}".format(self.fps.fps())
                cv.putText(displayFrame, ts, (10, displayFrame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.50,
                           (0, 0, 255), 2)

            # save displayFrame to recording
            if int(args["recording"]) > 0:
                self.out.write(displayFrame)

            # show the displayFrame and check if a key is pressed
            cv.imshow("Car wheel Detection", displayFrame)
            key = cv.waitKey(1) & 0xFF

            # if the 'q' key is pressed, stop the loop
            if key == ord("q") or key == ord("Q"):
                break

            # update the FPS counter
            self.fps.update()

            finish = time.perf_counter()
            # code to stabilise video output
            while finish - start < 0.08:
                time.sleep(0.01)
                finish = time.perf_counter()
            # print(f' Frame took {round(finish - start, 2)}second(s)')
        # stop the timer and display FPS information

        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        # cleanup the camera and close any open windows
        self.camera.stop()

        # release video recorder
        if int(args["recording"]) > 0:
            self.out.release()

        cv.destroyAllWindows()

    # code to reorder bounding boxes, prioritises boxes that are bigger and closer to the center of the image.
    def BBreorder(self,rectanglesOriginal):
        # set up variables
        alpha = 1.5                 # size coefficient
        beta = 0.5                  # distance from mid pont coefficient
        orderedBB_withscore = []    # new bounding box array

        # take second element for sort, used in: orderedBB_withscore.sort(key=takeSecond, reverse=True)
        def takeSecond(elem):
            return elem[1]

        # iterate through original bounding boxes
        for rectangleOri in rectanglesOriginal:
            # x,y being the BB top left corner position and w,h being width and height respectively
            (x, y, w, h) = rectangleOri

            # make sure the number is positive and rounded to the closest int.
            height_BB_center = abs(int(round(y + (h / 2))))

            # dividing with self.original_videofeed_height is to normalise the numbers between 1-0
            size = h/self.original_videofeed_height
            distance = (self.midpointOfFeed - height_BB_center)/self.original_videofeed_height
            score = (alpha*size)-(beta*distance)
            orderedBB_withscore.append([rectangleOri, score])


        if len(orderedBB_withscore) != 0:
            orderedBB_withscore.sort(key=takeSecond, reverse=True)
            orderedBB = []
            for item in orderedBB_withscore:
                (singleBB, score) = item
                orderedBB.append(singleBB)
            return orderedBB
        else:
            return rectanglesOriginal

    # Malisiewicz et al.
    def non_max_suppression_fast(self, bb_boxes, overlapThresh=0.1):
        # if there are no boxes, return an empty list
        if len(bb_boxes) == 0:
            return []

        boxes = []
        # transform the bounding box data type to work with the algorithm
        for box in bb_boxes:
            (x1, y1, w1, h1) = box
            y2 = y1 + h1
            x2 = x1 + w1
            a_box = (float(x1), float(y1), float(x2), float(y2))
            boxes.append(a_box)

        boxes = np.array(boxes)

        # initialize the list of picked indexes
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        new_bb_boxes = []
        # transform the bounding box data type to work with the algorithm
        for box in boxes[pick].astype("int"):
            (x1, y1, x2, y2) = box
            w1 = y2 - y1
            h1 = x2 - x1
            a_box = (x1, y1, w1, h1)
            new_bb_boxes.append(a_box)
        new_bb_boxes = np.array(new_bb_boxes)
        return new_bb_boxes


getThemWheels = findWheels()
getThemWheels.initialseVideoIntake()
getThemWheels.runDetector()




