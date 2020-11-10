from wheelDetection.wheelDetectionTransferLearning import wheelDetectorTransferLearning
from boundingbox_obj import boundingBoxObject
from wheelDetection.FeatureDetectAndMatch import OrbDetection
from quick_check_bb_thread import quick_check_bb_thread  # be careful this is quick_check_bb_thread!! not quick_check_BB
import time


class cnnThread:
    def __init__(self, videoBufferObj):
        # create wheel detector that used a convolutional neural network with a logistic regression model fit on top.
        # Note: the logistic regression model was trained using machine learning. Hence why the thread is called CNN_ML
        self.wheelDetection = wheelDetectorTransferLearning()
        # create a feature matching object
        self.featureMatcher = OrbDetection()
        # create a quick_bounding_box checker object, this one is specially made for the CNN_ML thread
        self.quick_check_for_catchup = quick_check_bb_thread()

        # get the video buffer object from drawObj, note: CNN is on a secondary thread but can access the videoBufferObj
        # even though it's being updated on the Main thread.
        self.videoBuffer = videoBufferObj

    # run the bounding box patch of image on the CNN_ML models to see if it's a car wheel
    def single_BB_checker(self, originalFrame, boundingBox, secondary_thread_frame_ID):
        start = time.perf_counter()
        # x1,y1 are the top left corner of the box. w1,h1 are the width and height of the box
        (x1, y1, w1, h1) = boundingBox
        # get the center x, y position of the current BB
        location = (int(round(x1 + w1 / 2)), int(round(y1 + h1 / 2)))

        # call extract method which runs the cnn on the current bounding box
        proba = self.test_roi_carwheel(originalFrame, boundingBox)  # Note: proba is an unpackable variable
        probability = "{:.2f}%".format(proba[True] * 100)

        # if statement for when the bounding box is a car wheel
        if proba[True] >= 0.5:
            carWheelName = "wheel_CNN_and_ML"
            # creation of a new bounding box object
            bbObj = boundingBoxObject(id=carWheelName, box=boundingBox, location=location)
            bbObj.lastProbability = probability

            # prepare current BB for feature matching
            wheelROI = self.featureMatcher.prepareImages(image=originalFrame, boundingBox=boundingBox)
            # train features for current image
            currentFeatures = self.featureMatcher.trainingImage(wheelROI)

            # create ROI and train the feature matcher on it
            bbObj.setFeatures(currentFeatures)  # set the Orb features for the new ROI
            bbObj.last_ROI = wheelROI

            # make this bounding box object is the one used in the quick_check_for_catchup instance
            self.quick_check_for_catchup.update_carWheel(bbObj)

            # useful for measuring how long the CNN takes
            finish = time.perf_counter()
            #print(f' cnn_ml took {round(finish - start, 2)}second(s)')
            # return the name of the car wheel and probability
            return self.catchUp_to_current_frame(secondary_thread_frame_ID)
            ## Incredibly placing this statment in the return matters. This reduces a 15 frame difference to a 3 frame
            ## difference between catchup to current frame occuring and then the bounding box object is added
            ## to the quick_BB dictionary.
        else:
            return None
                #(None, probability)  # a failure state so the code can continue

    # used to test if a car wheel is present in a ROI (region of interest).
    def test_roi_carwheel(self, originalFrame, bounding_box):
        (x, y, w, h) = bounding_box
        # crop the patch of image of the bounding box
        cropImg = originalFrame[y:y + h, x:x + w]
        return self.wheelDetection.testImage(cropImg)  # returns the probability associated with the ROI
        # this returns a meaningful probability


    # use the frame buffer to catch the CNN_ML thread frame to the current frame using Quick assumptions
    # on the bounding boxes and images. test_roi_carwheel() takes 0.7s and a lot could of happened in the img.
    def catchUp_to_current_frame(self, secondary_thread_frame_ID):
        # set the frame ID of the main thread
        current_main_thread_frame_ID = self.videoBuffer.frameID

        # set the frame ID of the secondary thread, it wll get updated in this methode
        secondary_thread_frame_ID = secondary_thread_frame_ID

        # get the size of the buffer reader object
        buffer_size = self.videoBuffer.get_buffer_length()

        #  while the method hasn't caught up to current frame
        while secondary_thread_frame_ID != current_main_thread_frame_ID:

            # update the frame ID of the main thread
            current_main_thread_frame_ID = self.videoBuffer.frameID

            # get the buffer item
            buffer_item = self.videoBuffer.retrieve(current_main_thread_frame_ID, secondary_thread_frame_ID)
            (frame, BB, frameID) = buffer_item  # unpack buffer_item
            #print("the Bounding boxes are: " + str(BB))

            # check if the frame t1 bounding box can be found in the frame t2, with t1 being the frame prior to t2
            self.quick_check_for_catchup.run(frame, BB)

            # increment the secondary_thread_frame_ID by one
            secondary_thread_frame_ID += 1
            # make sure not to go over buffer length, otherwise loop round back to 0
            if secondary_thread_frame_ID >= buffer_size:
                secondary_thread_frame_ID = 0

            # useful to debug code or to see how fast the thread catches up with current frame
            # print("The secondary_thread_frame_ID: " + str(secondary_thread_frame_ID)+
            #         " The current_main_thread_frame_ID: " + str(current_main_thread_frame_ID))

        # return the bounding box object relevant to the current frame
        return self.quick_check_for_catchup.t1_bounding_box_obj

        ## Some checks will have to be made to make sure we didn't lose the frame.
        ## Like doing a final ORB test



