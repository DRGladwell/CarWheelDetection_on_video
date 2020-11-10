from wheelDetection.FeatureDetectAndMatch import OrbDetection
import numpy as np
import math


class quick_check_bb_thread:
    def __init__(self, very_close_distance = 40):

        # the pixel distance between two bounding boxes for them to be considered the same,
        # very important to tune correctly as no further checks occur!!
        self.very_close_distance = very_close_distance

        # holds the old bounding box
        self.t1_bounding_box_obj = None

        # create a feature matching object
        self.featureMatcher = OrbDetection()

    # run a check on time step 2 (frame and bounding box) for the time step 1 bounding box object.
    # if correctly spotted, time step 1 object is updated with time step 2 information.
    # Note: time step 1 is the item that the car wheel was last observed occurred.
    # Time step 2 is the latest item from the video buffer
    # Note: the return statement of this method is never used and thus meaningless
    def run(self, t2_frame, t2_bounding_boxes):
        # convert bounding boxes into numpy array type
        t2_bounding_boxes = np.array(t2_bounding_boxes)
        # if no bounding boxes are in current frame then their is nothing to check against
        if len(t2_bounding_boxes) == 0:
            return None
        # handles the shape issue encountered when only a single BB is given to the code
        if t2_bounding_boxes.ndim == 1:
            t2_bounding_boxes = np.array([t2_bounding_boxes])

        # check if nearby & similar past bounding box was a car wheel
        closest_t2_bb_info = self.boudingBoxSimulartity(t2_bounding_boxes)  # closestBox is [[key,distance]]

        if closest_t2_bb_info is not None:
            # unpack closest_t2_bb_info as it has (bounding box name, distance)
            (closest_t2_bb, distance_between_t1_and_best_t2) = closest_t2_bb_info
            # prepare current BB for feature matching
            wheelROI = self.featureMatcher.prepareImages(image=t2_frame, boundingBox=closest_t2_bb)
            # train features for current image
            t2_features = self.featureMatcher.trainingImage(wheelROI)

            # retrieve the closest bounding box at time step 2
            (x2, y2, w2, h2) = closest_t2_bb
            # get the center x, y position of the current BB
            location_t2 = (int(round(x2 + w2 / 2)), int(round(y2 + h2 / 2)))

            if distance_between_t1_and_best_t2 < self.very_close_distance:
                #print("we found a BB that is close and that matches")
                #print("the distance away is: " + str(distance_between_t1_and_best_t2))
                # in this special case, don't update the features as no orb test was performed
                return self.updateBB(new_ROI=self.t1_bounding_box_obj.last_ROI,
                                     currentFeatures=self.t1_bounding_box_obj.lastFeatures,
                                     boundingBox=closest_t2_bb, location=location_t2)

            # check if there is a feature match between the past closest BB and the current BB
            (success, score) = self.featureMatcher.computeQuery(t2_features,
                                            self.t1_bounding_box_obj.lastFeatures)  # success is a tuple of (bool,int)
            if success:
                # update past BB
                return self.updateBB(new_ROI=wheelROI, currentFeatures=t2_features,boundingBox=closest_t2_bb, location=location_t2)

        # check through all bounding boxes for ORB match
        else:
            highestScore = 0
            bestMatch = None
            for boundingBox in t2_bounding_boxes:
                # prepare current BB for feature matching
                wheelROI = self.featureMatcher.prepareImages(image=t2_frame, boundingBox=boundingBox)
                # train features for current image
                t2_features = self.featureMatcher.trainingImage(wheelROI)
                # check if there is a feature match between a past BB and the current BB
                (success, score) = self.featureMatcher.computeQuery(t2_features, self.t1_bounding_box_obj.lastFeatures)

                if success and highestScore < score:
                    # update the high score and best match
                    highestScore = score
                    bestMatch = (boundingBox, t2_features)

            if bestMatch is not None:
                # update past BB
                (boundingBox, t2_features) = bestMatch
                (x2, y2, w2, h2) = boundingBox
                # get the center x, y position of the current BB
                location_t2 = (int(round(x2 + w2 / 2)), int(round(y2 + h2 / 2)))
                return self.updateBB(new_ROI=wheelROI,currentFeatures=t2_features, boundingBox=boundingBox, location=location_t2)

    # check the manhattan distance of current frame bounding box to previously found bounding boxes
    def boudingBoxSimulartity(self, t2_bounding_boxes, maxRatioDiff=1.2, minRatioDiff=0.8, maxDistance = 1000):
        maxRatioDiff = maxRatioDiff  # maximum area ratio diff allowed
        minRatioDiff = minRatioDiff  # minimum area ratio diff allowed
        maxPixelDistance = maxDistance  # maximum pixel distance allowed for exhaustive search
        potentialCarWheel = []  # a list of close similar bounding boxes from prior frames

        # take second element for sort
        def takeSecond(elem):
            return elem[1]

        for boundingBox in t2_bounding_boxes:
            # box shape and location for t2 box
            (x2, y2, w2, h2) = boundingBox
            # get the center x, y position of the t2 BB
            location_t2 = (int(round(x2 + w2 / 2)), int(round(y2 + h2 / 2)))

            (x2center, y2center) = location_t2

            if self.t1_bounding_box_obj is not None:
                t1_bb_obj = self.t1_bounding_box_obj
                (x1, y1, w1, h1) = t1_bb_obj.box
                (x1center, y1center) = t1_bb_obj.location

                # create a ratio between the area of new and old bounding boxes
                ratio = (w2 * h2) / (w1 * h1)
                distance = math.hypot((x2center - x1center), (y2center - y1center))

                # test that current bounding box is close to size and location of a prior carWheel bounding box
                if minRatioDiff < ratio < maxRatioDiff:
                    if distance < maxPixelDistance:
                        # if it is add it to the list of potential candidates
                        potentialCarWheel.append([boundingBox, distance])
        # return that best car wheel candidate from list of bounding boxes based purely on size and location
        if len(potentialCarWheel) != 0:
            potentialCarWheel.sort(key=takeSecond)
            best_bb_and_distance = potentialCarWheel[0]
            return best_bb_and_distance
        else:
            return None  # a failure state so the code can continue

    # update the position of t1 to it's new position in t2
    def updateBB(self, new_ROI, currentFeatures, boundingBox, location):
        self.t1_bounding_box_obj.setFeatures(currentFeatures)  # set the Orb features for the new ROI
        self.t1_bounding_box_obj.set_BB(boundingBox)  # set new bounding box
        self.t1_bounding_box_obj.setLocation(location)  # set new location
        self.t1_bounding_box_obj.last_ROI = new_ROI
        return boundingBox  # return the bounding box

    # a setter function to add an entry into the car wheel dictionary
    def update_carWheel(self, bbObj):
        self.t1_bounding_box_obj = bbObj