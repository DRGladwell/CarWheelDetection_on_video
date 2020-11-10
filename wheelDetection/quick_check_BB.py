from wheelDetection.FeatureDetectAndMatch import OrbDetection
import threading
import math

class quick_check_BB:
    def __init__(self, very_close_distance = 40, max_num_car_wheels = 7):

        # the pixel distance between two bounding boxes for them to be considered the same,
        # very important to tune correctly as no further checks occur!!
        self.very_close_distance = very_close_distance

        # holds car wheel objects and none car wheel objects in dictionaries
        self.carWheel = {}
        self.notCarWheel = {}

        # this is a potential car wheel to add, it is checked in the run methode to see if it's a new BB
        self.potential_car_wheel = None

        # create a feature matching object
        self.featureMatcher = OrbDetection()

        # lock the dictionaries to avoid race conditions cause by threading
        self.lock = threading.Lock()

        self.max_num_car_wheels = max_num_car_wheels

        # used to name car wheels
        self.wheel_num = 0


    def run(self, originalFrame, boundingBox):
        (x1, y1, w1, h1) = boundingBox
        # get the center x, y position of the current BB
        location = (int(round(x1 + w1 / 2)), int(round(y1 + h1 / 2)))

        # prepare current BB for feature matching
        wheelROI = self.featureMatcher.prepareImages(image=originalFrame, boundingBox=boundingBox)
        # train features for current image
        currentFeatures = self.featureMatcher.trainingImage(wheelROI)

        # check if nearby & similar past bounding box was a car wheel
        closestBox = self.boudingBoxSimulartity(boundingBox, location)  # closestBox is [[key,distance]]

        if closestBox is not None:
            (pastBB_BoxName, passBB_distance) = closestBox
            pastBB = self.carWheel[pastBB_BoxName]

            if passBB_distance < self.very_close_distance:
                # print("we found a BB that is close and that matches")
                # print("the distance away is: "+str(passBB_distance))
                # pastBB.lastFeatures because image might be blurry

                # this is a very strong guess, so the the ORB features are not updated here
                pastBB.setProbability("Quick search")
                return self.updateBB(pastBB=pastBB, currentFeatures=pastBB.lastFeatures,
                                     boundingBox=boundingBox, location=location)
            # check if there is a feature match between the past closest BB and the current BB

            # (success, score) = self.featureMatcher.computeQuery(currentFeatures,
            #                                                     pastBB.lastFeatures,wheelROI,pastBB.last_ROI)  # success is a tuple of (bool,int)
            (success, score) = self.featureMatcher.computeQuery(currentFeatures, pastBB.lastFeatures,matches_threshold=40)
            #print("we continue the search, the score value is: " +str(score))
            if success:
                # update past BB
                pastBB.setProbability("Quick Orb")
                pastBB.last_ROI = wheelROI
                return self.updateBB(pastBB=pastBB, currentFeatures=currentFeatures,
                                     boundingBox=boundingBox, location=location)

        ## SHOULD PROBABLY ADD AN EXTRA LEVEL HERE TO PRIORITISE A CHECK OF CLOSER BB?
        ## boudingBoxSimulartity COULD DO WITH RETURNING A LIST OF BEST CANDIDATES, THIS COULD
        ## HELP ORB FEATURES DETECTOR NOT MAKE MISTAKES AND RUN FASTER, is that not what's just happening above?
        # I think it is...

        # check through know car wheels if there is a match
        else:
            highestScore = 0
            bestMatch = None
            self.lock.acquire()
            for key, value in self.carWheel.items():
                pastBBs = value
                # check if there is a feature match between a past BB and the current BB
                #(success, score) = self.featureMatcher.computeQuery(currentFeatures, pastBBs.lastFeatures,wheelROI,pastBBs.last_ROI)
                (success, score) = self.featureMatcher.computeQuery(currentFeatures, pastBBs.lastFeatures,matches_threshold=40)
                if success and highestScore < score:
                    highestScore = score
                    bestMatch = pastBBs
            self.lock.release()

            if bestMatch is not None:
                # update past BB
                bestMatch.setProbability("Extensive Orb")
                bestMatch.last_ROI = wheelROI
                return self.updateBB(pastBB=bestMatch, currentFeatures=currentFeatures,
                                     boundingBox=boundingBox, location=location)

            else:
                return None


        # check the manhattan distance of current frame bounding box to previously found bounding boxes

    # check the manhattan distance of current frame bounding box to previously found bounding boxes
    def boudingBoxSimulartity(self, boundingBox, location, maxRatioDiff=1.2, minRatioDiff=0.8, maxDistance = 300):
        maxRatioDiff = maxRatioDiff  # maximum area ratio diff allowed
        minRatioDiff = minRatioDiff  # minimum area ratio diff allowed
        maxPixelDistance = maxDistance  # maximum pixel distance allowed for exhaustive search
        potentialCarWheel = []  # a list of close similar bounding boxes from prior frames

        # take second element for sort
        def takeSecond(elem):
            return elem[1]

        # box shape and location for current box
        (x1, y1, w1, h1) = boundingBox
        (x1center, y1center) = location
        self.lock.acquire()
        # check the dictionary isn't empty
        if len(self.carWheel) != 0:
            for key, value in self.carWheel.items():  # run through the keys in the dictionary
                priorBoundingBox = value  # the value of the dictionary is a bounding box object
                # print("##############")
                # print(priorBoundingBox.id, priorBoundingBox.box)
                # print("new bounding box", boundingBox)

                (x2, y2, w2, h2) = priorBoundingBox.box
                (x2center, y2center) = priorBoundingBox.location

                # create a ratio between the area of new and old bounding boxes
                ratio = (w1 * h1) / (w2 * h2)
                distance = math.hypot((x1center - x2center), (y1center - y2center))
                # print(distance)
                # print("##############")
                # test that current bounding box is close to size and location of a prior carWheel bounding box
                if minRatioDiff < ratio < maxRatioDiff:
                    if distance < maxPixelDistance:
                        potentialCarWheel.append([key, distance])
        self.lock.release()
        if len(potentialCarWheel) != 0:
            potentialCarWheel.sort(key=takeSecond)
            # print("distance car wheel is: ")
            # print(potentialCarWheel[0][1])
            return potentialCarWheel[0][0], potentialCarWheel[0][1]
            # returns the carWheel dictionary key closest to current bounding box & distance
        else:
            return None  # a failure state so the code can continue

    def updateBB(self, pastBB, currentFeatures, boundingBox, location):
        self.lock.acquire()
        pastBB.seen_BB()  # this increases the bounding boxes score
        pastBB.setFeatures(currentFeatures)  # set the Orb features for the new ROI
        pastBB.set_BB(boundingBox)  # set new bounding box
        pastBB.setLocation(location)  # set new location
        self.lock.release()
        return (pastBB.id, pastBB.lastProbability)  # return the name of the car wheel

    # used to check is the new BB corresponds to a bounding box. Ideally it would.
    def check_current_frame_new_bb(self, originalFrame, bounding_boxes, new_bb):
        # iterate through the leftover bounding boxes
        for boundingBox in bounding_boxes:

            # prepare current BB for feature matching
            wheelROI = self.featureMatcher.prepareImages(image=originalFrame, boundingBox=boundingBox)
            # train features for current image
            currentFeatures = self.featureMatcher.trainingImage(wheelROI)
            # run an orb feature test
            (success, score) = self.featureMatcher.computeQuery(currentFeatures, new_bb.lastFeatures)
            if success:
                return True, score
            else:
                return None

    # used to check if a new BB corresponds to a existing BB. Ideally is would not.
    def check_dictionary(self, originalFrame, new_bb_obj):
        # prepare current BB for feature matching
        # wheelROI = self.featureMatcher.prepareImages(image=originalFrame, boundingBox=new_bb_obj.box)
        # train features for current image
        new_bb_features = self.featureMatcher.trainingImage(new_bb_obj.last_ROI)
        #print("the length of the dictionary is " + str(len(self.carWheel.items())))
        for key, value in self.carWheel.items():
            prior_bb = value
            (success, score) = self.featureMatcher.computeQuery(queryfeatures=new_bb_features,
                                                                trainfeatures=prior_bb.lastFeatures,
                                                                matches_threshold=9,querry_ROI=new_bb_obj.last_ROI,
                                                                train_ROI=prior_bb.last_ROI)

            #print(success, score)
            if success:
                return True, score
            # else:
            #     cv2.imshow("in dictionary " + prior_bb.id, prior_bb.last_ROI)
            #     cv2.imshow("in dictionary new BB", new_bb_obj.last_ROI)
            #     cv2.waitKey(0)
            #     cv2.destroyWindow("in dictionary " + prior_bb.id)
            #     cv2.destroyWindow("in dictionary new BB")
        return None

    # add a potential car wheel to the temp variable, it will be checked with check_if_carwheel_already_exist in draw.py
    def add_potential_carwheel(self, buffer_item, bb_obj):
        # used frame buffer to retrieve last frame
        (frame, bb, frameID) = buffer_item
        # this is the potenrial bounding box. MIGHT NEED TO MAKE IT INTO A LIST
        self.potential_car_wheel = (frame, bb_obj)

    # a setter function to add an entry into the car wheel dictionary. Most of the method is to avoid duplicates.
    def check_if_carwheel_already_exist(self,originalFrame, undetermined_bounding_boxes):
        # a conditional statement to make sure there is a potential car wheel
        if self.potential_car_wheel is not None:
            (frame, new_bb_obj) = self.potential_car_wheel

            # run a check_dictionary. If it's a previously known obj, result == True
            #print("check_dictionary")
            result_known_bb = self.check_dictionary(originalFrame=frame, new_bb_obj=new_bb_obj)
            # verify that the CNN_ML thread has produced a new car wheel object
            if result_known_bb is None and len(self.carWheel) < self.max_num_car_wheels:
                # check that new Car wheel is visible in current frame
                #print("check_current_frame_new_bb")
                result_undetermined_bb = self.check_current_frame_new_bb(originalFrame,
                                                                         undetermined_bounding_boxes, new_bb_obj)
                if result_undetermined_bb is not None:
                    print("result_undetermined_bb is not none is True")
                    print(result_undetermined_bb)
                    self.wheel_num += 1
                    car_wheel_name = ("Wheel " + str(self.wheel_num))
                    new_bb_obj.id = car_wheel_name
                    print("the new car wheel " + new_bb_obj.id + " has been found.")

                    self.lock.acquire()
                    self.carWheel[car_wheel_name] = new_bb_obj
                    self.lock.release()
                    ########
                    # # used for debugging
                    # #wheelROI = self.featureMatcher.prepareImages(image=originalFrame, boundingBox=new_bb_obj.box)
                    # # train features for current image
                    # new_bb_features = self.featureMatcher.trainingImage(new_bb_obj.last_ROI)
                    #
                    # print("Secondary check")
                    # for key, value in self.carWheel.items():
                    #     prior_bb = value
                    #     # cv2.imshow(prior_bb.id, prior_bb.last_ROI)
                    #     # cv2.waitKey(0)
                    #     # cv2.destroyWindow(prior_bb.id)
                    #
                    #     (success, score) = self.featureMatcher.computeQuery(queryfeatures=new_bb_features,
                    #                                                     trainfeatures=prior_bb.lastFeatures,
                    #                                                     matches_threshold=6, querry_ROI=new_bb_obj.last_ROI,
                    #                                                     train_ROI=prior_bb.last_ROI)
                    #     print(success, score)

        self.potential_car_wheel = None  # maybe this needs to be thread locked

    # Give a bounding box a score to figure out if it should be deleted. The longer the BB presence on screen the longer
    # it will be remembered.
    def scoreBB(self):
        deleteTheseKeys = []
        for key, pastBB in self.carWheel.items():
            if pastBB.wasUpdated == False:
                # methode will reduce the score, if the score is low enough, death = True
                death = pastBB.killInstance()   # when score is low enough the pastBB will be removed from dictionary
                if death == True:
                    deleteTheseKeys.append(key)
            # reset the update boolean
            pastBB.wasUpdated = False
        for key in deleteTheseKeys:
            # print(self.carWheel[key].id + " was deleted!")
            del self.carWheel[key]