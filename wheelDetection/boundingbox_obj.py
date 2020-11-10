import time

class boundingBoxObject:
    def __init__(self, id, box, location):
        self.id = id
        self.box = box               # x, y, w, h
        self.location = location     # x, y  <- of the center of the box and not the top left corner
        self.lastTimeSeen = float
        self.lastProbability = 0
        self.lastFeatures = None
        self.last_time_seen = time.perf_counter()
        self.Show = False
        self.wasUpdated = False
        self.last_ROI = None

    # setter of bounding box and add score
    def seen_BB(self):
        self.wasUpdated = True
        current = time.perf_counter()
        self.last_time_seen = current


    def set_BB(self, bounding_box):
        self.box = bounding_box

    def setProbability(self, probability):
        self.lastProbability = probability

    # setter of bounding box center location
    def setLocation(self, location):
        self.location = location

    # setter of ORB features. Used in ORB feature detection
    def setFeatures(self, trainedFeatures):
        self.lastFeatures = trainedFeatures

    # all BB that aren't seen in current frame. When score is low enough the BB is killed
    def killInstance(self):
        current = time.perf_counter()
        duration = current - self.last_time_seen
        # if the bounding box object hasn't been seen for 8 seconds forget it itom memory
        if duration > 20:
            return True
        else:
            return False

        # The point of killing of a BB instance is to reduce memory usage
    # and to sensibly limit the number of wheels searched for