# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
import imutils
from imutils import paths
# code to take training images with bounding boxes and export a file of wheel images (content of the bounding boxes ..)
import argparse
import os

# example: python Cascade_classifier_test.py -f data/Cascade_classifier_dataset/posObjectDetection.txt -i data/Cascade_classifier_dataset/testset_negatives/


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the pos.txt file", required=True) # carful pos.txt has the right image paths
ap.add_argument("-i", "--images", required=True, help="Path to the negative images")

# take in the test images
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["images"]))

# Returns float value describing how well the predicted and ground truth bounding boxes over lapped.
def bb_intersection_over_union(bb_a, bb_b):
    x_a,y_a,w_a,h_a = bb_a
    x_b,y_b,w_b,h_b = bb_b
    
    boxA = [x_a,y_a,x_a+w_a,y_a+h_a]
    boxB = [x_b,y_b,x_b+w_b,y_b+h_b]
    # print("these are the bounding boxes")
    # print(boxA)
    # print(boxB)

    # determine the (x, y)-coordinates of the intersection rectangle
    ### NEED TO CREATE A WAY TO GET TOP LEFT AND BOTTOM RIGHT OF BB
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou



# returns content of an item as a list
def splitItem(item):
    content = item.split()
    return content



# main code block
class fileRead:
    def __init__(self):
        self.cascade_wheel = cv2.CascadeClassifier("models/cascadeWheelClassifier.xml")

        self.fileContents = []  # paths split line by line
        self.fileItemsObj = []  # array of fileItem objects
        self.wheels = []        # will contain images of wheels extracted from original set of images

        self.cascade_width = 600     # pixel with of cascade classifier input images

        self.average_iou = 0

        GT = 0  # ground truth
        P = 0  # proposed
        TP = 0  # true positive
        FP = 0  # false positive
        FN = 0  # false negative

        # Beginning of the script
        self.readInputFile()    # generates the paths split line by line

        # generate fileItem objects
        for line in self.fileContents:
            self.fileItemsObj.append(fileItem(line))

        # extract img of car wheel img from bounding box information
        for fileObj in self.fileItemsObj:
            results = self.compare_cascade_to_ground_truth(fileObj)
            total_GT, total_P, total_TP, total_FP, total_FN = results

            GT += total_GT
            P  += total_P
            TP += total_TP
            FP += total_FP
            FN += total_FN

        print("The total number of ground truths bounding boxes: " + str(GT))
        print("The total number of proposed: " + str(P))
        print("The total number of true positives: " + str(TP))
        print("The total number of false positives: " + str(FP))
        print("The total number of false negatives: " + str(FN))

        print("the average iou = " + str(self.average_iou/TP))

        # Test the cascade doesn't spot car wheels in images with no car wheels
        self.test_negative_images()

    # reads the bounding box CSV file line by line
    def readInputFile(self):
        f = open(args["file"])
        self.fileContents = f.readlines()

    def test_negative_images(self):
        total_FP = 0  # false positive
        total_TN = 0  # true negatives
        total_number_of_negatives = 0
        for image in imagePaths:
            total_number_of_negatives += 1
            img = cv2.imread(image)
            ratio2workedimg = img.shape[1] / self.cascade_width
            resize_img = imutils.resize(img, self.cascade_width)
            proposed_bb = self.cascade_classifer(resize_img)
            number_of_bb = len(proposed_bb)
            if number_of_bb > 0:
                total_FP += number_of_bb
            else:
                total_TN += 1
            self.draw_rectangles(resize_img, proposed_bb)

        print("the number of negative images: " + str(total_number_of_negatives))
        print("the number of true negatives: " + str(total_TN))
        print("the number of false positives: " + str(total_FP))

    # # extracts the the image inside the bounding box
    def compare_cascade_to_ground_truth(self, fileObj):
        img = cv2.imread(fileObj.name)
        # cv2.imshow("cascade classifier image", img)
        # cv2.waitKey(0)

        ratio2workedimg = img.shape[1] / self.cascade_width
        #print(ratio2workedimg)
        resize_img = imutils.resize(img, self.cascade_width)

        proposed_bb = self.cascade_classifer(resize_img)
        rescaled_proposed_bb = self.boundingBoxRescale(proposed_bb, ratio2workedimg)

        #self.draw_rectangles(img, rescaled_proposed_bb)
        #self.draw_rectangles(img, fileObj.rectangles)
        #print(rescaled_proposed_bb)

        total_GT = 0    # ground truth
        total_P  = 0    # proposed
        total_TP = 0    # true positive
        total_FP = 0    # false positive
        total_FN = 0    # false negative


        # Use this as there's only one bounding box per image
        # USED ON POSITIVE DATA SET
        for gt_bb in fileObj.rectangles:
            total_GT += 1  # add a ground truth to the total tally
            for p_bb in rescaled_proposed_bb:
                temp_TP_count = 0
                # compute the intersection over union and display it
                iou = bb_intersection_over_union(gt_bb, p_bb)
                print(iou)
                if iou >= 0.5:
                    #print("this proposed_bb has a match")
                    total_TP += 1  # add a true positive to the total tally
                    if temp_TP_count == 1:
                        print("[Warning] two proposed bounding boxes matched. This shouldn't happen as "
                              "there is only one ground truth. Probably overlapping bounding boxes")
                    else:
                        self.average_iou += iou
                    temp_TP_count += 1

                else:
                    #print("the proposed_bb didn't get a match")
                    total_FN += 1



        # # iterate through proposed bb to see if any align with a ground truth bb
        # # if none do then False Positive
        # # if More than one does note that there is an overlap issue.
        # for gt_bb in fileObj.rectangles:
        #     total_GT += 1   # add a ground truth to the total tally
        #     best_iou = 0
        #     for p_bb in rescaled_proposed_bb:
        #         # compute the intersection over union and display it
        #         iou = bb_intersection_over_union(gt_bb, p_bb)
        #         if iou >= best_iou:
        #             best_iou = iou
        #     if best_iou >= 0.5:
        #         print("this proposed_bb has a match")
        #         total_TP += 1   # add a true positive to the total tally
        #     else:
        #         print("the proposed_bb didn't get a match")
        #         total_FP += 1
        #
        # # iterate through ground truth bb to see if any align with a prosed bb
        # # if none do then False Negative
        # # if More than one does note that there is an overlap issue.
        # for p_bb in rescaled_proposed_bb:
        #     total_P += 1
        #     best_iou = 0
        #     for gt_bb in fileObj.rectangles:
        #         # compute the intersection over union and display it
        #         iou = bb_intersection_over_union(gt_bb, p_bb)
        #         if iou >= best_iou:
        #             best_iou = iou
        #     if best_iou >= 0.5:
        #         print("this gt has a match")
        #     else:
        #         print("the proposed_bb didn't get a match")
        #         total_FN += 1

        return total_GT, total_P, total_TP, total_FP, total_FN

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


    def cascade_classifer(self, img):
        # detection of car wheel locations, high recall low precision/high true positives and high false positives
        rectangles = self.cascade_wheel.detectMultiScale(img)
        return rectangles

    def draw_rectangles(self, haystack_img, rectangles):
        # these colors are actually BGR
        line_color_wheel = (0, 255, 0)
        line_color_notwheel = (0, 0, 255)
        line_type = cv2.LINE_4

        for (x, y, w, h) in rectangles:
            # determine the box positions
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # draw the box
            cv2.rectangle(haystack_img, top_left, bottom_right, line_color_wheel, lineType=line_type)

        cv2.imshow("cascade classifier image", haystack_img)
        cv2.waitKey(0)


# object that describes a single line of the bounding box CSV file
class fileItem:
    def __init__(self, fileLine):

        # seperate the contents on the file by space
        splitContent = np.array(splitItem(fileLine))

        self.name = "data/Cascade_classifier_dataset/" + splitContent[0]           # path file name
        #print(self.name)
        self.rectangles = []                    # list of bounding boxes. held in a list of 4 numbers

        # get bounding boxes
        self.getBoundingBoxes(splitContent)

    # extractes the bounding boxes of a file
    def getBoundingBoxes(self,fileItem):

        fileItem = fileItem[1:].astype(int)     # REMEMBER DOING THIS REMOVES THE 0 index which is the file path
        # fileItem[1] is the number of bounding boxes
        for x in range(fileItem[0]):
            rec = (fileItem[1 + 4 * x], fileItem[2+4*x], fileItem[3+4*x], fileItem[4+4*x])
            self.rectangles.append(rec)
        return self.rectangles

# creates wheel images by using bounding boxes from pos and the image paths
read_POStxt_file_and_generate_wheels = fileRead()
