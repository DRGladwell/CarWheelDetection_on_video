# import the necessary packages
# code to take training images with bounding boxes and export a file of wheel images (content of the bounding boxes ..)
import pickle
from tensorflow.keras.applications import MobileNet
import numpy as np
import tensorflow as tf
import cv2
import imutils
from imutils import paths
import argparse
from tensorflow.keras.preprocessing import image

# example: python Cascade_and_CNN_test.py -f data/Cascade_classifier_and_CNN_dataset/posObjectDetection.txt -i data/Cascade_classifier_and_CNN_dataset/testset_negatives/


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", help="path to the pos.txt file",
                required=True)  # carful pos.txt has the right image paths
ap.add_argument("-i", "--images", required=True, help="Path to the negative images")

# take in the test images
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["images"]))


# Returns float value describing how well the predicted and ground truth bounding boxes over lapped.
def bb_intersection_over_union(bb_a, bb_b):
    x_a, y_a, w_a, h_a = bb_a
    x_b, y_b, w_b, h_b = bb_b

    boxA = [x_a, y_a, x_a + w_a, y_a + h_a]
    boxB = [x_b, y_b, x_b + w_b, y_b + h_b]

    # determine the (x, y)-coordinates of the intersection rectangle
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
        self.wheels = []  # will contain images of wheels extracted from original set of images

        self.cascade_width = 300  # pixel with of cascade classifier input images
        self.test_cnn = CNN()

        self.average_iou = 0

        GT = 0  # ground truth
        P = 0  # proposed
        TP = 0  # true positive
        FP = 0  # false positive
        FN = 0  # false negative

        # Beginning of the script
        self.readInputFile()  # generates the paths split line by line

        # generate fileItem objects
        for line in self.fileContents:
            self.fileItemsObj.append(fileItem(line))

        # extract img of car wheel img from bounding box information
        # for fileObj in self.fileItemsObj:
        #     results = self.compare_cascade_to_ground_truth(fileObj)
        #     total_GT, total_P, total_TP, total_FP, total_FN = results

            # GT += total_GT
            # P += total_P
            # TP += total_TP
            # FP += total_FP
            # FN += total_FN

        # print("The total number of ground truths bounding boxes: " + str(GT))
        # print("The total number of proposed: " + str(P))
        # print("The total number of true positives: " + str(TP))
        # print("The total number of false positives: " + str(FP))
        # print("The total number of false negatives: " + str(FN))
        #
        # print("the average iou = " + str(self.average_iou / TP))

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
            rescaled_proposed_bb = self.boundingBoxRescale(proposed_bb, ratio2workedimg)

            # the first bounding box should be the first potential car wheel
            potential_car_wheels = self.extractROI(img, rescaled_proposed_bb)

            is_carwheel_bb = []
            for iteration, croped_img in enumerate(potential_car_wheels):
                img = self.test_cnn.prepare_image(croped_img)
                flat_feature_vector = self.test_cnn.get_cnn_feature(img)
                result = self.test_cnn.testImage(flat_feature_vector)
                print(result)
                if result[True] >= 0.50:
                    is_carwheel_bb.append(rescaled_proposed_bb[iteration])

            number_of_bb = len(is_carwheel_bb)
            if number_of_bb > 0:
                total_FP += number_of_bb
                #self.draw_rectangles(resize_img, proposed_bb)
            else:
                total_TN += 1


        print("the number of negative images: " + str(total_number_of_negatives))
        print("the number of true negatives: " + str(total_TN))
        print("the number of false positives: " + str(total_FP))

    # # extracts the the image inside the bounding box
    def compare_cascade_to_ground_truth(self, fileObj):
        img_original = cv2.imread(fileObj.name)
        # cv2.imshow("cascade classifier image", img)
        # cv2.waitKey(0)

        ratio2workedimg = img_original.shape[1] / self.cascade_width
        # print(ratio2workedimg)
        resize_img = imutils.resize(img_original, self.cascade_width)

        proposed_bb = self.cascade_classifer(resize_img)
        rescaled_proposed_bb = self.boundingBoxRescale(proposed_bb, ratio2workedimg)

        # self.draw_rectangles(img, rescaled_proposed_bb)
        # self.draw_rectangles(img, fileObj.rectangles)
        # print(rescaled_proposed_bb)

        # the first bounding box should be the first potential car wheel
        potential_car_wheels = self.extractROI(img_original, rescaled_proposed_bb)

        is_carwheel_bb = []
        for iteration, croped_img in enumerate(potential_car_wheels):
            img = self.test_cnn.prepare_image(croped_img)
            flat_feature_vector = self.test_cnn.get_cnn_feature(img)
            result = self.test_cnn.testImage(flat_feature_vector)
            print(result)
            if result[True] >= 0.50:
                is_carwheel_bb.append(rescaled_proposed_bb[iteration])

        total_GT = 0  # ground truth
        total_P = 0  # proposed
        total_TP = 0  # true positive
        total_FP = 0  # false positive
        total_FN = 0  # false negative

        # Use this as there's only one bounding box per image
        # USED ON POSITIVE DATA SET
        for gt_bb in fileObj.rectangles:
            total_GT += 1  # add a ground truth to the total tally
            temp_TP_count = 0
            for p_bb in is_carwheel_bb:
                # compute the intersection over union and display it
                iou = bb_intersection_over_union(gt_bb, p_bb)
                print(iou)
                if iou >= 0.50:
                    # print("this proposed_bb has a match")
                    total_TP += 1  # add a true positive to the total tally
                    if temp_TP_count == 1:
                        print("[Warning] two proposed bounding boxes matched. This shouldn't happen as "
                              "there is only one ground truth. Probably overlapping bounding boxes")
                    else:
                        self.average_iou += iou
                    temp_TP_count += 1

                else:
                    temp_bb = []
                    temp_bb.append(p_bb)
                    # print("the proposed_bb didn't get a match")
                    self.draw_rectangles(img_original, temp_bb)
                    total_FP += 1
            if temp_TP_count == 0:
                total_FN += 1
                self.draw_rectangles(img_original, fileObj.rectangles)


        return total_GT, total_P, total_TP, total_FP, total_FN

    # extract a region of image to feed into the CNN+ML
    def extractROI(self, image, bounding_boxes):
        potential_wheels = []
        for rec in bounding_boxes:
            (x, y, w, h) = rec
            cropImg = image[y:y + h, x:x + w]
            potential_wheels.append(cropImg)  # this is an image object(so a set of arrays)
        # return a list of image patches
        return potential_wheels

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

        self.name = "data/Cascade_classifier_and_CNN_dataset/" + splitContent[0]  # path file name
        # print(self.name)
        self.rectangles = []  # list of bounding boxes. held in a list of 4 numbers

        # get bounding boxes
        self.getBoundingBoxes(splitContent)

    # extractes the bounding boxes of a file
    def getBoundingBoxes(self, fileItem):
        fileItem = fileItem[1:].astype(int)  # REMEMBER DOING THIS REMOVES THE 0 index which is the file path
        # fileItem[1] is the number of bounding boxes
        for x in range(fileItem[0]):
            rec = (fileItem[1 + 4 * x], fileItem[2 + 4 * x], fileItem[3 + 4 * x], fileItem[4 + 4 * x])
            self.rectangles.append(rec)
        return self.rectangles

class CNN():
    def __init__(self):
        #load the trained convolutional neural network
        self.ML_model = None
        with open('models/mobileNet4wheels.cpickle', 'rb') as f:
            self.ML_model = pickle.load(f)
        # load the MobileNet network and initialize the label encoder
        print("[INFO] loading MobileNet network...")
        self.MobileNet = MobileNet(weights="imagenet", include_top=False)


        # create the keynames of the dictionary
        self.fieldnames = ["feat_{}".format(i) for i in range(0, 7 * 7 * 1024)]



    def prepare_image(self,img):
        img = cv2.resize(img, (224, 244))
        img = tf.keras.preprocessing.image.array_to_img(img, data_format=None, scale=True, dtype=None)
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        prepared_image = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
        return prepared_image

    def get_cnn_feature(self, prepared_image):
        features = self.MobileNet.predict(prepared_image)
        flat_feature_vector = features.reshape(7 * 7 * 1024)
        return flat_feature_vector


    def testImage(self, flat_feature_vector):
        dictionary = {}
        for item in range(len(self.fieldnames)):
            dictionary[self.fieldnames[item]] = flat_feature_vector[item]

        # make predictions on the current set of features,
        output = self.ML_model.predict_proba_one(dictionary)
        return output

# creates wheel images by using bounding boxes from pos and the image paths
read_POStxt_file_and_generate_wheels = fileRead()
