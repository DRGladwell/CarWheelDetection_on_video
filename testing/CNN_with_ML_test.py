# import the necessary packages
import pickle
from tensorflow.keras.applications import MobileNet
import numpy as np
import tensorflow as tf
import cv2
import imutils
from imutils import paths
import argparse
from tensorflow.keras.preprocessing import image


# example:  python CNN_with_ML_test.py -i data/CNN_with_ML_dataset/CNN_wheels_test/

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to input directory of images")
args = vars((ap.parse_args()))

# grab the paths to the images
imagePaths = list(paths.list_images(args["images"]))

class CNN_tester():
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

    def runCNN_and_ML_on_dataset(self):
        carwheels = 0
        none_carwheels = 0
        for image in imagePaths:
            img = image
            img = cv2.imread(img)
            img = self.prepare_image(img)
            flat_feature_vector = self.get_cnn_feature(img)
            result = self.testImage(flat_feature_vector)
            print(result)

            if result[True] >= 0.5:
                carwheels += 1
            else:
                none_carwheels += 1

        print("the number of car wheels: " + str(carwheels))
        print("the number of none car wheels: " + str(none_carwheels))
        print("total number of images: " + str(len(imagePaths)))
        return carwheels, none_carwheels

    def display_image(self):
        for image in imagePaths:
            img = image
            img = cv2.imread(img)
            cv2.imshow("Car wheel Detection", img)
            cv2.waitKey(0)


cnn_tester = CNN_tester()
cnn_tester.runCNN_and_ML_on_dataset()
#cnn_tester.display_image()