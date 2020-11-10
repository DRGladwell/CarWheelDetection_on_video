# import the necessary packages
import pickle
from tensorflow.keras.applications import MobileNet
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import multiprocessing
import concurrent.futures


class wheelDetectorTransferLearning():
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


    def testImage(self,img):
        #print("begin single CNN+ML")
        prepared_image = self.prepare_image(img)
        flat_feature_vector = self.get_cnn_feature(prepared_image)

        dictionary = {}
        for item in range(len(self.fieldnames)):
            dictionary[self.fieldnames[item]] = flat_feature_vector[item]

        # make predictions on the current set of features,
        output = self.ML_model.predict_proba_one(dictionary)
        #print("finish single CNN+ML")
        return output


    # def run_cnn_ml_process(self, prepared_image):
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         f1 = executor.submit(runML, prepared_image)
    #         return f1.result()

    # def runCNNProcess(self, dictionary):
    #     print("begin single PROCESS")
    #     manager = multiprocessing.Manager()
    #     #return_dict = manager.dict()
    #     return_output = manager.Array()
    #     # extracts the image patch of a bounding box and runs a cnn classifier on it
    #     p1 = multiprocessing.Process(target=self.runML, args=[dictionary,return_output])
    #     p1.start()
    #     p1.join()  #if you want to make sure the process finishes before continuing
    #     print(return_output)
    #     return p1



# # load the trained convolutional neural network
# model = None
# #  A necessary evil so multiprocessing can happen. Multiprocessing
# #  doesn't work on instance functions as they can't be pickled
# with open('models/mobileNet4wheels.cpickle', 'rb') as f:
#     model = pickle.load(f)
#
# # load the MobileNet network and initialize the label encoder
# print("[INFO] loading MobileNet network...")
# MobileNet = MobileNet(weights="imagenet", include_top=False)
#
#
# # create the keynames of the dictionary
# fieldnames = ["feat_{}".format(i) for i in range(0, 7 * 7 * 1024)]
#
# dictionary = {}
#
# def runML(prepared_image):
#     features = MobileNet.predict(prepared_image)
#     features = features.reshape((features.shape[0], 7 * 7 * 1024))
#     flat_feature_vector = features[0]
#
#     for item in range(len(fieldnames)):
#         dictionary[fieldnames[item]] = flat_feature_vector[item]
#
#     # make predictions on the current set of features, train the
#     # model on the features, and then update our metric
#     output = model.predict_proba_one(dictionary)
#     dictionary.clear()
#     return output

