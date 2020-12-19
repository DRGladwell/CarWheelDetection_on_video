# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from setup import config
from imutils import paths
import numpy as np
import pickle
import random
import os
import csv
# load the MobileNet network and initialize the label encoder
print("[INFO] loading network...")
model = MobileNet(weights="imagenet", include_top=False)
#model.summary()
le = None


# grab all image paths in the current config.TRAIN
print("[INFO] processing '{} '...".format(config.TRAIN))
p = os.path.sep.join([config.BASE_PATH, config.TRAIN])
imagePaths = list(paths.list_images(p))

# randomly shuffle the image paths and then extract the class
# labels from the file paths
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-2] for p in imagePaths]

# if the label encoder is None, create it
if le is None:
    le = LabelEncoder()
    le.fit(labels)
# open the output CSV file for writing
csvPath = os.path.sep.join([config.BASE_CSV_PATH,"{}.csv".format(config.TRAIN)])

# write in the fierld names of the CSV, not the length of the header depends on the CNN output
cols = ["feat_{}".format(i) for i in range(0, 7 * 7 * 1024)]
fieldnames = ["class"] + cols

with open(csvPath, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(fieldnames)

    # loop over the images in batches
    for (b, i) in enumerate(range(0, len(imagePaths), config.BATCH_SIZE)):
        # extract the batch of images and labels, then initialize the
        # list of actual images that will be passed through the network
        # for feature extraction
        print("[INFO] processing batch {}/{}".format(b + 1,
                                                     int(np.ceil(len(imagePaths) / float(config.BATCH_SIZE)))))
        batchPaths = imagePaths[i:i + config.BATCH_SIZE]
        batchLabels = le.transform(labels[i:i + config.BATCH_SIZE])
        batchImages = []

        # loop over the images and labels in the current batch
        for imagePath in batchPaths:
            # load the input image using the Keras helper utility
            # while ensuring the image is resized to 224x224 pixels
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)
            # preprocess the image by (1) expanding the dimensions and
            # (2) subtracting the mean RGB pixel intensity from the
            # ImageNet dataset
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            # add the image to the batch
            batchImages.append(image)

            # pass the images through the network and use the outputs as
            # our actual features, then reshape the features into a
            # flattened volume
            batchImages_np = np.vstack(batchImages)
            features = model.predict(batchImages_np, batch_size=config.BATCH_SIZE)
            features = features.reshape((features.shape[0], 7 * 7 * 1024))
            ## making a second variable batchImages_np is neccessary as batchImages.append doesn't work on np arrays.

            # loop over the class labels and extracted features
            for (label, vec) in zip(batchLabels, features):
                # construct a row that exists of the class label and
                # extracted features
                vec = ",".join([str(v) for v in vec])
                outcsv.write("{},{}\n".format(label, vec))
# close the CSV file
outcsv.close()
# serialize the label encoder to disk
f = open(config.LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()