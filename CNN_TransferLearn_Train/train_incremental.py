# USAGE EXAMPLE
# python train_incremental.py --csv train.csv -n 50176

# import the necessary packages
import pickle
from creme.linear_model import LogisticRegression
from creme import optim
from creme import model_selection
from creme.preprocessing import StandardScaler
from creme.compose import Pipeline
from creme.metrics import Accuracy
from creme import stream
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--csv", required=True,
	help="path to features CSV file")
ap.add_argument("-n", "--cols", type=int, required=True,
	help="# of feature columns in the CSV file (excluding class column")
args = vars(ap.parse_args())

# construct our data dictionary which maps the data types of the
# columns in the CSV file to built-in data types
print("[INFO] building column names...")
types = {"feat_{}".format(i): float for i in range(0, args["cols"])}
types["class"] = int

# create a CSV data generator for the extracted Keras features
dataset = stream.iter_csv(args["csv"], target="class", converters=types)
# construct our pipeline (maybe set to .0000003)
model = Pipeline(StandardScaler(), LogisticRegression(optimizer=optim.SGD(.0000001)))

# initialize our metric
print("[INFO] starting training...")
metric = Accuracy()


# loop over the dataset
for (i, (X, y)) in enumerate(dataset):
	# make predictions on the current set of features, train the
	# model on the features, and then update our metric
	preds = model.predict_one(X)
	model = model.fit_one(X, y)
	metric = metric.update(y, preds)
	print("INFO] update {} - {}".format(i, metric))
	if i == 2500:
		break
# show the accuracy of the model
print("[INFO] final - {}".format(metric))

print("[INFO] saving model...")
f = open("model4.cpickle", "wb")
f.write(pickle.dumps(model))
f.close()
