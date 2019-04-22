import argparse

from keras.utils import multi_gpu_model
from keras.optimizers import SGD
from keras.datasets import cifar10
from models.ResNet import ResNet
from keras.models import save_model, load_model
import numpy as np
from sklearn.preprocessing import LabelBinarizer


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, type=str, help="path to specific model checkpoint to load")
args = vars(ap.parse_args())

BASE_LR = 1e-1
BATCH_SIZE = 272 * 2

print("[INFO] loading cifar-10 data...")
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis=0)
std = np.std(trainX, axis=0)
testX = (testX - mean) / std

lb = LabelBinarizer()
testY = lb.fit_transform(testY)


model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (128, 128, 256, 512), reg=5e-4)
multi_model = multi_gpu_model(model, gpus=2, cpu_merge=True)

print(f"[INFO] loading {args['model']}...")
opt = SGD(lr=BASE_LR, momentum=0.9)
multi_model.load_weights(args["model"], by_name=True)
multi_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

_, accuracy = multi_model.evaluate(testX, testY, batch_size=BATCH_SIZE)
print(f"Accuracy:{accuracy}")

print("[INFO] storing the best model...")
save_model(multi_model, filepath="output/model.h5")

print("[INFO] loading the best model...")
new_model = load_model(filepath="output/model.h5")

_, accuracy = new_model.evaluate(testX, testY, batch_size=BATCH_SIZE)
print(f"New Accuracy:{accuracy}")
