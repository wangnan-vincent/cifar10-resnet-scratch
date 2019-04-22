import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from models.ResNet import ResNet

from utils.callbacks import EpochCheckPoint, TrainingMonitor

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler

import numpy as np
import argparse

BASE_LR = 1e-1
EPOCHS = 120
BATCH_SIZE = 272 * 2


def poly_decay(epoch):
    max_epochs = EPOCHS
    base_lr = BASE_LR
    power = 3.0
    return base_lr * (1 - epoch / float(max_epochs)) ** power


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="pass to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to specific model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

print("[INFO] loading cifar-10 data...")
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis=0)
std = np.std(trainX, axis=0)
trainX = (trainX - mean) / std
testX = (testX - mean) / std


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

aug = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3, zoom_range=0.3, rotation_range=20,
                         shear_range=0.1, horizontal_flip=True, fill_mode="nearest")

model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (128, 128, 256, 512), reg=5e-4)
multi_model = multi_gpu_model(model, gpus=2, cpu_merge=True)


if args["model"] is None:
    print("[INFO] compiling model...")
    opt = SGD(lr=BASE_LR, momentum=0.9)
else:
    print(f"[INFO] loading {args['model']}...")
    opt = SGD(lr=BASE_LR, momentum=0.9)
    multi_model.load_weights(args["model"], by_name=True)

multi_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

callbacks = [
    EpochCheckPoint(args["checkpoints"], every=1, model=model, start_at=args["start_epoch"]),
    TrainingMonitor("output/resnet56_cifar10.png", json_path="output/resnet56_cifar10.json", start_at=args["start_epoch"]),
    LearningRateScheduler(poly_decay)
]

print("[INFO] training network...")
TRAIN_STEPS = len(trainY) // BATCH_SIZE
TEST_STEPS = len(testY // BATCH_SIZE)
multi_model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                          steps_per_epoch=TRAIN_STEPS,
                          epochs=EPOCHS, callbacks=callbacks,
                          validation_data=(testX, testY),
                          validation_steps=TEST_STEPS,
                          max_queue_size=BATCH_SIZE)



