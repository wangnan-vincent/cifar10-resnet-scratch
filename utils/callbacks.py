from keras.callbacks import BaseLogger, Callback
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import keras.backend as K


class EpochCheckPoint(Callback):
    def __init__(self, output_dir, every, model, start_at=0, ):
        self.output_dir = output_dir
        self.every = every
        self.start_at = start_at
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        cur_epoch = self.start_at + epoch
        if cur_epoch % self.every == 0:
            filepath = os.sep.join([self.output_dir, "weights.{epoch:02d}.h5".format(epoch=cur_epoch)])
            self.model.save_weights(filepath, overwrite=True)

        print("learning rate:", K.eval(self.model.optimizer.lr))


class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path, json_path=None, start_at=0):
        super().__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_at = start_at
        self.H = {}

    def on_train_begin(self, logs={}):
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                if self.start_at > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]

    def on_epoch_end(self, epoch, logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H, cls=MyEncoder))
            f.close()

        size = len(self.H["loss"])
        if size > 1:
            N = np.arange(0, size)
            plt.style.use("ggplot")
            plt.figure()
            plt.subplot(211)
            plt.title(f"Loss and Accuracy, [Epoch {size}]")
            plt.plot(N, self.H["loss"], label="loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.subplot(212)
            plt.plot(N, self.H["acc"], label="acc")
            plt.plot(N, self.H["val_acc"], label="val_acc")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(self.fig_path)
            plt.close()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

