from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers import Input, add
from keras.models import Model
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
from keras.constraints import unit_norm
from keras import backend as K
from keras.utils import plot_model


class ResNet:

    @staticmethod
    def residual_module(data, K, stride, chan_dim, reduce=False, reg=5e-4, bn_eps=2e-5, bn_mom=0.9):
        shortcut = data

        bn1 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(data)
        act1 = Activation("elu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        bn2 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(conv1)
        act2 = Activation("elu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        bn3 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(conv2)
        act3 = Activation("elu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        if reduce:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        return add([conv3, shortcut])

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=1e-4, bn_eps=2e-5, bn_mom=0.9, dataset="cifar"):
        input_shape = (height, width, depth)
        chan_dim = -1

        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(inputs)

        if dataset == "cifar":
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)

        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i+1], stride, chan_dim, reduce=True, bn_eps=bn_eps, bn_mom=bn_mom)

            for j in range(0, stages[i] - 1):
                x = ResNet.residual_module(x, filters[i+1], (1, 1), chan_dim, bn_eps=bn_eps, bn_mom=bn_mom)

        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom)(x)
        x = Activation("elu")(x)
        x = AveragePooling2D((8, 8))(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        return Model(inputs, x, name="resnet")

#model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (128, 128, 256, 512))
#plot_model(model, to_file="model.png", show_shapes=True)




