import random
from enum import Enum, unique


class Loss(Enum):
    mean_squared_error = 1
    mean_absolute_error = 2
    mean_absolute_percentage_error = 3
    mean_squared_logarithmic_error = 4
    binary_crossentropy = 5
    categorical_crossentropy = 6
    sparse_categorical_crossentropy = 7
    kullback_leibler_divergence = 8
    poisson = 9
    cosine_proximity = 10

    @staticmethod
    def random_loss():
        return Loss(random.randrange(1, 10))


@unique
class LayerType(Enum):
    conv = 1
    relu = 2
    softmax = 3
    tanh = 4
    sigmoid = 5
    dropout = 6
    maxpool = 7
    loss = 8
    optimizer = 9

    @staticmethod
    def random_layer():
        return LayerType(random.randrange(1, 7)), random.randrange(10, 1000)

    @staticmethod
    def loss_layer():
        return LayerType(LayerType.loss), Loss.random_loss()
