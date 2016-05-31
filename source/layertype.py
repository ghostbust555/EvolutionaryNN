import random
from enum import Enum, unique


@unique
class LayerType(Enum):
    conv = 1
    relu = 2
    softmax = 3
    tanh = 4
    sigmoid = 5
    dropout = 6
    maxpool = 7

    @staticmethod
    def random_layer():
        return LayerType(random.randrange(1, 7)), random.randrange(10, 1000)
