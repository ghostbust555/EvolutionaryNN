import random

from layertype import LayerType


class Individual:
    def __init__(self, dna=None):
        self.score = 0
        self.norm_score = 0
        if dna:
            self.dna = dna
        else:
            r = random.randrange(2, 10)
            self.dna = [LayerType.random_layer() for _ in range(r)]

    def get_fitness(self):
        return self.score
