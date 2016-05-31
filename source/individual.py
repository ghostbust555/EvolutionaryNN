import random

from layertype import LayerType
from neural import Neural


class Individual:
    def __init__(self, dna=None):
        self.score = 0
        self.norm_score = 0
        if dna:
            self.dna = dna
        else:
            r = random.randrange(2, 10)
            self.dna = [LayerType.random_layer() for _ in range(r)]

    def __run_network_(self):
        # s = 1

        # for x in self.dna:
        #     if x:
        #         s += 1

        try:
            score = Neural(self).run_network()
        except:
            raise
            return 0

        return score

    def get_fitness(self):
        self.score = self.__run_network_()
        return self.score

    def get_network(self):
        for x in self.dna:
            if x:
                print("softmax ", end="")
            else:
                print("relu ", end="")
