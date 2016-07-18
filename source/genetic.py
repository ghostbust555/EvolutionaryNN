import random
import time
from itertools import chain

from individual import Individual
from geneticqueue import GeneticQueue
from layertype import LayerType, Loss


def process_input(x):
    x.get_fitness()
    return x


class Genetic:

    def __init__(self, pop_size=15, extinction_timer=5):
        self.epoch_count = 0
        self.best = Individual()
        self.best.dna = []
        self.best.score = 0
        self.init_pop_size = pop_size
        self.population = [Individual() for _ in range(pop_size)]
        self.workQueue = GeneticQueue()
        self.results = []
        self.extinction_event = extinction_timer

    def run(self, goal=.95, max_epochs=100):
        for x in range(max_epochs):
            self.epoch()
            print('population size = {0}'.format(len(self.population)))
            for individual in self.population:
                if individual.score >= goal:
                    return individual

    def epoch(self, goal=None):
        return self.__epoch_parallel(goal)

    def __epoch_serial(self, goal):
        print("Beginning evolutionary epoch {}".format(self.epoch_count))

        for x in self.population:
            x.get_fitness()
            if x.score > self.best.score:
                self.best.dna = x.dna
                self.best.score = x.score
                print("New Best! Score = {} Dna = {}".format(x.score, x.dna))
            if goal is not None and x.score >= goal:
                return x

        print("epoch {}".format(self.epoch_count))
        for x in self.population:
            print("{} - {}".format(x.dna, x.score))

        self.selection()
        self.epoch_count += 1

        return None

    def __epoch_parallel(self, goal):
        self.workQueue.clear()

        for x in self.population:
            self.workQueue.enqueue(x)

        self.wait_for_results(self.population)

        for x in self.population:
            if x.score > self.best.score:
                self.best.dna = x.dna
                self.best.score = x.score
                print("New Best! Score = {} Dna = {}".format(x.score, x.dna))
            if goal is not None and x.score >= goal:
                return x

        print("epoch {}".format(self.epoch_count))
        for x in self.population:
            print("{} - {}".format(x.dna, x.score))

        self.selection(self.epoch_count)
        self.epoch_count += 1

        return None

    def wait_for_results(self, pop):
        self.results = []
        while len(pop) > len(self.results):
            time.sleep(1)

        self.population = self.results

        print("Finished waiting for reslts")

    @staticmethod
    def mutate(seed: Individual):

        from layertype import LayerType
        y = [None] * len(seed.dna)

        # loop through and mutate randomly
        for idx, x in enumerate(seed.dna):
            if x[0] != LayerType.loss:
                r = random.random()
                if r < .02:
                    y[idx] = LayerType.random_layer()
                elif r < .2:
                    y[idx] = x[0], max(1, int(random.gauss(x[1], 100)))
                else:
                    y[idx] = x
            else:
                y[idx] = x

        # randomly shorten or lengthen dna
        r = random.random()
        if r < .02:
            from layertype import LayerType
            y = [LayerType.random_layer()]+y
        elif r < .04:
            del y[random.randrange(0, max(len(y) - 2, 1))]

        seed.dna = y
        return seed

    @staticmethod
    def mate(a: Individual, b: Individual):

        min_len = (min(len(a.dna), len(b.dna)))

        if min_len > 2 and random.random() < .9:
            cross_point = random.randrange(1, min_len - 1)

            acopy = []
            bcopy = []

            for x in a.dna:
                acopy.append(x)

            for x in b.dna:
                bcopy.append(x)

            new_a_dna = acopy[:cross_point] + bcopy[cross_point:]
            new_b_dna = bcopy[:cross_point] + acopy[cross_point:]

            for x in range(0, min_len):
                if new_a_dna[x][0] != LayerType.loss and new_b_dna[x][0] != LayerType.loss and random.random() < .5:
                    new_a_dna[x] = new_a_dna[x][0], int(round((new_a_dna[x][1] + new_a_dna[x][1] + new_b_dna[x][1]) / 3))
                elif new_a_dna[x][0] == LayerType.loss and random.random() < .5:
                    new_a_dna[x] = new_a_dna[x][0], Loss.random_loss()

            return Individual(new_a_dna), Individual(new_b_dna)
        else:
            return a, b

    def selection(self, epoch: int):
        elite_count = 2
        keep_threshold = .5

        s = max(sum(c.score for c in self.population), .01)

        for individual in self.population:
            individual.norm_score = individual.score / s

        self.population.sort(key=lambda c: c.score, reverse=True)

        if epoch > 0 and epoch % self.extinction_event == 0:
            print("Extinction Event...")

            keep = int(self.init_pop_size * .7)
            new_pop = self.init_pop_size - keep

            self.population = self.population[:keep]

            for i in range(0, new_pop):
                self.population.append(Individual())

            cs = list(map(lambda c: self.mate(c, random.choice(self.population)), self.population))

            self.population = list(map(lambda c: self.mutate(c), list(chain.from_iterable(cs))))
        else:
            total = 0
            i = 0
            for individual in self.population:
                total += individual.norm_score
                i += 1
                if total > keep_threshold:
                    break

            elite = self.population[:elite_count]
            self.population = self.population[:i]

            cs = list(map(lambda c: self.mate(c, random.choice(self.population)), self.population))

            self.population = elite + list(map(lambda c: self.mutate(c), list(chain.from_iterable(cs))))
