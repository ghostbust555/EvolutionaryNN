import random
from itertools import chain
from individual import Individual


def process_input(x):
    x.get_fitness()
    return x


class Genetic:
    def __init__(self):
        self.epoch_count = 0
        self.best = Individual()
        self.best.dna = []
        self.best.score = 0
        self.population = [Individual() for _ in range(2)]

    def run(self, goal=.95, max_epochs=100):
        for x in range(max_epochs):
            self.epoch()
            for p in self.population:
                if p.score >= goal:
                    return p.get_network()

    def epoch(self, goal=None):
        return self.__epoch_serial(goal)

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
        from joblib import Parallel, delayed
        import multiprocessing

        num_cores = multiprocessing.cpu_count()

        results = Parallel(n_jobs=num_cores, backend="threading")(delayed(process_input)(i) for i in self.population)

        self.population = results
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

        self.selection()
        self.epoch_count += 1

        return None

    @staticmethod
    def mutate(seed: Individual):
        y = [None] * len(seed.dna)

        # loop through and mutate randomly
        for idx, x in enumerate(seed.dna):
            r = random.random()
            if r < .01:
                from layertype import LayerType
                y[idx] = LayerType.random_layer()
            elif r < .03:
                y[idx] = x[0], random.randrange(10, 1000)
            else:
                y[idx] = x

        # randomly shorten or lengthen dna
        r = random.random()
        if r < .02:
            from layertype import LayerType
            y.append(LayerType.random_layer())
        elif r < .04:
            del y[-1]

        seed.dna = y
        return seed

    @staticmethod
    def mate(a: Individual, b: Individual):
        if random.random() < .9:
            min_len = (min(len(a.dna), len(b.dna)))
            cross_point = random.randrange(1, min_len)

            acopy = []
            bcopy = []

            for x in a.dna:
                acopy.append(x)

            for x in b.dna:
                bcopy.append(x)

            new_a_dna = acopy[:cross_point] + bcopy[cross_point:]
            new_b_dna = bcopy[:cross_point] + acopy[cross_point:]

            for x in range(0, min_len):
                if random.random() < .5:
                    new_a_dna[x] = new_a_dna[x][0], int(round((new_a_dna[x][1] + new_a_dna[x][1] + new_b_dna[x][1]) / 3))

            return Individual(new_a_dna), Individual(new_b_dna)
        else:
            return a, b

    def selection(self):
        elite_count = 2
        keep_threshold = .6

        s = sum(c.score for c in self.population)

        for x in self.population:
            x.norm_score = x.score / s

        self.population.sort(key=lambda c: c.score, reverse=True)

        total = 0
        i = 0
        for x in self.population:
            total += x.norm_score
            i += 1
            if total > keep_threshold:
                break

        elite = self.population[:elite_count]
        self.population = self.population[:i]

        cs = list(map(lambda c: self.mate(c, random.choice(self.population)), self.population))

        children = elite + list(map(lambda c: self.mutate(c), list(chain.from_iterable(cs))))

        self.population = children
