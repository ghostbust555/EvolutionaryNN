from genetic import Genetic
from individual import Individual


# def test_theano():
#     import numpy as np
#     import time
#     import theano
#     A = np.random.rand(1000,10000).astype(theano.config.floatX)
#     B = np.random.rand(10000,1000).astype(theano.config.floatX)
#     np_start = time.time()
#     AB = A.dot(B)
#     np_end = time.time()
#     X,Y = theano.tensor.matrices('XY')
#     mf = theano.function([X,Y],X.dot(Y))
#     t_start = time.time()
#     tAB = mf(A,B)
#     t_end = time.time()
#     print("NP time: %f[s], theano time: %f[s] (times should be close when run on CPU!)" %(
#                                                np_end-np_start, t_end-t_start))
#     print("Result difference: %f" % (np.abs(AB-tAB).max(), ))
#

def test_mutate():
    g = Genetic()
    for i in range(100):
        a = Individual()

        o = g.mutate(a)
        assert not o.dna.__contains__(None)


def test_mate():
    g = Genetic()
    for i in range(100):
        a = Individual()
        b = Individual()

        o = g.mate(a, b)
        assert not o[0].dna.__contains__(None)
        assert not o[1].dna.__contains__(None)

