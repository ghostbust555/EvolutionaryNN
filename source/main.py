from genetic import Genetic

from threading import Thread
import cherrypy

from base64 import b64encode, b64decode
from json import dumps, loads, JSONEncoder
import pickle

from individual import Individual


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return super().default(obj)
        return {'_python_object': b64encode(pickle.dumps(obj)).decode('utf-8')}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(b64decode(dct['_python_object'].encode('utf-8')))
    return dct

g = Genetic(15, 5)
t = Thread(target=g.run)
t.start()

class GeneticWebHost(object):
    @cherrypy.expose
    def getDna(self, id, uuid):
        if not g.workQueue.isEmpty():
            x = dumps(g.workQueue.dequeue(), cls=PythonObjectEncoder)
            loads(x, object_hook=as_python_object)
            return x
        else:
            return None

    @cherrypy.expose
    def dnaResult(self, result):
        x = loads(result, object_hook=as_python_object)
        g.results.append(x)
        return "Ok"

cherrypy.config.update({'log.screen': False})
cherrypy.quickstart(GeneticWebHost())


