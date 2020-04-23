from multiprocess.managers import BaseManager


# Grab the shared proxy class.  All methods in that class will be availble here
class DataClient(object):
    def __init__(self, port=5000):
        class myManager(BaseManager): pass
        myManager.register('DataProxy')
        self.mgr = myManager(address=('localhost', port), authkey='DataProxy01')
        self.mgr.connect()
        self.proxy = self.mgr.DataProxy()