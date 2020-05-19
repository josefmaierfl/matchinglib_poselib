from multiprocessing.managers import BaseManager


# Grab the shared proxy class.  All methods in that class will be availble here
class DataClient(object):
    def __init__(self, load_path, test_name, test_nr, cpu_use, message_path, port=5000, ctx=None):
        class myManager(BaseManager): pass
        myManager.register('DataProxy')
        if ctx:
            self.mgr = myManager(address=('localhost', port), authkey=b'DataProxy01', ctx=ctx)
        else:
            self.mgr = myManager(address=('localhost', port), authkey=b'DataProxy01')
        self.mgr.connect()
        self.proxy = self.mgr.DataProxy(load_path, test_name, test_nr, cpu_use, message_path)