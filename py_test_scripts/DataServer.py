from multiprocess.managers import SyncManager
import numpy as np
import pandas as pd

# Global for storing the data to be served
gData = pd.DataFrame()


# Proxy class to be shared with different processes
# Don't put big data in here since that will force it to be piped to the
# other process when instantiated there, instead just return a portion of
# the global data when requested.
class DataProxy(object):
    def __init__(self):
        pass

    def getData(self, key=None, default=None):
        global gData
        if key:
            return gData.loc[:, key]
        else:
            return gData


def start_server(df, port=5000):
    global gData
    gData = df

    # Start the server on address(host,port)
    class myManager(SyncManager): pass
    myManager.register('DataProxy', DataProxy)
    mgr = myManager(address=('', port), authkey='DataProxy01')
    server = mgr.get_server()
    server.start()