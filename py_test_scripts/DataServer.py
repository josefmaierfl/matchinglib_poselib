# from multiprocess.managers import SyncManager
# multiprocessing.set_start_method('spawn')
from multiprocessing import get_context
import time
from multiprocessing.managers import SyncManager
from multiprocessing import Queue
# from multiprocessing import Process as proc
import numpy as np
import pandas as pd
from eval_tests_main import get_data_files
import queue as queuem
import sys

# Global for storing the data to be served
# gData = pd.DataFrame()


# Proxy class to be shared with different processes
# Don't put big data in here since that will force it to be piped to the
# other process when instantiated there, instead just return a portion of
# the global data when requested.
class DataProxy(object):
    gdata = None

    def __init__(self, load_path, test_name, test_nr, cpu_use, message_path):
        # global gData
        gdata, load_time = get_data_files(load_path, test_name, test_nr, cpu_use, message_path)
        if gdata.empty:
            sys.exit(1)

    def getData(self, key=None, default=None):
        # global gData
        if key:
            return self.gData.loc[:, key]
        else:
            return self.gData


class myManager(SyncManager):
    pass


myManager.register('DataProxy', DataProxy)


def set_data(df):
    global gData
    gdata = df


#def start_server(load_path, test_name, test_nr, cpu_use, message_path, mutex, port=5000, ctx=None):
def start_server(port=5000, ctx=None):
    # global gData
    # mutex.acquire()
    # gdata, load_time = get_data_files(load_path, test_name, test_nr, cpu_use, message_path)
    # if gdata.empty:
    #     mutex.release()
    #     return 1

    # Start the server on address(host,port)
    if ctx:
        mgr = myManager(address=('', port), authkey=b'DataProxy01', ctx=ctx)
    else:
        # ctx = get_context('forkserver')
        mgr = myManager(address=('', port), authkey=b'DataProxy01')
    # mgr.start()
    # return mgr, ctx
    server = mgr.get_server()
    # mutex.release()
    server.serve_forever()


def shutdown_server(mgr):
    mgr.shutdown()


def data_handler(load_path, test_name, test_nr, cpu_use, message_path, mutex, queue_c, queue_r):
    mutex.acquire()
    data, load_time = get_data_files(load_path, test_name, test_nr, cpu_use, message_path)
    if data.empty:
        return 1
    mutex.release()
    while True:
        try:
            # Get the index we sent in
            idx = queue_c.get(False)
        except queuem.Empty:
            time.sleep(0.2)
            continue
        else:
            if idx == 'finished':
                return 0
            else:
                try:
                    # Send back some data
                    queue_r.put(data)
                except:
                    pass
