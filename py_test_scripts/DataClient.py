"""
Released under the MIT License - https://opensource.org/licenses/MIT

Copyright (c) 2020 Josef Maier

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)
"""
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