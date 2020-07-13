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
import os, numpy as np
import modin.pandas as pd

tdata_folder = '/home/maierj/work/Sequence_Test/py_test/test'
num_pts = int(5000)
data = {'err1': [0.3, 0.5, 0.7, 0.4, 0.6] * int(num_pts/5),
        'err2': 1000 + np.abs(np.random.randn(num_pts) * 10),
        'err3': 10 + np.random.randn(num_pts) * 5,
        'err4': -1000 + np.abs(np.random.randn(num_pts)),
        'err5': [0.3, 0.5, 0.7, 0.4, 0.6] * int(num_pts/5),
        'err6': np.abs(np.random.randn(num_pts) * 100),
        'err7': -10000 + np.random.randn(num_pts) * 100,
        'err8': 20000 + np.random.randn(num_pts),
        'err9': -450 + np.random.randn(num_pts),
        'par1': np.random.randint(0, 3, num_pts),
        'par2': np.random.randint(0, 7, num_pts),
        'par3': np.random.randint(8, 10, num_pts),
        'th': np.tile(np.arange(0.4, 0.9, 0.1), int(num_pts/5)),
        'th2': np.tile(np.arange(0.05, 0.45, 0.1), int(num_pts/4)),
        'useless': [1, 1, 2, 3] * int(num_pts/4),
        'filter1': [0] * 10 + [1] * int(num_pts - 10),
        'filter2': [0] * 10 + [0] * int(num_pts - 10),
        'filter3': [float(0)] * 10 + [0.1] * int(num_pts - 10),
        'filter4': [0] * 10 + [1] * int(num_pts - 10),
        'filter5': [0] * 10 + [1] * int(num_pts - 10),
        'filter6': [0] * 10 + [0] * int(num_pts - 10),
        'filter7': [float(0)] * 10 + [0.1] * int(num_pts - 10),
        'filter8': [0] * 10 + [0] * int(num_pts - 10),
        'filter9': [0] * 10 + [0] * int(num_pts - 10)}
eval_columns = ['err1', 'err2', 'err3', 'err4',
                'err5', 'err6', 'err7', 'err8', 'err9']
it_parameters = ['par1', 'par2', 'par3']
xy_axis_columns = ['th', 'th2']
data = pd.DataFrame(data)

data = data.loc[~((data['filter1'] == 0) &
                  (data['filter2'] == 0) &
                  (data['filter3'] == 0) &
                  (data['filter4'] == 0) &
                  (data['filter5'] == 0) &
                  (data['filter6'] == 0) &
                  (data['filter7'] == 0) &
                  (data['filter8'] == 0) &
                  (data['filter9'] == 0))]
needed_columns = eval_columns + it_parameters + xy_axis_columns
df = data[needed_columns]
stats = df.groupby(it_parameters + xy_axis_columns).describe()
errvalnames = stats.columns.values
grp_names = stats.index.names
for it in errvalnames:
    if it[-1] != 'count':
        tmp = stats[it[0]].unstack()
        tmp = tmp[it[1]]
        tmp = tmp.unstack()
        tmp = tmp.T
        tmp.columns = ['-'.join(map(str, a)) for a in tmp.columns]
        tmp.columns.name = '-'.join(grp_names[0:-1])
        tmp = tmp.reset_index()
        dataf_name = 'data_' + '_'.join(map(str, it)) + '_vs_' + \
                     str(grp_names[-1]) + '.csv'
        dataf_name = dataf_name.replace('%', 'perc')
        fdataf_name = os.path.join(tdata_folder, dataf_name)
        with open(fdataf_name, 'a') as f:
            f.write('# ' + str(it[-1]) + ' values for ' + str(it[0]) + '\n')
            f.write('# Column parameters: ' + '-'.join(grp_names[0:-1]) + '\n')
            tmp.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')