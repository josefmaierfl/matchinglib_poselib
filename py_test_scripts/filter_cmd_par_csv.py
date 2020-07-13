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
import numpy as np
import pandas as pd, os, warnings, sys
import ruamel.yaml as yaml


warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


def readYaml(file, isstr = False):
    if isstr:
        data = file.split('\n')
    else:
        with open(file, 'r') as fi:
            data = fi.readlines()
    data = [line for line in data if line and line[0] is not '%']
    try:
        data = yaml.load_all("\n".join(data))
        data1 = {}
        for d in data:
            data1.update(d)
        data = data1
    except:
        print('Exception during reading yaml.')
        e = sys.exc_info()
        print(str(e))
        sys.stdout.flush()
        raise BaseException
    return data


def find_line(filename, inlcrate, kpacc, cmd, sub_path, last_parset, path, it_nr, sequDir, depth_distr, nr_missing, existing_f=None):
    cf = pd.read_csv(filename, delimiter=';')
    if cf.empty:
        raise ValueError("File " + filename + " is empty.")
    cf.rename(columns={ cf.columns[0]: "nr" }, inplace=True)
    cmdl = cf.loc[:, 'cmd'][0]
    cmdl1 = cmdl.split('--')[1:]
    c_use = []
    c_use1 = []
    for c in cmdl1:
        tmp0 = c.strip().split(' ')[0]
        tmp = '--' + tmp0
        if tmp in cmd:
            c_use.append(tmp0)
            c_use1.append(tmp)

    vals = []
    cmdl1 = cmd.split('--')[1:]
    for val in cmdl1:
        tmp = val.strip().split(' ')
        vals.append([tmp[0], float(tmp[1])])

    ss = []
    for a, b in zip(c_use1, c_use):
        ss.append(a + ' (?P<' + b + '>[\\d\\.]+)')

    dfs = [cf]
    for reg in ss:
        dfs.append(cf['cmd'].str.extract(reg).astype(float))
    df = pd.concat(dfs, axis=1, sort=False)
    df1 = df.loc[(df['inlrat_c_rate'] > inlcrate - 0.01) & (df['inlrat_c_rate'] < inlcrate + 0.01)]
    df1 = df1.loc[(df1['sub_path'] > sub_path - 0.01) & (df1['sub_path'] < sub_path + 0.01)]
    # df = df.loc[(df['inlrat_min'] > inlcrate - 0.01) & (df['inlrat_min'] < inlcrate + 0.01)]
    df1 = df1.loc[(df1['kp_acc_sd'] > kpacc - 0.01) & (df1['kp_acc_sd'] < kpacc + 0.01)]
    df1 = df1.loc[(df1['depth_distr'].str.contains(depth_distr))]
    df1 = df1.loc[(df1['sequDir'].str.contains(sequDir))]
    for val in vals:
        df1 = df1.loc[(df1[val[0]] > val[1] - 0.01) & (df1[val[0]] < val[1] + 0.01)]
    if df1.shape[0] > 1 or df1.empty:
        print('Too many or no solutions')
        return
    nr = int(df1['nr'])
    surr = 72
    path_csv = os.path.dirname(path)
    base_yaml = 'parSetNr_'
    ext = '.yaml'

    dfc = df.loc[(df['nr'] < nr + 72) & (df['nr'] > nr - 72)]
    fl = [False] * dfc.shape[0]
    used_idx = []
    for i, (idx, val) in enumerate(dfc.iterrows()):
        if int(val['nr']) == nr or not np.isclose(val['sub_path'], sub_path):
            fl[i] = True
            continue
        idx_m = max(last_parset - 71, 0)
        for idx_ps in range(idx_m, last_parset - 1):
            if idx_ps in used_idx:
                continue
            ovf = os.path.join(path, base_yaml + str(idx_ps) + ext)
            y_data = readYaml(ovf)
            csvf = os.path.join(path_csv, y_data['hashTestingPars'])
            c_data = pd.read_csv(csvf, delimiter=';')
            asi = c_data.loc[:, 'addSequInfo'][0].split('_')
            cmp_a = [val['inlrat_c_rate'], val['kp_acc_sd']]
            cmp_a += [val[a[0]] for a in vals]
            cmp_b = [float(asi[asi.index('inlratCRate') + 1]), float(asi[asi.index('kpAccSd') + 1])]
            cmp_b += [y_data['stereoParameters'][a[0]] for a in vals]
            cmp = np.isclose(cmp_a, cmp_b)
            if not all(cmp):
                continue
            p_tmp, sequNr = os.path.split(val['sequDir'])
            sequ_name = os.path.basename(p_tmp)
            p_tmp, sequNr1 = os.path.split(y_data['sequ_path'])
            sequ_name1 = os.path.basename(p_tmp)
            dep = val['depth_distr']
            dep1 = asi[asi.index('depthDistr') + 1]
            if sequNr == sequNr1 and sequ_name == sequ_name1 and dep == dep1:
                used_idx.append(idx_ps)
                fl[i] = True
                break

    dfc['cmp'] = fl
    df = dfc.loc[dfc['cmp'] == False].loc[:, cf.columns]
    df1 = cf.loc[cf['nr'] >= nr + 72]
    if existing_f:
        df = pd.concat([pd.read_csv(existing_f, delimiter=';'), df, df1], axis=0, sort=False)
    df = pd.concat([df, df1], axis=0, sort=False)
    f_new = os.path.join(os.path.dirname(filename), 'commands_and_parameters_unsuccessful_' + str(it_nr) + '.csv')
    if os.path.exists(f_new):
        raise ValueError('File ' + f_new + ' already exists')
    df.to_csv(index=False, sep=';', path_or_buf=f_new)


# file = '/home/maierj/work/results/results_001/results/testing_results/correspondence_pool/1/commands_and_parameters_full.csv'
file = '/home/maierj/work/results/results_001/results/testing_results/robustness/1/commands_and_parameters_full.csv'
inlratCRate = 0.55
kpAccSd = 1.75
cmd1 = '--relInlRatThLast 0.5 --relInlRatThNew 0.55 --minInlierRatSkip 0.55 --relMinInlierRatSkip 0.8 --minInlierRatioReInit 0.8'
sequ_dir = 'robustness_kp-distr-half-img_depth-NMF_TP-30to500_extr-F1-Rv-tf/15953827762273448583'
depth_distr = 'NMF'
sub_path = 97
last_parset = 9199
path = '/home/maierj/work/results/results_001/results/testing_results/robustness/1/results/97/pars'
it_nr = 1
nr_missing = 1
existing_f = '/home/maierj/work/results/results_001/results/testing_results/robustness/1/commands_and_parameters_unsuccessful_0.csv'

# cmd0 = '--minPtsDistance 7.5 --maxPoolCorrespondences 8000'
# inlrat = 0.3

find_line(file, inlratCRate, kpAccSd, cmd1, sub_path, last_parset, path, it_nr, sequ_dir, depth_distr)
