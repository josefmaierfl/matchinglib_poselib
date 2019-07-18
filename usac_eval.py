"""
Calculates the best performin parameter values with given testing data as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np, math
# import modin.pandas as pd
import pandas as pd
#from jinja2 import Template as ji
import jinja2 as ji
# import tempfile
# import shutil
from copy import deepcopy
import shutil
# import time

def combineRt(data):
    #Get R and t mean and standard deviation values
    stat_R = data['R_diffAll'].unstack()
    stat_t = data['t_angDiff_deg'].unstack()
    stat_R_mean = stat_R['mean']
    stat_t_mean = stat_t['mean']
    stat_R_std = stat_R['std']
    stat_t_std = stat_t['std']
    comb_stat_r = stat_R_mean + 2 * stat_R_std
    comb_stat_t = stat_t_mean + 2 * stat_t_std
    ma = comb_stat_r.select_dtypes(include=[np.number]).dropna().values.max()
    mi = comb_stat_r.select_dtypes(include=[np.number]).dropna().values.min()
    r_r = ma - mi
    ma = comb_stat_t.select_dtypes(include=[np.number]).dropna().values.max()
    mi = comb_stat_t.select_dtypes(include=[np.number]).dropna().values.min()
    r_t = ma - mi
    comb_stat_r = comb_stat_r / r_r
    comb_stat_t = comb_stat_t / r_t
    b = comb_stat_r + comb_stat_t
    return b


def get_best_comb_and_th_1(data):
    grp_names = data.index.names
    b = combineRt(data)
    b = b.T
    b.columns = ['-'.join(map(str, a)) for a in b.columns]
    b.columns.name = '-'.join(grp_names[0:-1])
    b_best_idx = b.idxmin(axis=0)
    b_best_l = [[val, b.loc[val].iloc[i], b.columns[i]] for i, val in enumerate(b_best_idx)]
    b_best = pd.DataFrame.from_records(data=b_best_l, columns=['th', 'val', 'alg'])
    c.set_index('alg', inplace=True)
    b_worst_idx = b.idxmax(axis=0)
