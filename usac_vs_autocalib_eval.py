"""
Compares USAC with correspondence aggregation and the autocalibration with given testing data as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np, math
# import modin.pandas as pd
import pandas as pd
#from jinja2 import Template as ji
import jinja2 as ji
import ruamel.yaml as yaml
from usac_eval import ji_env, get_time_fixed_kp, insert_opt_lbreak, prepare_io
from statistics_and_plot import compile_tex
from copy import deepcopy

def get_accum_corrs_sequs(**keywords):
    if 'partitions' in keywords:
        if 'x_axis_column' in keywords:
            individual_grps = keywords['it_parameters'] + keywords['partitions'] + keywords['x_axis_column']
        elif 'xy_axis_columns' in keywords:
            individual_grps = keywords['it_parameters'] + keywords['partitions'] + keywords['xy_axis_columns']
        else:
            raise ValueError('Either x_axis_column or xy_axis_columns must be provided')
    elif 'x_axis_column' in keywords:
        individual_grps = keywords['it_parameters'] + keywords['x_axis_column']
    elif 'xy_axis_columns' in keywords:
        individual_grps = keywords['it_parameters'] + keywords['xy_axis_columns']
    elif 'it_parameters' in keywords:
        individual_grps = keywords['it_parameters']
    else:
        raise ValueError('Either x_axis_column or xy_axis_columns and it_parameters must be provided')
    if 'data_seperators' not in keywords:
        raise ValueError('data_seperators missing!')
    needed_cols = list(dict.fromkeys(individual_grps + keywords['data_seperators'] +
                                     ['Nr', 'stereoRef', 'accumCorrs'] + keywords['eval_columns']))
    # Split in sequences
    df_grp = keywords['data'].loc[:, needed_cols].groupby(keywords['data_seperators'])
    grp_keys = df_grp.groups.keys()
    df_list = []
    for grp in grp_keys:
        tmp = df_grp.get_group(grp)
        tmp1 = tmp.set_index('Nr', append=True)
        tmp_iter = tmp1.iterrows()
        idx_old, _ = next(tmp_iter)
        idx_first = idx_old
        idx_list = []
        for idx, _ in tmp_iter:
            if idx[1] < idx_old[1]:
                idx_list.append((idx_first[0], idx_old[0]))
                idx_first = idx
            idx_old = idx
        idx_list.append((idx_first[0], tmp.index.values[-1]))
        df_list2 = []
        for idx in idx_list:
            tmp1 = tmp.loc[idx[0]:idx[1], :]
            row_cnt = tmp1.shape[0]
            accumCorrs_max = tmp1['accumCorrs'].max()
            tmp1['accumCorrs_max'] = [accumCorrs_max] * row_cnt
            df_list2.append(tmp1)
        df_list.append(pd.concat(df_list2, ignore_index=False))
    tmp = pd.concat(df_list, ignore_index=False)
    keywords['it_parameters'] = list(dict.fromkeys(keywords['it_parameters'] + ['accumCorrs_max']))
    keywords['data'] = tmp
    return keywords

