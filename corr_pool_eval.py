"""
Calculates the best performing parameter values with given testing data as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np, math
# import modin.pandas as pd
import pandas as pd
#from jinja2 import Template as ji
import jinja2 as ji
import ruamel.yaml as yaml
from usac_eval import ji_env, get_time_fixed_kp, insert_opt_lbreak, prepare_io

def filter_take_end_frames(**vars):
    return vars['data'].loc[vars['data']['Nr'] > 119]


def filter_max_pool_size(**vars):
    return vars['data'].loc[vars['data']['stereoParameters_maxPoolCorrespondences'] == 40000]


def calc_rt_diff_frame_to_frame(**vars):
    if 'partitions' not in vars:
        raise ValueError('Partitions are necessary.')
    for key in vars['partitions']:
        if key not in vars['data_separators']:
            raise ValueError('All partition names must be included in the data separators.')
    if 'xy_axis_columns' not in vars:
        raise ValueError('xy_axis_columns are necessary.')
    if len(vars['data_separators']) != (len(vars['partitions']) + 2):
        raise ValueError('Wrong number of data separators.')

    needed_cols = vars['eval_columns'] + vars['it_parameters'] + vars['data_separators']
    df = vars['data'][needed_cols]
    grpd_cols = vars['data_separators'] + vars['it_parameters']
    df_grp = df.groupby(grpd_cols)
    grp_keys = df_grp.groups.keys()
    eval_log = {}
    eval_cols_log_scaling = []
    for eval in vars['eval_columns']:
        eval_log[eval] = []
    for grp in grp_keys:
        tmp = df_grp.get_group(grp)
        for eval in vars['eval_columns']:
            eval_log[eval].append(True if np.abs(np.log10(np.abs(tmp[eval].min())) -
                                                 np.log10(np.abs(tmp[eval].max()))) > 1 else False)
    for eval in vars['eval_columns']:
        if any(eval_log[eval]):
            eval_cols_log_scaling.append(True)
        else:
            eval_cols_log_scaling.append(False)

    from statistics_and_plot import replaceCSVLabels
    ret = {'data': df,
           'it_parameters': vars['it_parameters'],
           'eval_columns': vars['eval_columns'],
           'eval_cols_lname': [replaceCSVLabels(a) for a in vars['eval_columns']],
           'eval_cols_log_scaling': eval_cols_log_scaling,
           'units': vars['units'],
           'eval_init_input': None,
           'xy_axis_columns': [],
           'partitions': vars['partitions']}
    for key in vars['data_separators']:
        if key not in vars['partitions']:
            ret['xy_axis_columns'].append(key)
    return ret


def calc_rt_diff2_frame_to_frame(**vars):
    if 'partitions' not in vars:
        raise ValueError('Partitions are necessary.')
    for key in vars['partitions']:
        if key not in vars['data_separators']:
            raise ValueError('All partition names must be included in the data separators.')
    if 'xy_axis_columns' not in vars:
        raise ValueError('xy_axis_columns are necessary.')
    if len(vars['data_separators']) != (len(vars['partitions']) + 2):
        raise ValueError('Wrong number of data separators.')

    needed_cols = vars['eval_columns'] + vars['it_parameters'] + vars['data_separators']
    df = vars['data'][needed_cols]
    grpd_cols = vars['data_separators'] + vars['it_parameters']
    df_grp = df.groupby(grpd_cols)
    grp_keys = df_grp.groups.keys()
    eval_log = {}
    eval_cols_log_scaling = []
    for eval in vars['eval_columns']:
        eval_log[eval] = []
    data_list = []
    for grp in grp_keys:
        tmp = df_grp.get_group(grp)
        tmp.set_index('Nr', inplace=True)
        row_iterator = tmp.iterrows()
        _, last = next(row_iterator)
        tmp1 = []
        for i, row in row_iterator:
            tmp1.append(row[vars['eval_columns']] - last[vars['eval_columns']])
            tmp1[-1]['Nr'] = i
            tmp1[-1].append(row[[a for a in grpd_cols if a != 'Nr']])
        data_list.append(pd.concat(tmp1, axis=1).T)

        for eval in vars['eval_columns']:
            eval_log[eval].append(True if np.abs(np.log10(np.abs(data_list[-1][eval].min())) -
                                                 np.log10(np.abs(data_list[-1][eval].max()))) > 1 else False)
    data_new = pd.concat(data_list, ignore_index=True)
    for eval in vars['eval_columns']:
        if any(eval_log[eval]):
            eval_cols_log_scaling.append(True)
        else:
            eval_cols_log_scaling.append(False)

    data_new.columns = [a + '_diff' if a in vars['eval_columns'] else a for a in data_new.columns]
    units = [(a[0] + '_diff', a[1]) for a in vars['units']]

    from statistics_and_plot import replaceCSVLabels
    ret = {'data': data_new,
           'it_parameters': vars['it_parameters'],
           'eval_columns': vars['eval_columns'],
           'eval_cols_lname': [replaceCSVLabels(a) for a in data_new.columns],
           'eval_cols_log_scaling': eval_cols_log_scaling,
           'units': units,
           'eval_init_input': None,
           'xy_axis_columns': [],
           'partitions': vars['partitions']}
    for key in vars['data_separators']:
        if key not in vars['partitions']:
            ret['xy_axis_columns'].append(key)
    return ret