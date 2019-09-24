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
    if 'keepEval' in vars:
        for i in vars['keepEval']:
            if i not in vars['eval_columns']:
                raise ValueError('Label ' + i + ' not found in \'eval_columns\'')

    needed_cols = vars['eval_columns'] + vars['it_parameters'] + vars['data_separators']
    df = vars['data'][needed_cols]
    grpd_cols = [a for a in vars['data_separators'] if a != 'Nr'] + vars['it_parameters']
    df_grp = df.groupby(grpd_cols)
    grp_keys = df_grp.groups.keys()
    data_list = []
    eval_columns_diff = [a + '_diff' for a in vars['eval_columns']]
    if 'keepEval' in vars:
        eval_columns_diff1 = eval_columns_diff + vars['keepEval']
    else:
        eval_columns_diff1 = eval_columns_diff
    eval_log = {}
    eval_cols_log_scaling = []
    for evalv in eval_columns_diff1:
        eval_log[evalv] = []
    for grp in grp_keys:
        tmp = df_grp.get_group(grp)
        tmp.set_index('Nr', inplace=True)
        row_iterator = tmp.iterrows()
        _, last = next(row_iterator)
        tmp1 = []
        for i, row in row_iterator:
            tmp1.append(row[vars['eval_columns']] - last[vars['eval_columns']])
            tmp1[-1].index = eval_columns_diff
            if 'keepEval' in vars:
                for i1 in vars['keepEval']:
                    tmp1[-1][i1] = row[i1]
            tmp1[-1]['Nr'] = i
            tmp1[-1].append(row[[a for a in grpd_cols if a != 'Nr']])
        data_list.append(pd.concat(tmp1, axis=1).T)

        for evalv in eval_columns_diff1:
            eval_log[evalv].append(True if np.abs(np.log10(np.abs(data_list[-1][evalv].min())) -
                                                  np.log10(np.abs(data_list[-1][evalv].max()))) > 1 else False)
    data_new = pd.concat(data_list, ignore_index=True)
    for evalv in eval_columns_diff1:
        if any(eval_log[evalv]):
            eval_cols_log_scaling.append(True)
        else:
            eval_cols_log_scaling.append(False)

    # data_new.columns = [a + '_diff' if a in vars['eval_columns'] else a for a in data_new.columns]
    units = [(a[0] + '_diff', a[1]) for a in vars['units']]
    if 'keepEval' in vars:
        units1 = [a for a in vars['units'] if a[0] in vars['keepEval']]
        units += units1

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


def eval_corr_pool_converge(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    if 'eval_columns' not in keywords:
        raise ValueError('Missing parameter eval_columns')
    if 'poolSize' not in keywords['eval_columns'] or \
       'poolSize_diff' not in keywords['eval_columns'] or \
       'R_diffAll_diff' not in keywords['eval_columns'] or \
       't_angDiff_deg_diff' not in keywords['eval_columns']:
        raise ValueError('Some specific entries within parameter eval_columns is missing')

    keywords = prepare_io(**keywords)
    needed_evals = ['poolSize', 'poolSize_diff', 'R_diffAll_diff', 't_angDiff_deg_diff']
    needed_cols = needed_evals + keywords['partitions'] + keywords['xy_axis_columns'] + keywords['it_parameters']
    tmp = keywords['data'].loc[:, needed_cols]
    comb_vars = ['R_diffAll_diff', 't_angDiff_deg_diff']
    tmp_mm = tmp[comb_vars]
    min_vals = tmp_mm.abs().min()
    max_vals = tmp_mm.abs().max()
    r_vals = max_vals - min_vals
    tmp3 = tmp_mm.div(r_vals, axis=1)
    tmp['Rt_diff2'] = (tmp3[comb_vars[0]] + tmp3[comb_vars[1]]) / 2

    grpd_cols = keywords['partitions'] + \
                [a for a in keywords['xy_axis_columns'] if a != 'Nr'] + \
                keywords['it_parameters']
    df_grp = tmp.groupby(grpd_cols)
    grp_keys = df_grp.groups.keys()
    for grp in grp_keys:
        tmp1 = df_grp.get_group(grp)

    grpd_cols = keywords['partitions'] + \
                [a for a in keywords['xy_axis_columns'] if a != 'Nr'] + \
                keywords['it_parameters']
    df_grp = tmp.groupby(grpd_cols)
    grp_keys = df_grp.groups.keys()
    data_list = []
    for grp in grp_keys:
        tmp1 = df_grp.get_group(grp)
        tmp2 = tmp1[comb_vars]
        tmp3 = tmp2.div(r_vals, axis=1)
        err_vals = (tmp3[comb_vars[0]] + tmp3[comb_vars[1]]) / 2
        err_vals.name = 'Rt_diff2'
        data_list.append(pd.concat([tmp1.loc[:, [a for a in needed_cols if a not in comb_vars]], err_vals], axis=1))

    data_new = pd.concat(data_list, ignore_index=True)


def get_converge_img(df, nr_parts):
    img_min = df['Nr'].min()
    img_max = df['Nr'].max()
    ir = img_max - img_min
    ir_part = round(ir / nr_parts, 0)
    parts = [[img_min + a * ir_part] for a in range(0, nr_parts - 1)]
    p
    tmp = df.loc[]