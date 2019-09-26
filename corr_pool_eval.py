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
            tmp1[-1] = tmp1[-1].append(row[[a for a in grpd_cols if a != 'Nr']])
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
    needed_evals = ['poolSize', 'poolSize_diff', 'R_diffAll_diff', 't_angDiff_deg_diff', 'R_diffAll', 't_angDiff_deg']
    if not all([a in keywords['eval_columns'] for a in needed_evals]):
        raise ValueError('Some specific entries within parameter eval_columns is missing')

    keywords = prepare_io(**keywords)
    needed_cols = needed_evals + keywords['partitions'] + keywords['xy_axis_columns'] + keywords['it_parameters']
    tmp = keywords['data'].loc[:, needed_cols]

    # comb_vars = ['R_diffAll', 't_angDiff_deg']
    # comb_diff_vars = ['R_diffAll_diff', 't_angDiff_deg_diff']
    # tmp_mm = tmp.loc[:,comb_vars]
    # row = tmp_mm.loc[tmp_mm['Nr'].idxmin()]
    # row['Nr'] -= 1
    # row[comb_vars] -= row[comb_diff_vars]
    # tmp_mm = tmp_mm.append(row)
    # min_vals = tmp_mm.abs().min()
    # max_vals = tmp_mm.abs().max()
    # r_vals = max_vals - min_vals
    # tmp2 = tmp_mm.div(r_vals, axis=1)
    # tmp['Rt_diff_single'] = (tmp2[comb_vars[0]] + tmp2[comb_vars[1]]) / 2

    tmp = combine_rt_diff2(tmp)

    grpd_cols = keywords['partitions'] + \
                [a for a in keywords['xy_axis_columns'] if a != 'Nr'] + \
                keywords['it_parameters']
    df_grp = tmp.groupby(grpd_cols)
    grp_keys = df_grp.groups.keys()
    data_list = []
    for grp in grp_keys:
        tmp1 = df_grp.get_group(grp)
        tmp2, succ = get_converge_img(tmp1, 3, 0.33, 0.02)
        data_list.append(tmp2)
    data_new = pd.concat(data_list, ignore_index=True)


def combine_rt_diff2(df):
    comb_vars = ['R_diffAll', 't_angDiff_deg']
    comb_diff_vars = ['R_diffAll_diff', 't_angDiff_deg_diff']
    tmp_mm = df[comb_diff_vars]
    tmp3 = (df[comb_vars[0]] * tmp_mm[comb_diff_vars[0]] / df[comb_vars[0]].abs() +
            df[comb_vars[1]] * tmp_mm[comb_diff_vars[1]] / df[comb_vars[1]].abs()) / 2
    min_vals = tmp3.min()
    max_vals = tmp3.max()
    r_vals = max_vals - min_vals
    df['Rt_diff2'] = (tmp3 - min_vals) / r_vals

    return df


def get_mean_data_parts(df, nr_parts):
    img_min = df['Nr'].min()
    img_max = df['Nr'].max()
    ir = img_max - img_min
    if ir <= 1:
        return {}, False
    elif ir < nr_parts:
        nr_parts = ir
    ir_part = round(ir / nr_parts, 0)
    parts = [[img_min + a * ir_part, img_min + (a + 1) * ir_part] for a in range(0, nr_parts)]
    parts[-1][1] = img_max + 1
    if parts[-1][1] - parts[-1][0] <= 0:
        parts.pop()
    data_parts = []
    mean_dd = []
    for sl in parts:
        if sl[1] - sl[0] == 1:
            tmp = df.set_index('Nr')
            data_parts.append(tmp.loc[[sl[0]],:])
            data_parts[-1].reset_index(inplace=True)
            mean_dd.append(data_parts[-1]['Rt_diff2'].values[0])
        else:
            data_parts.append(df.loc[df['Nr'] >= sl[0] & df['Nr'] < sl[1]])
            # A negative value indicates a decreasing error value and a positive number an increasing error
            mean_dd.append(data_parts[-1]['Rt_diff2'].mean())
    data = {'data_parts': data_parts, 'mean_dd': mean_dd}
    return data, True

def get_converge_img(df, nr_parts, th_diff2=0.33, th_diff3=0.02):
    data, succ = get_mean_data_parts(df, nr_parts)
    if not succ:
        return df, False
    data_parts = data['data_parts']
    # A negative value indicates a decreasing error value and a positive number an increasing error
    mean_dd = data['mean_dd']

    error_gets_smaller = [False] * nr_parts
    nr_parts1 = nr_parts
    while not any(error_gets_smaller) and nr_parts1 < 10:
        for i, val in enumerate(mean_dd):
            if val < 0:
                error_gets_smaller[i] = True
        if not any(error_gets_smaller) and nr_parts1 < 10:
            nr_parts1 += 1
            data, succ = get_mean_data_parts(df, nr_parts1)
            if not succ:
                return df, False
            data_parts = data['data_parts']
            mean_dd = data['mean_dd']
        else:
            break
    if not any(error_gets_smaller):
        df1, succ = get_converge_img(data_parts[0], nr_parts, th_diff2, th_diff3)
        if not succ:
            return df1, False
        else:
            data, succ = get_mean_data_parts(df1, nr_parts)
            if not succ:
                return df1, False
            data_parts = data['data_parts']
            mean_dd = data['mean_dd']
            error_gets_smaller = [False] * nr_parts
            for i, val in enumerate(mean_dd):
                if val < 0:
                    error_gets_smaller[i] = True
            if not any(error_gets_smaller):
                return data_parts[0].loc[[data_parts[0]['Nr'].idxmin()], :], False

    l1 = len(mean_dd) - 1
    l2 = l1 - 1
    sel_parts = []
    last = 0
    for i, val in enumerate(mean_dd):
        if not error_gets_smaller[i]:
            last = 0
            continue
        if i < l1:
            if not error_gets_smaller[i + 1]:
                sel_parts.append(i)
                break
            diff1 = (abs(mean_dd[i + 1]) - abs(val)) / abs(val)
            if diff1 > 0:
                last = i + 2
            else:
                last = 0
                if i < l2:
                    if not error_gets_smaller[i + 2]:
                        sel_parts.append(i + 1)
                        break
                    diff2 = (abs(mean_dd[i + 2]) - abs(mean_dd[i + 1])) / abs(mean_dd[i + 1])
                    if abs(diff2) < th_diff3 and mean_dd[i + 1] < th_diff2 * mean_dd[0]:
                        sel_parts.append(i + 1)
                        break
                    else:
                        last = i + 3
                else:
                    sel_parts.append(i + 1)
                    break
        else:
            sel_parts.append(i)
    if sel_parts:
        return get_converge_img(data_parts[sel_parts[0]], nr_parts, th_diff2, th_diff3)
    elif last != 0:
        return get_converge_img(data_parts[min(last, l1)], nr_parts, th_diff2, th_diff3)
    else:
        return get_converge_img(data_parts[0], nr_parts, th_diff2, th_diff3)

