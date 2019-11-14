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
from statistics_and_plot import compile_tex
from copy import deepcopy

def get_rt_change_type(**keywords):
    if 'data_seperators' not in keywords:
        raise ValueError('data_seperators missing!')

    df_grp = keywords['data'].groupby(keywords['data_seperators'])
    grp_keys = df_grp.groups.keys()
    df_list = []
    change_j_occ = {'jrt': 0, 'jra': 0, 'jta': 0, 'jrx': 0, 'jry': 0, 'jrz': 0, 'jtx': 0, 'jty': 0, 'jtz': 0}
    change_pos = []
    for grp in grp_keys:
        tmp = df_grp.get_group(grp)
        nr_min = tmp['Nr'].min()
        nr_max = tmp['Nr'].max()
        rng1 = nr_max - nr_min + 1
        tmp1 = tmp['Nr'].loc[(tmp['Nr'] == nr_min)]
        tmp1_it = tmp1.iteritems()
        idx_prev, _ = next(tmp1_it)
        indexes = {'first': [], 'last': []}
        idx_prev0 = int(tmp.index[0])
        if idx_prev0 != idx_prev:
            indexes['first'].append(idx_prev0)
            tmp2 = tmp['Nr'].iloc[((tmp.index > idx_prev0) & (tmp.index < idx_prev))]
            tmp2_it = tmp2.iteritems()
            idx1_prev, nr_prev = next(tmp2_it)
            for idx1, nr in tmp2_it:
                if nr_prev > nr:
                    indexes['last'].append(idx1_prev + 1)
                    indexes['first'].append(idx1)
                nr_prev = nr
                idx1_prev = idx1
            indexes['last'].append(idx_prev)
        for idx, _ in tmp1_it:
            diff = idx - idx_prev
            indexes['first'].append(idx_prev)
            if diff != rng1:
                tmp2 = tmp['Nr'].iloc[((tmp.index > idx_prev) & (tmp.index < idx))]
                tmp2_it = tmp2.iteritems()
                idx1_prev, nr_prev = next(tmp2_it)
                for idx1, nr in tmp2_it:
                    if nr_prev > nr:
                        indexes['last'].append(idx1_prev + 1)
                        indexes['first'].append(idx1)
                    nr_prev = nr
                    idx1_prev = idx1
            indexes['last'].append(idx)
            idx_prev = idx
        indexes['first'].append(int(tmp1.index[-1]))
        diff = tmp.index[-1] - tmp1.index[-1] + 1
        if diff != rng1:
            tmp2 = tmp['Nr'].iloc[(tmp.index > tmp1.index[-1])]
            tmp2_it = tmp2.iteritems()
            idx1_prev, nr_prev = next(tmp2_it)
            for idx1, nr in tmp2_it:
                if nr_prev > nr:
                    indexes['last'].append(idx1_prev + 1)
                    indexes['first'].append(idx1)
                nr_prev = nr
                idx1_prev = idx1
        indexes['last'].append(int(tmp.index[-1]) + 1)

        for first, last in zip(indexes['first'], indexes['last']):
            tmp1 = tmp.iloc[((tmp.index >= first) & (tmp.index < last))].copy(deep=True)
            hlp = (tmp1['R_GT_n_diffAll'] + tmp1['t_GT_n_angDiff']).fillna(0).round(decimals=6)
            cnt = float(np.count_nonzero(hlp.to_numpy()))
            frac = cnt / float(tmp1.shape[0])
            if frac > 0.5:
                rxc = tmp1['R_GT_n_diff_roll_deg'].fillna(0).abs().sum() > 1e-3
                ryc = tmp1['R_GT_n_diff_pitch_deg'].fillna(0).abs().sum() > 1e-3
                rzc = tmp1['R_GT_n_diff_yaw_deg'].fillna(0).abs().sum() > 1e-3
                txc = tmp1['t_GT_n_elemDiff_tx'].fillna(0).abs().sum() > 1e-4
                tyc = tmp1['t_GT_n_elemDiff_ty'].fillna(0).abs().sum() > 1e-4
                tzc = tmp1['t_GT_n_elemDiff_tz'].fillna(0).abs().sum() > 1e-4
                if rxc and ryc and rzc and txc and tyc and tzc:
                    tmp1['rt_change_type'] = ['crt'] * int(tmp1.shape[0])
                elif rxc and ryc and rzc:
                    tmp1['rt_change_type'] = ['cra'] * int(tmp1.shape[0])
                elif txc and tyc and tzc:
                    tmp1['rt_change_type'] = ['cta'] * int(tmp1.shape[0])
                elif rxc:
                    tmp1['rt_change_type'] = ['crx'] * int(tmp1.shape[0])
                elif ryc:
                    tmp1['rt_change_type'] = ['cry'] * int(tmp1.shape[0])
                elif rzc:
                    tmp1['rt_change_type'] = ['crz'] * int(tmp1.shape[0])
                elif txc:
                    tmp1['rt_change_type'] = ['ctx'] * int(tmp1.shape[0])
                elif tyc:
                    tmp1['rt_change_type'] = ['cty'] * int(tmp1.shape[0])
                elif tzc:
                    tmp1['rt_change_type'] = ['ctz'] * int(tmp1.shape[0])
                else:
                    tmp1['rt_change_type'] = ['nv'] * int(tmp1.shape[0])# no variation
                tmp1['rt_change_pos'] = [0] * int(tmp1.shape[0])
            else:
                rxc = tmp1['R_GT_n_diff_roll_deg'].fillna(0).abs().sum() > 1e-3
                ryc = tmp1['R_GT_n_diff_pitch_deg'].fillna(0).abs().sum() > 1e-3
                rzc = tmp1['R_GT_n_diff_yaw_deg'].fillna(0).abs().sum() > 1e-3
                txc = tmp1['t_GT_n_elemDiff_tx'].fillna(0).abs().sum() > 1e-4
                tyc = tmp1['t_GT_n_elemDiff_ty'].fillna(0).abs().sum() > 1e-4
                tzc = tmp1['t_GT_n_elemDiff_tz'].fillna(0).abs().sum() > 1e-4
                change_positions = []
                if rxc and ryc and rzc and txc and tyc and tzc:
                    tmp1['rt_change_type'] = ['jrt'] * int(tmp1.shape[0])
                    change_j_occ['jrt'] += 1
                    change_positions.append(
                        int(tmp1.loc[tmp1['R_GT_n_diff_roll_deg'].fillna(0).abs() > 1e-3, 'Nr'].iloc[0]))
                    change_positions.append(
                        int(tmp1.loc[tmp1['R_GT_n_diff_pitch_deg'].fillna(0).abs() > 1e-3, 'Nr'].iloc[0]))
                    change_positions.append(
                        int(tmp1.loc[tmp1['R_GT_n_diff_yaw_deg'].fillna(0).abs() > 1e-3, 'Nr'].iloc[0]))
                    change_positions.append(
                        int(tmp1.loc[tmp1['t_GT_n_elemDiff_tx'].fillna(0).abs() > 1e-4, 'Nr'].iloc[0]))
                    change_positions.append(
                        int(tmp1.loc[tmp1['t_GT_n_elemDiff_ty'].fillna(0).abs() > 1e-4, 'Nr'].iloc[0]))
                    change_positions.append(
                        int(tmp1.loc[tmp1['t_GT_n_elemDiff_tz'].fillna(0).abs() > 1e-4, 'Nr'].iloc[0]))
                    change_pos.append(change_positions[-1])
                elif rxc and ryc and rzc:
                    tmp1['rt_change_type'] = ['jra'] * int(tmp1.shape[0])
                    change_j_occ['jra'] += 1
                    change_positions.append(
                        int(tmp1.loc[tmp1['R_GT_n_diff_roll_deg'].fillna(0).abs() > 1e-3, 'Nr'].iloc[0]))
                    change_positions.append(
                        int(tmp1.loc[tmp1['R_GT_n_diff_pitch_deg'].fillna(0).abs() > 1e-3, 'Nr'].iloc[0]))
                    change_positions.append(
                        int(tmp1.loc[tmp1['R_GT_n_diff_yaw_deg'].fillna(0).abs() > 1e-3, 'Nr'].iloc[0]))
                    change_pos.append(change_positions[-1])
                elif txc and tyc and tzc:
                    tmp1['rt_change_type'] = ['jta'] * int(tmp1.shape[0])
                    change_j_occ['jta'] += 1
                    change_positions.append(
                        int(tmp1.loc[tmp1['t_GT_n_elemDiff_tx'].fillna(0).abs() > 1e-4, 'Nr'].iloc[0]))
                    change_positions.append(
                        int(tmp1.loc[tmp1['t_GT_n_elemDiff_ty'].fillna(0).abs() > 1e-4, 'Nr'].iloc[0]))
                    change_positions.append(
                        int(tmp1.loc[tmp1['t_GT_n_elemDiff_tz'].fillna(0).abs() > 1e-4, 'Nr'].iloc[0]))
                    change_pos.append(change_positions[-1])
                elif rxc:
                    tmp1['rt_change_type'] = ['jrx'] * int(tmp1.shape[0])
                    change_j_occ['jrx'] += 1
                    change_positions.append(
                        int(tmp1.loc[tmp1['R_GT_n_diff_roll_deg'].fillna(0).abs() > 1e-3, 'Nr'].iloc[0]))
                    change_pos.append(change_positions[-1])
                elif ryc:
                    tmp1['rt_change_type'] = ['jry'] * int(tmp1.shape[0])
                    change_j_occ['jry'] += 1
                    change_positions.append(
                        int(tmp1.loc[tmp1['R_GT_n_diff_pitch_deg'].fillna(0).abs() > 1e-3, 'Nr'].iloc[0]))
                    change_pos.append(change_positions[-1])
                elif rzc:
                    tmp1['rt_change_type'] = ['jrz'] * int(tmp1.shape[0])
                    change_j_occ['jrz'] += 1
                    change_positions.append(
                        int(tmp1.loc[tmp1['R_GT_n_diff_yaw_deg'].fillna(0).abs() > 1e-3, 'Nr'].iloc[0]))
                    change_pos.append(change_positions[-1])
                elif txc:
                    tmp1['rt_change_type'] = ['jtx'] * int(tmp1.shape[0])
                    change_j_occ['jtx'] += 1
                    change_positions.append(
                        int(tmp1.loc[tmp1['t_GT_n_elemDiff_tx'].fillna(0).abs() > 1e-4, 'Nr'].iloc[0]))
                    change_pos.append(change_positions[-1])
                elif tyc:
                    tmp1['rt_change_type'] = ['jty'] * int(tmp1.shape[0])
                    change_j_occ['jty'] += 1
                    change_positions.append(
                        int(tmp1.loc[tmp1['t_GT_n_elemDiff_ty'].fillna(0).abs() > 1e-4, 'Nr'].iloc[0]))
                    change_pos.append(change_positions[-1])
                elif tzc:
                    tmp1['rt_change_type'] = ['jtz'] * int(tmp1.shape[0])
                    change_j_occ['jtz'] += 1
                    change_positions.append(
                        int(tmp1.loc[tmp1['t_GT_n_elemDiff_tz'].fillna(0).abs() > 1e-4, 'Nr'].iloc[0]))
                    change_pos.append(change_positions[-1])
                else:
                    change_j_occ1 = deepcopy(change_j_occ)
                    min_key = 'nv'
                    del_list = []
                    for k, v in change_j_occ1.items():
                        if v == 0:
                            del_list.append(k)
                    for k in del_list:
                        del change_j_occ1[k]
                    if change_j_occ1 and len(change_j_occ1.keys()) > 1:
                        max_val = change_j_occ1[max(change_j_occ1, key=(lambda key: change_j_occ1[key]))]
                        min_val = change_j_occ1[min(change_j_occ1, key=(lambda key: change_j_occ1[key]))]
                        if max_val - min_val == 1:
                            change_j_occ1_sort = sorted(change_j_occ1.items(), key=lambda kv: kv[1])
                            if change_j_occ1_sort[0][1] != change_j_occ1_sort[1][1]:
                                min_key = change_j_occ1_sort[0][0]
                                change_j_occ[min_key] += 1
                    # max_val = change_j_occ[max(change_j_occ, key=(lambda key: change_j_occ[key]))]
                    # min_val = max_val
                    # min_key = 'nv'
                    # for key, value in change_j_occ.items():
                    #     if value > 0 and value < min_val:
                    #         min_key = key
                    #     elif value > 0 and value == min_val:
                    #         min_key = 'nv'
                    tmp1['rt_change_type'] = [min_key] * int(tmp1.shape[0])# no variation
                    if len(change_pos) == 0:
                        change_positions.append(0)
                    elif len(change_pos) < 3:
                        change_positions.append(change_pos[0])
                    else:
                        change_pos.sort()
                        cl = len(change_pos)
                        if cl % 2 == 0:
                            change_positions.append(change_pos[int(round(cl / 2))])
                        else:
                            change_positions.append(change_pos[int(round((cl - 1) / 2))])
                if len(change_positions) < 3:
                    tmp1['rt_change_pos'] = [change_positions[0]] * int(tmp1.shape[0])
                else:
                    change_positions.sort()
                    cl = len(change_positions)
                    if cl % 2 == 0:
                        tmp1['rt_change_pos'] = [change_positions[int(round(cl / 2))]] * int(tmp1.shape[0])
                    else:
                        tmp1['rt_change_pos'] = [change_positions[int(round((cl - 1) / 2))]] * int(tmp1.shape[0])
            df_list.append(tmp1)
    df_new = pd.concat(df_list, axis=0, ignore_index=False)
    if 'filter_scene' in keywords:
        if isinstance(keywords['filter_scene'], str):
            df_new = df_new.loc[df_new['rt_change_type'].str.contains(keywords['filter_scene'], regex=False)]
        else:
            if len(keywords['filter_scene']) > 1:
                df_new = df_new.loc[df_new['rt_change_type'].str.contains('|'.join(keywords['filter_scene']),
                                                                          regex=True)]
            else:
                df_new = df_new.loc[df_new['rt_change_type'].str.contains(keywords['filter_scene'][0], regex=False)]
    if 'check_mostLikely' in keywords and keywords['check_mostLikely']:
        ml_evals = [a for a in keywords['eval_columns'] if 'mostLikely' in a]
        df_new.loc[((df_new['R_mostLikely(0,0)'] == 0) &
                   (df_new['R_mostLikely(0,1)'] == 0) &
                   (df_new['R_mostLikely(0,2)'] == 0) &
                   (df_new['R_mostLikely(1,0)'] == 0) &
                   (df_new['R_mostLikely(1,1)'] == 0) &
                   (df_new['R_mostLikely(1,2)'] == 0) &
                   (df_new['R_mostLikely(2,0)'] == 0) &
                   (df_new['R_mostLikely(2,1)'] == 0) &
                   (df_new['R_mostLikely(2,2)'] == 0)), ml_evals] = np.NaN
    elif 'filter_mostLikely' in keywords and keywords['filter_mostLikely']:
        df_new = df_new.loc[~((df_new['R_mostLikely(0,0)'] == 0) &
                              (df_new['R_mostLikely(0,1)'] == 0) &
                              (df_new['R_mostLikely(0,2)'] == 0) &
                              (df_new['R_mostLikely(1,0)'] == 0) &
                              (df_new['R_mostLikely(1,1)'] == 0) &
                              (df_new['R_mostLikely(1,2)'] == 0) &
                              (df_new['R_mostLikely(2,0)'] == 0) &
                              (df_new['R_mostLikely(2,1)'] == 0) &
                              (df_new['R_mostLikely(2,2)'] == 0))]
    return df_new


def get_best_comb_scenes_ml_1(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    eval_columns_init = deepcopy(keywords['eval_columns'])
    eval_cols1 = [a for a in eval_columns_init if 'mostLikely' not in a]
    eval_cols2 = [a for a in eval_columns_init if 'mostLikely' in a]
    data1 = keywords['data'].drop(eval_cols2, axis=1)
    data2 = keywords['data'].drop(eval_cols1, axis=1)
    res = 0
    if eval_cols1:
        keywords['eval_columns'] = eval_cols1
        keywords['data'] = data1
        res += get_best_comb_scenes_1(**keywords)

    if eval_cols2:
        keywords['eval_columns'] = eval_cols2
        keywords['data'] = data2
        keywords['is_mostLikely'] = True
        res_folder_parent = os.path.abspath(os.path.join(keywords['res_folder'], os.pardir))  # Get parent directory
        last_folder = os.path.basename(os.path.normpath(keywords['res_folder']))
        ml_folder = os.path.join(res_folder_parent, last_folder + '_ml')
        cnt = 1
        calc_vals = True
        ml_folder_init = ml_folder
        while os.path.exists(ml_folder):
            ml_folder = ml_folder_init + '_' + str(int(cnt))
            cnt += 1
        try:
            os.mkdir(ml_folder)
        except FileExistsError:
            print('Folder', ml_folder, 'for storing statistics data already exists')
        except:
            print("Unexpected error (Unable to create directory for storing special function data):", sys.exc_info()[0])
            calc_vals = False
        if calc_vals:
            keywords['res_folder'] = ml_folder
            keywords['res_par_name'] += '_ml'
            res += get_best_comb_scenes_1(**keywords)
    return res


def get_best_comb_scenes_1(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    from usac_eval import pars_calc_single_fig_partitions
    from statistics_and_plot import tex_string_coding_style, \
        short_concat_str, \
        replaceCSVLabels, \
        insert_str_option_values, \
        get_limits_log_exp, \
        split_large_titles, \
        enl_space_title, \
        strToLower, \
        check_if_neg_values
    ret = pars_calc_single_fig_partitions(**keywords)
    b_min = ret['b'].stack().reset_index()
    b_min.rename(columns={b_min.columns[-1]: 'Rt_diff'}, inplace=True)
    b_min = b_min.loc[b_min.groupby(ret['partitions'] + keywords['x_axis_column'])['Rt_diff'].idxmin()]
    b_min1 = b_min.set_index(ret['it_parameters'])
    if len(ret['it_parameters']) > 1:
        b_min1.index = ['-'.join(map(str, a)) for a in b_min1.index]
        it_pars_name = '-'.join(map(str, ret['it_parameters']))
        b_min1.index.name = it_pars_name
    else:
        it_pars_name = ret['it_parameters'][0]
    b_min_grp = b_min1.reset_index().set_index(keywords['x_axis_column']).groupby(ret['partitions'])
    grp_keys = b_min_grp.groups.keys()
    if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
        base_name = 'min_ml_'
    else:
        base_name = 'min_'
    base_name += 'RTerrors_vs_' + keywords['x_axis_column'][0] + '_and_corresp_opts_' + \
                 short_concat_str(ret['it_parameters']) + '_for_part_' + ret['dataf_name_partition']
    title = 'Smallest Combined R \\& t Errors '
    if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
        title += 'of Most Likely Extrinsics '
    title += 'and Their Corresponding ' + \
            'Parameters ' + ret['sub_title_it_pars'] + \
            ' vs ' + replaceCSVLabels(keywords['x_axis_column'][0], True, True, True) + \
            ' for Different ' + ret['sub_title_partitions']
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': ret['gloss']
                 }
    for grp in grp_keys:
        if grp == 'nv':
            continue
        tmp = b_min_grp.get_group(grp).drop(ret['partitions'], axis=1)
        if len(ret['it_parameters']) > 1:
            tmp['options_tex'] = [', '.join(['{:.3f}'.format(float(b)) for b in a.split('-')])
                                  for a in tmp[it_pars_name].values]
        else:
            tmp['options_tex'] = ['{:.3f}'.format(float(a)) for a in tmp[it_pars_name].values]
        b_mean_name = 'data_' + base_name + \
                      (str(grp) if len(ret['partitions']) == 1 else '-'.join(map(str, grp))) + '.csv'
        fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
        with open(fb_mean_name, 'a') as f:
            if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
                f.write('# Minimum combined R & t errors (Rt_diff) of most likely extrinsics '
                        'and corresponding option vs ' +
                        keywords['x_axis_column'][0] +
                        ' for partition ' + '-'.join(map(str, ret['partitions'])) + ' = ' +
                        (str(grp) if len(ret['partitions']) == 1 else '-'.join(map(str, grp))) + '\n')
            else:
                f.write('# Minimum combined R & t errors (Rt_diff) and corresponding option vs ' +
                        keywords['x_axis_column'][0] +
                        ' for partition ' + '-'.join(map(str, ret['partitions'])) + ' = ' +
                        (str(grp) if len(ret['partitions']) == 1 else '-'.join(map(str, grp))) + '\n')
            f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
            tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        sc_beginning = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ '
        if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
            section_name = sc_beginning + 'of most likely extrinsics and their ' + \
                           'corresponding options vs ' + \
                           replaceCSVLabels(keywords['x_axis_column'][0], False, False, True) + \
                           ' for property ' + insert_str_option_values(ret['partitions'], grp)
            caption = sc_beginning + 'of most likely extrinsics and their ' + \
                      'corresponding options on top of each bar separated by a comma in the order ' + \
                      strToLower(ret['sub_title_it_pars']) + ' vs ' + \
                      replaceCSVLabels(keywords['x_axis_column'][0], False, False, True) + \
                      ' for the ' + insert_str_option_values(ret['partitions'], grp)
        else:
            section_name = sc_beginning + 'and their ' + \
                           'corresponding options vs ' + \
                           replaceCSVLabels(keywords['x_axis_column'][0], False, False, True) + \
                           ' for property ' + insert_str_option_values(ret['partitions'], grp)
            caption = sc_beginning + 'and their ' + \
                      'corresponding options on top of each bar separated by a comma in the order ' + \
                      strToLower(ret['sub_title_it_pars']) + ' vs ' + \
                      replaceCSVLabels(keywords['x_axis_column'][0], False, False, True) + \
                      ' for the ' + insert_str_option_values(ret['partitions'], grp)
        _, use_limits, use_log, exp_value = get_limits_log_exp(tmp, True, True, False, ['options_tex', it_pars_name])
        is_neg = check_if_neg_values(tmp, 'Rt_diff', use_log, use_limits)
        is_numeric = pd.to_numeric(tmp.reset_index()[keywords['x_axis_column'][0]], errors='coerce').notnull().all()
        section_name = split_large_titles(section_name)
        exp_value = enl_space_title(exp_value, section_name, tmp, keywords['x_axis_column'],
                                    1, 'ybar')
        tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                      'name': section_name.replace('\\\\', ' '),
                                      'title': section_name,
                                      'title_rows': section_name.count('\\\\'),
                                      'fig_type': 'ybar',
                                      'plots': ['Rt_diff'],
                                      'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                      # Label/column name of axis with bars. For xbar it labels the y-axis
                                      'label_x': replaceCSVLabels(keywords['x_axis_column'][0]),
                                      # Column name of axis with bars. For xbar it is the column for the y-axis
                                      'print_x': keywords['x_axis_column'][0],
                                      # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                      'print_meta': True,
                                      'plot_meta': ['options_tex'],
                                      # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                      'rotate_meta': 45,
                                      'limits': None,
                                      # If None, no legend is used, otherwise use a list
                                      'legend': None,
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      # The x/y-axis values are given as strings if True
                                      'use_string_labels': True if not is_numeric else False,
                                      'use_log_y_axis': use_log,
                                      'xaxis_txt_rows': 1,
                                      'enlarge_lbl_dist': None,
                                      'enlarge_title_space': exp_value,
                                      'large_meta_space_needed': True,
                                      'is_neg': is_neg,
                                      'caption': caption
                                      })

    base_out_name = 'tex_' + base_name
    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if ret['build_pdf'][1]:
        res1 = compile_tex(rendered_tex,
                           ret['tex_folder'],
                           texf_name,
                           tex_infos['make_index'],
                           os.path.join(ret['pdf_folder'], pdf_name),
                           tex_infos['figs_externalize'])
    else:
        res1 = compile_tex(rendered_tex, ret['tex_folder'], texf_name)
    if res1 != 0:
        ret['res'] += abs(res1)
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    par_stat = b_min.drop(keywords['x_axis_column'][0], axis=1).groupby(ret['partitions']).describe()
    base_name = 'stats_min_RTerrors_and_stats_corresp_opts_' + \
                short_concat_str(ret['it_parameters']) + '_for_part_' + ret['dataf_name_partition']
    b_mean_name = 'data_' + base_name  + '.csv'
    fb_mean_name = os.path.join(ret['res_folder'], b_mean_name)
    with open(fb_mean_name, 'a') as f:
        if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
            f.write('# Statistic over minimum combined R & t errors of most likely extrinsics (Rt_diff) and statistic '
                    'of corresponding parameters ' +
                    ' over all ' + keywords['x_axis_column'][0] + ' for different ' +
                    '-'.join(map(str, ret['partitions'])) + '\n')
        else:
            f.write('# Statistic over minimum combined R & t errors (Rt_diff) and statistic of '
                    'corresponding parameters ' +
                    ' over all ' + keywords['x_axis_column'][0] + ' for different ' +
                    '-'.join(map(str, ret['partitions'])) + '\n')
        f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
        par_stat.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    b_mean = par_stat.xs('mean', axis=1, level=1, drop_level=True).copy(deep=True)
    if len(ret['it_parameters']) > 1:
        b_mean['options_tex'] = [', '.join(['{:.3f}'.format(float(val)) for _, val in row.iteritems()])
                                 for _, row in b_mean[ret['it_parameters']].iterrows()]
    else:
        b_mean['options_tex'] = ['{:.3f}'.format(float(val)) for _, val in b_mean[ret['it_parameters'][0]].iteritems()]

    base_name = 'mean_min_'
    title = 'Mean Values of Smallest Combined R \\& t Errors '
    if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
        base_name += 'ml_'
        title += 'of Most Likely Extrinsics '
    base_name += 'RTerrors_and_mean_corresp_opts_' + \
                 short_concat_str(ret['it_parameters']) + '_for_part_' + ret['dataf_name_partition']
    title += 'and Their Corresponding ' + \
             'Mean Parameters ' + ret['sub_title_it_pars'] + \
             ' Over All ' + replaceCSVLabels(keywords['x_axis_column'][0], True, True, True) + \
             ' for Different ' + ret['sub_title_partitions']
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': ret['gloss']
                 }
    b_mean_name = 'data_' + base_name + '.csv'
    fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
    with open(fb_mean_name, 'a') as f:
        if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
            f.write('# Mean values over minimum combined R & t errors of most likely extrinsics (Rt_diff) '
                    'and corresponding mean parameters ' +
                    ' over all ' + keywords['x_axis_column'][0] + ' for different ' +
                    '-'.join(map(str, ret['partitions'])) + '\n')
        else:
            f.write('# Mean values over minimum combined R & t errors (Rt_diff) and corresponding mean parameters ' +
                    ' over all ' + keywords['x_axis_column'][0] + ' for different ' +
                    '-'.join(map(str, ret['partitions'])) + '\n')
        f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
        b_mean.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    sc_beginning = 'Mean values of smallest combined R \\& t errors '
    if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
        section_name = sc_beginning + 'of most likely extrinsics and their corresponding\\\\' + \
                       'mean parameters ' + strToLower(ret['sub_title_it_pars']) + \
                       ' over all ' + replaceCSVLabels(keywords['x_axis_column'][0], True, False, True) + \
                       ' for different ' + strToLower(ret['sub_title_partitions'])
        caption = sc_beginning + 'of most likely extrinsics and their ' + \
                  'corresponding mean parameters on top of each bar separated by a comma in the order ' + \
                  strToLower(ret['sub_title_it_pars']) + ' over all ' + \
                  replaceCSVLabels(keywords['x_axis_column'][0], True, False, True)
    else:
        section_name = sc_beginning + 'and their corresponding\\\\' + \
                       'mean parameters ' + strToLower(ret['sub_title_it_pars']) + \
                       ' over all ' + replaceCSVLabels(keywords['x_axis_column'][0], True, False, True) + \
                       ' for different ' + strToLower(ret['sub_title_partitions'])
        caption = sc_beginning + 'and their ' + \
                  'corresponding mean parameters on top of each bar separated by a comma in the order ' + \
                  strToLower(ret['sub_title_it_pars']) + ' over all ' + \
                  replaceCSVLabels(keywords['x_axis_column'][0], True, False, True)
    _, use_limits, use_log, exp_value = get_limits_log_exp(b_mean, True, True, False, ['options_tex'] +
                                                           ret['it_parameters'])
    is_neg = check_if_neg_values(b_mean, 'Rt_diff', use_log, use_limits)
    is_numeric = pd.to_numeric(b_mean.reset_index()[ret['partitions'][0]], errors='coerce').notnull().all()
    section_name = split_large_titles(section_name)
    exp_value = enl_space_title(exp_value, section_name, b_mean, ret['partitions'][0],
                                1, 'ybar')
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'ybar',
                                  'plots': ['Rt_diff'],
                                  'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': replaceCSVLabels(ret['partitions'][0]),
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': ret['partitions'][0],
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': ['options_tex'],
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 45,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': None,
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': True if not is_numeric else False,
                                  'use_log_y_axis': use_log,
                                  'xaxis_txt_rows': 1,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': exp_value,
                                  'large_meta_space_needed': True,
                                  'is_neg': is_neg,
                                  'caption': caption
                                  })

    base_out_name = 'tex_' + base_name
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if ret['build_pdf'][2]:
        res1 = compile_tex(rendered_tex,
                           ret['tex_folder'],
                           texf_name,
                           tex_infos['make_index'],
                           os.path.join(ret['pdf_folder'], pdf_name),
                           tex_infos['figs_externalize'])
    else:
        res1 = compile_tex(rendered_tex, ret['tex_folder'], texf_name)
    if res1 != 0:
        ret['res'] += abs(res1)
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    b_mmean = b_mean.drop('options_tex', axis=1).mean()

    main_parameter_name = keywords['res_par_name']  # 'USAC_opt_refine_min_time'
    # Check if file and parameters exist
    from usac_eval import check_par_file_exists, NoAliasDumper
    ppar_file, ret['res'] = check_par_file_exists(main_parameter_name, keywords['res_folder'], ret['res'])
    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = b_mmean[ret['it_parameters']].to_numpy()
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
            alg_w[val] = float(alg_comb_bestl[i])
        yaml.dump({main_parameter_name: {'Algorithm': alg_w,
                                         'mean_Rt_error': float(b_mmean['Rt_diff'])}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    return ret['res']


def get_best_comb_3d_scenes_ml_1(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    eval_columns_init = deepcopy(keywords['eval_columns'])
    eval_cols1 = [a for a in eval_columns_init if 'mostLikely' not in a]
    eval_cols2 = [a for a in eval_columns_init if 'mostLikely' in a]
    data1 = keywords['data'].drop(eval_cols2, axis=1)
    data2 = keywords['data'].drop(eval_cols1, axis=1)
    res = 0
    if eval_cols1:
        keywords['eval_columns'] = eval_cols1
        keywords['data'] = data1
        res += get_best_comb_3d_scenes_1(**keywords)

    if eval_cols2:
        keywords['eval_columns'] = eval_cols2
        keywords['data'] = data2
        keywords['is_mostLikely'] = True
        res_folder_parent = os.path.abspath(os.path.join(keywords['res_folder'], os.pardir))  # Get parent directory
        last_folder = os.path.basename(os.path.normpath(keywords['res_folder']))
        ml_folder = os.path.join(res_folder_parent, last_folder + '_ml')
        cnt = 1
        calc_vals = True
        ml_folder_init = ml_folder
        while os.path.exists(ml_folder):
            ml_folder = ml_folder_init + '_' + str(int(cnt))
            cnt += 1
        try:
            os.mkdir(ml_folder)
        except FileExistsError:
            print('Folder', ml_folder, 'for storing statistics data already exists')
        except:
            print("Unexpected error (Unable to create directory for storing special function data):", sys.exc_info()[0])
            calc_vals = False
        if calc_vals:
            keywords['res_folder'] = ml_folder
            keywords['res_par_name'] += '_ml'
            res += get_best_comb_3d_scenes_1(**keywords)
    return res


def get_best_comb_3d_scenes_1(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    from usac_eval import pars_calc_multiple_fig_partitions
    from statistics_and_plot import tex_string_coding_style, \
        short_concat_str, \
        replaceCSVLabels, \
        insert_str_option_values, \
        get_limits_log_exp, \
        split_large_titles, \
        enl_space_title, \
        strToLower, \
        check_legend_enlarge, \
        calcNrLegendCols, \
        check_if_neg_values
    ret = pars_calc_multiple_fig_partitions(**keywords)
    b_min = ret['b'].stack().reset_index()
    b_min.rename(columns={b_min.columns[-1]: 'Rt_diff'}, inplace=True)
    b_min = b_min.loc[b_min.groupby(ret['partitions'] + keywords['xy_axis_columns'])['Rt_diff'].idxmin()]
    b_min1 = b_min.set_index(ret['it_parameters'])
    if len(ret['it_parameters']) > 1:
        b_min1.index = ['-'.join(map(str, a)) for a in b_min1.index]
        it_pars_name = '-'.join(map(str, ret['it_parameters']))
        b_min1.index.name = it_pars_name
    else:
        it_pars_name = ret['it_parameters'][0]
    b_min_grp = b_min1.reset_index().set_index(keywords['xy_axis_columns']).groupby(ret['partitions'])
    grp_keys = b_min_grp.groups.keys()
    base_name = 'min_'
    title = 'Smallest Combined R \\& t Errors '
    if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
        base_name += 'ml_'
        title += 'of Most Likely Extrinsics '
    base_name += 'RTerrors_vs_' + keywords['xy_axis_columns'][0] + '_and_' + \
                 keywords['xy_axis_columns'][1] + '_with_corresp_opts_' + \
                 short_concat_str(ret['it_parameters']) + '_for_part_' + ret['dataf_name_partition']
    title += 'and Their Corresponding ' + \
             'Parameters ' + ret['sub_title_it_pars'] + \
             ' vs ' + replaceCSVLabels(keywords['xy_axis_columns'][0], True, True, True) + \
             ' and ' + replaceCSVLabels(keywords['xy_axis_columns'][1], True, True, True) + \
             ' for Different ' + ret['sub_title_partitions']
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': ret['gloss']
                 }
    for grp in grp_keys:
        if grp == 'nv':
            continue
        tmp = b_min_grp.get_group(grp).drop(ret['partitions'], axis=1)
        if len(ret['it_parameters']) > 1:
            tmp['options_tex'] = [', '.join(['{:.3f}'.format(float(b)) for b in a.split('-')])
                                  for a in tmp[it_pars_name].values]
        else:
            tmp['options_tex'] = ['{:.3f}'.format(float(a)) for a in tmp[it_pars_name].values]
        # tmp = tmp.reset_index().set_index([it_pars_name] + keywords['xy_axis_columns'])
        tmp = tmp.unstack()
        tmp.columns = ['-'.join(map(str, a)) for a in tmp.columns]
        # tmp = tmp.reset_index().set_index(keywords['xy_axis_columns'][0])
        plots = [a for a in tmp.columns if 'Rt_diff' in a]
        meta_cols = [a for a in tmp.columns if 'options_tex' in a]
        it_pars_names = [a for a in tmp.columns if it_pars_name in a]
        b_mean_name = 'data_' + base_name + \
                      (str(grp) if len(ret['partitions']) == 1 else '-'.join(map(str, grp))) + '.csv'
        fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
        with open(fb_mean_name, 'a') as f:
            if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
                f.write('# Minimum combined R & t errors of most likely extrinsics (Rt_diff) '
                        'and corresponding option vs ' +
                        keywords['xy_axis_columns'][0] + ' and ' + keywords['xy_axis_columns'][1] +
                        ' for partition ' + '-'.join(map(str, ret['partitions'])) + ' = ' +
                        (str(grp) if len(ret['partitions']) == 1 else '-'.join(map(str, grp))) + '\n')
            else:
                f.write('# Minimum combined R & t errors (Rt_diff) and corresponding option vs ' +
                        keywords['xy_axis_columns'][0] + ' and ' + keywords['xy_axis_columns'][1] +
                        ' for partition ' + '-'.join(map(str, ret['partitions'])) + ' = ' +
                        (str(grp) if len(ret['partitions']) == 1 else '-'.join(map(str, grp))) + '\n')
            f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
            tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        sc_begin = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ '
        if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
            section_name = sc_begin + 'of most likely extrinsics and their ' + \
                           'corresponding options vs ' + \
                           replaceCSVLabels(keywords['xy_axis_columns'][0], False, False, True) + ' and ' + \
                           replaceCSVLabels(keywords['xy_axis_columns'][1], False, False, True) + \
                           ' for property ' + insert_str_option_values(ret['partitions'], grp)
            caption = sc_begin + 'of most likely extrinsics and their ' + \
                      'corresponding options on top of each bar separated by a comma in the order ' + \
                      strToLower(ret['sub_title_it_pars']) + ' vs ' + \
                      replaceCSVLabels(keywords['xy_axis_columns'][0], False, False, True) + ' and ' + \
                      replaceCSVLabels(keywords['xy_axis_columns'][1], False, False, True) + \
                      ' for the ' + insert_str_option_values(ret['partitions'], grp)
        else:
            section_name = sc_begin + 'and their ' + \
                           'corresponding options vs ' + \
                           replaceCSVLabels(keywords['xy_axis_columns'][0], False, False, True) + ' and ' + \
                           replaceCSVLabels(keywords['xy_axis_columns'][1], False, False, True) + \
                           ' for property ' + insert_str_option_values(ret['partitions'], grp)
            caption = sc_begin + 'and their ' + \
                      'corresponding options on top of each bar separated by a comma in the order ' + \
                      strToLower(ret['sub_title_it_pars']) + ' vs ' + \
                      replaceCSVLabels(keywords['xy_axis_columns'][0], False, False, True) + ' and ' + \
                      replaceCSVLabels(keywords['xy_axis_columns'][1], False, False, True) + \
                      ' for the ' + insert_str_option_values(ret['partitions'], grp)
        _, use_limits, use_log, exp_value = get_limits_log_exp(tmp, True, True, False, it_pars_names + meta_cols)
        is_neg = check_if_neg_values(tmp, plots, use_log, use_limits)
        is_numeric = pd.to_numeric(tmp.reset_index()[keywords['xy_axis_columns'][0]], errors='coerce').notnull().all()
        section_name = split_large_titles(section_name)
        enlarge_lbl_dist = check_legend_enlarge(tmp, keywords['xy_axis_columns'][0], len(plots), 'xbar')
        exp_value = enl_space_title(exp_value, section_name, tmp, keywords['xy_axis_columns'][0],
                                    len(plots), 'xbar')
        tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                      'name': section_name.replace('\\\\', ' '),
                                      'title': section_name,
                                      'title_rows': section_name.count('\\\\'),
                                      'fig_type': 'xbar',
                                      'plots': plots,
                                      'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                      # Label/column name of axis with bars. For xbar it labels the y-axis
                                      'label_x': replaceCSVLabels(keywords['xy_axis_columns'][0]),
                                      # Column name of axis with bars. For xbar it is the column for the y-axis
                                      'print_x': keywords['xy_axis_columns'][0],
                                      # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                      'print_meta': True,
                                      'plot_meta': meta_cols,
                                      # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                      'rotate_meta': 0,
                                      'limits': None,
                                      # If None, no legend is used, otherwise use a list
                                      'legend': [' -- '.join([replaceCSVLabels(b)
                                                              for b in a.split('-')]) for a in plots],
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      # The x/y-axis values are given as strings if True
                                      'use_string_labels': True if not is_numeric else False,
                                      'use_log_y_axis': use_log,
                                      'xaxis_txt_rows': 1,
                                      'enlarge_lbl_dist': enlarge_lbl_dist,
                                      'enlarge_title_space': exp_value,
                                      'large_meta_space_needed': True,
                                      'is_neg': is_neg,
                                      'caption': caption
                                      })
        tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    base_out_name = 'tex_' + base_name
    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if ret['build_pdf'][1]:
        res1 = compile_tex(rendered_tex,
                           ret['tex_folder'],
                           texf_name,
                           tex_infos['make_index'],
                           os.path.join(ret['pdf_folder'], pdf_name),
                           tex_infos['figs_externalize'])
    else:
        res1 = compile_tex(rendered_tex, ret['tex_folder'], texf_name)
    if res1 != 0:
        ret['res'] += abs(res1)
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    par_stat = b_min.drop(keywords['xy_axis_columns'][0], axis=1).groupby(ret['partitions'] +
                                                                          [keywords['xy_axis_columns'][1]]).describe()
    base_name = 'stats_min_'
    if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
        base_name += 'ml_'
    base_name += 'RTerrors_and_stats_corresp_opts_' + \
                 short_concat_str(ret['it_parameters']) + '_for_part_' + ret['dataf_name_partition']
    b_mean_name = 'data_' + base_name  + '.csv'
    fb_mean_name = os.path.join(ret['res_folder'], b_mean_name)
    with open(fb_mean_name, 'a') as f:
        if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
            f.write('# Statistic over minimum combined R & t errors of most likely extrinsics (Rt_diff) '
                    'and statistic of corresponding parameters ' +
                    ' over all ' + keywords['xy_axis_columns'][0] + ' for different ' +
                    '-'.join(map(str, ret['partitions'])) + '\n')
        else:
            f.write('# Statistic over minimum combined R & t errors (Rt_diff) and statistic of '
                    'corresponding parameters ' +
                    ' over all ' + keywords['xy_axis_columns'][0] + ' for different ' +
                    '-'.join(map(str, ret['partitions'])) + '\n')
        f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
        par_stat.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    b_mean = par_stat.xs('mean', axis=1, level=1, drop_level=True).copy(deep=True)
    if len(ret['it_parameters']) > 1:
        b_mean['options_tex'] = [', '.join(['{:.3f}'.format(float(val)) for _, val in row.iteritems()])
                                 for _, row in b_mean[ret['it_parameters']].iterrows()]
    else:
        b_mean['options_tex'] = ['{:.3f}'.format(float(val)) for _, val in b_mean[ret['it_parameters'][0]].iteritems()]
    b_mean1 = b_mean.unstack()
    b_mean1.columns = ['-'.join(map(str, a)) for a in b_mean1.columns]
    plots = [a for a in b_mean1.columns if 'Rt_diff' in a]
    meta_cols = [a for a in b_mean1.columns if 'options_tex' in a]
    it_pars_names = [a for a in b_mean1.columns for b in ret['it_parameters'] if b in a]

    base_name = 'mean_min_'
    title = 'Mean Values of Smallest Combined R \\& t Errors '
    if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
        base_name += 'ml_'
        title += 'of Most Likely Extrinsics '
    title += 'and Their Corresponding ' + \
             'Mean Parameters ' + ret['sub_title_it_pars'] + \
             ' Over All ' + replaceCSVLabels(keywords['xy_axis_columns'][0], True, True, True) + ' vs ' + \
             replaceCSVLabels(keywords['xy_axis_columns'][1], True, True, True) + \
             ' for Different ' + ret['sub_title_partitions']
    base_name += 'RTerrors_and_mean_corresp_opts_' + \
                 short_concat_str(ret['it_parameters']) + '_for_part_' + ret['dataf_name_partition']
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': ret['gloss']
                 }
    b_mean_name = 'data_' + base_name + '.csv'
    fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
    with open(fb_mean_name, 'a') as f:
        if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
            f.write('# Mean values over minimum combined R & t errors of most likely extrinsics (b_min) '
                    'and corresponding mean parameters ' +
                    ' over all ' + keywords['xy_axis_columns'][0] + ' vs ' +
                    keywords['xy_axis_columns'][1] + ' for different ' +
                    '-'.join(map(str, ret['partitions'])) + '\n')
        else:
            f.write('# Mean values over minimum combined R & t errors (b_min) '
                    'and corresponding mean parameters ' +
                    ' over all ' + keywords['xy_axis_columns'][0] + ' vs ' +
                    keywords['xy_axis_columns'][1] + ' for different ' +
                    '-'.join(map(str, ret['partitions'])) + '\n')
        f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
        b_mean1.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    sc_begin = 'Mean values of smallest combined R \\& t errors '
    if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
        section_name = sc_begin + 'of most likely extrinsics and their corresponding\\\\' + \
                       ' mean parameters ' + strToLower(ret['sub_title_it_pars']) + \
                       ' over all ' + replaceCSVLabels(keywords['xy_axis_columns'][0], True, False, True) + ' vs ' + \
                       replaceCSVLabels(keywords['xy_axis_columns'][1], True, False, True) + \
                       ' for different ' + strToLower(ret['sub_title_partitions'])
        caption = sc_begin + 'of most likely extrinsics and their ' + \
                  'corresponding mean parameters on top of each bar separated by a comma in the order ' + \
                  strToLower(ret['sub_title_it_pars']) + ' over all ' + \
                  replaceCSVLabels(keywords['xy_axis_columns'][0], True, False, True)
    else:
        section_name = sc_begin + 'and their corresponding\\\\' + \
                       ' mean parameters ' + strToLower(ret['sub_title_it_pars']) + \
                       ' over all ' + replaceCSVLabels(keywords['xy_axis_columns'][0], True, False, True) + ' vs ' + \
                       replaceCSVLabels(keywords['xy_axis_columns'][1], True, False, True) + \
                       ' for different ' + strToLower(ret['sub_title_partitions'])
        caption = sc_begin + 'and their ' + \
                  'corresponding mean parameters on top of each bar separated by a comma in the order ' + \
                  strToLower(ret['sub_title_it_pars']) + ' over all ' + \
                  replaceCSVLabels(keywords['xy_axis_columns'][0], True, False, True)
    _, use_limits, use_log, exp_value = get_limits_log_exp(b_mean1, True, True, False, it_pars_names + meta_cols)
    is_neg = check_if_neg_values(b_mean1, plots, use_log, use_limits)
    is_numeric = pd.to_numeric(b_mean1.reset_index()[ret['partitions'][0]], errors='coerce').notnull().all()
    section_name = split_large_titles(section_name)
    enlarge_lbl_dist = check_legend_enlarge(b_mean1, ret['partitions'][0], len(plots), 'xbar')
    exp_value = enl_space_title(exp_value, section_name, b_mean1, ret['partitions'][0],
                                len(plots), 'xbar')

    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'xbar',
                                  'plots': plots,
                                  'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': replaceCSVLabels(ret['partitions'][0]),
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': ret['partitions'][0],
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': meta_cols,
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 0,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': [' -- '.join([replaceCSVLabels(b) for b in a.split('-')]) for a in plots],
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': not is_numeric,
                                  'use_log_y_axis': use_log,
                                  'xaxis_txt_rows': 1,
                                  'enlarge_lbl_dist': enlarge_lbl_dist,
                                  'enlarge_title_space': exp_value,
                                  'large_meta_space_needed': True,
                                  'is_neg': is_neg,
                                  'caption': caption
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    base_out_name = 'tex_' + base_name
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if ret['build_pdf'][2]:
        res1 = compile_tex(rendered_tex,
                           ret['tex_folder'],
                           texf_name,
                           tex_infos['make_index'],
                           os.path.join(ret['pdf_folder'], pdf_name),
                           tex_infos['figs_externalize'])
    else:
        res1 = compile_tex(rendered_tex, ret['tex_folder'], texf_name)
    if res1 != 0:
        ret['res'] += abs(res1)
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    b_mmean = b_mean.reset_index().drop(['options_tex'] +
                                        ret['partitions'], axis=1).groupby(keywords['xy_axis_columns'][1]).mean()
    if len(ret['it_parameters']) > 1:
        b_mmean['options_tex'] = [', '.join(['{:.3f}'.format(float(val)) for _, val in row.iteritems()])
                                  for _, row in b_mmean[ret['it_parameters']].iterrows()]
    else:
        b_mmean['options_tex'] = ['{:.3f}'.format(float(val))
                                  for _, val in b_mmean[ret['it_parameters'][0]].iteritems()]

    base_name = 'double_mean_min_'
    title = 'Mean Values of Smallest Combined R \\& t Errors '
    if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
        base_name += 'ml_'
        title += 'of Most Likely Extrinsics '
    base_name += 'RTerrors_and_mean_corresp_opts_' + \
                short_concat_str(ret['it_parameters'])
    title += 'and Their Corresponding ' + \
             'Mean Parameters ' + ret['sub_title_it_pars'] + \
             ' Over All ' + replaceCSVLabels(keywords['xy_axis_columns'][0], True, True, True) + \
             ' and ' + ret['sub_title_partitions'] + ' vs ' + \
             replaceCSVLabels(keywords['xy_axis_columns'][1], True, True, True)
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': ret['gloss']
                 }
    b_mean_name = 'data_' + base_name + '.csv'
    fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
    with open(fb_mean_name, 'a') as f:
        if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
            f.write('# Mean values over minimum combined R & t errors of most likely extrinsics (b_min) '
                    'and corresponding mean parameters ' +
                    ' over all ' + keywords['xy_axis_columns'][0] + ' and partitions ' +
                    '-'.join(map(str, ret['partitions'])) + ' vs ' +
                    keywords['xy_axis_columns'][1] + ' for different ' + '\n')
        else:
            f.write('# Mean values over minimum combined R & t errors (b_min) and corresponding mean parameters ' +
                    ' over all ' + keywords['xy_axis_columns'][0] + ' and partitions ' +
                    '-'.join(map(str, ret['partitions'])) + ' vs ' +
                    keywords['xy_axis_columns'][1] + ' for different ' + '\n')
        f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
        b_mmean.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    sc_begin = 'Mean values of smallest combined R \\& t errors '
    if 'is_mostLikely' in keywords and keywords['is_mostLikely']:
        section_name = sc_begin + 'of most likely extrinsics and their corresponding\\\\' + \
                       ' mean parameters ' + strToLower(ret['sub_title_it_pars']) + \
                       ' over all ' + replaceCSVLabels(keywords['xy_axis_columns'][0], True, False, True) + ' and ' + \
                       strToLower(ret['sub_title_partitions']) + ' vs ' + \
                       replaceCSVLabels(keywords['xy_axis_columns'][1], True, False, True)
        caption = sc_begin + 'of most likely extrinsics and their ' + \
                  ' corresponding mean parameters on top of each bar separated by a comma in the order ' + \
                  strToLower(ret['sub_title_it_pars']) + ' over all ' + \
                  replaceCSVLabels(keywords['xy_axis_columns'][0], True, False, True) + ' and ' + \
                  strToLower(ret['sub_title_partitions'])
    else:
        section_name = sc_begin + 'and their corresponding\\\\' + \
                       ' mean parameters ' + strToLower(ret['sub_title_it_pars']) + \
                       ' over all ' + replaceCSVLabels(keywords['xy_axis_columns'][0], True, False, True) + ' and ' + \
                       strToLower(ret['sub_title_partitions']) + ' vs ' + \
                       replaceCSVLabels(keywords['xy_axis_columns'][1], True, False, True)
        caption = sc_begin + 'and their ' + \
                  ' corresponding mean parameters on top of each bar separated by a comma in the order ' + \
                  strToLower(ret['sub_title_it_pars']) + ' over all ' + \
                  replaceCSVLabels(keywords['xy_axis_columns'][0], True, False, True) + ' and ' + \
                  strToLower(ret['sub_title_partitions'])
    _, use_limits, use_log, exp_value = get_limits_log_exp(b_mmean, True, True, False, ['options_tex'] +
                                                           ret['it_parameters'])
    is_neg = check_if_neg_values(b_mmean, 'Rt_diff', use_log, use_limits)
    is_numeric = pd.to_numeric(b_mmean.reset_index()[keywords['xy_axis_columns'][1]], errors='coerce').notnull().all()
    section_name = split_large_titles(section_name)
    enlarge_lbl_dist = check_legend_enlarge(b_mmean, keywords['xy_axis_columns'][1], 1, 'ybar')
    exp_value = enl_space_title(exp_value, section_name, b_mmean, keywords['xy_axis_columns'][1],
                                1, 'ybar')

    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'ybar',
                                  'plots': ['Rt_diff'],
                                  'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': replaceCSVLabels(keywords['xy_axis_columns'][1]),
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': keywords['xy_axis_columns'][1],
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': ['options_tex'],
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 45,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': None,
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': not is_numeric,
                                  'use_log_y_axis': use_log,
                                  'xaxis_txt_rows': 1,
                                  'enlarge_lbl_dist': enlarge_lbl_dist,
                                  'enlarge_title_space': exp_value,
                                  'large_meta_space_needed': True,
                                  'is_neg': is_neg,
                                  'caption': caption
                                  })

    base_out_name = 'tex_' + base_name
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if ret['build_pdf'][3]:
        res1 = compile_tex(rendered_tex,
                           ret['tex_folder'],
                           texf_name,
                           tex_infos['make_index'],
                           os.path.join(ret['pdf_folder'], pdf_name),
                           tex_infos['figs_externalize'])
    else:
        res1 = compile_tex(rendered_tex, ret['tex_folder'], texf_name)
    if res1 != 0:
        ret['res'] += abs(res1)
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    b_mmmean = b_mmean.drop('options_tex', axis=1).mean()

    main_parameter_name = keywords['res_par_name']  # 'USAC_opt_refine_min_time'
    # Check if file and parameters exist
    from usac_eval import check_par_file_exists, NoAliasDumper
    ppar_file, ret['res'] = check_par_file_exists(main_parameter_name, keywords['res_folder'], ret['res'])
    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = b_mmmean[ret['it_parameters']].to_numpy()
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
            alg_w[val] = float(alg_comb_bestl[i])
        yaml.dump({main_parameter_name: {'Algorithm': alg_w,
                                         'mean_Rt_error': float(b_mmmean['Rt_diff'])}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    if 'comp_res' in keywords and keywords['comp_res'] and isinstance(keywords['comp_res'], list):
        keywords['res'] = ret['res']
        ret['res'] = compare_evaluations(**keywords)

    return ret['res']


def calc_calib_delay(**keywords):
    if 'data_separators' not in keywords:
        raise ValueError('data_separators are necessary.')
    if 'eval_on' not in keywords:
        raise ValueError('Information (column name/s) for which evaluation is performed must be provided.')
    if 'change_Nr' not in keywords:
        raise ValueError('Frame number when the extrinsics change must be provided (index starts at 0)')
    if 'additional_data' not in keywords:
        raise ValueError('additional_data must be specified and should include column names like rt_change_pos.')
    if 'scene' not in keywords:
        raise ValueError('scene must be specified')
    if not isinstance(keywords['scene'], str):
        raise ValueError('Currently, only an evaluation on a single scene is supported')
    if 'comb_rt' in keywords and keywords['comb_rt']:
        from corr_pool_eval import combine_rt_diff2
        data, keywords = combine_rt_diff2(keywords['data'], keywords)
    else:
        data = keywords['data']
    keywords = prepare_io(**keywords)
    from statistics_and_plot import check_if_series, \
        short_concat_str, \
        get_short_scene_description, \
        replaceCSVLabels, \
        combine_str_for_title, \
        add_to_glossary_eval, \
        add_to_glossary, \
        strToLower, \
        capitalizeFirstChar, \
        calcNrLegendCols, \
        get_limits_log_exp, \
        split_large_titles, \
        check_legend_enlarge, \
        enl_space_title, \
        findUnit, \
        add_val_to_opt_str, \
        replace_stat_names, \
        split_large_str, \
        replace_stat_names_col_tex
    needed_cols = list(dict.fromkeys(keywords['data_separators'] +
                                     keywords['it_parameters'] +
                                     keywords['eval_on'] +
                                     keywords['additional_data'] +
                                     keywords['x_axis_column']))
    df = data[needed_cols]
    grpd_cols = keywords['data_separators'] + keywords['it_parameters']
    df_grp = df.groupby(grpd_cols)
    grp_keys = df_grp.groups.keys()
    df_list = []
    for grp in grp_keys:
        tmp = df_grp.get_group(grp).copy(deep=True)
        tmp.loc[:, keywords['eval_on'][0]] = tmp.loc[:, keywords['eval_on'][0]].abs()
        #Check for the correctness of the change number
        if int(tmp['rt_change_pos'].iloc[0]) != keywords['change_Nr']:
            warnings.warn('Given frame number when extrinsics change doesnt match the estimated number. '
                          'Taking estimated number.', UserWarning)
            keywords['change_Nr'] = int(tmp['rt_change_pos'].iloc[0])
        tmp1 = tmp.loc[tmp['Nr'] < keywords['change_Nr']]
        min_val = tmp1[keywords['eval_on'][0]].min()
        max_val = tmp1[keywords['eval_on'][0]].max()
        rng80 = 0.8 * (max_val - min_val) + min_val
        p1_stats = tmp1.loc[tmp1[keywords['eval_on'][0]] < rng80, keywords['eval_on']].describe()
        th = p1_stats[keywords['eval_on'][0]]['mean'] + 2.576 * p1_stats[keywords['eval_on'][0]]['std']
        test_rise = tmp.loc[((tmp[keywords['eval_on'][0]] > th) &
                            (tmp['Nr'] >= keywords['change_Nr']) &
                            (tmp['Nr'] < (keywords['change_Nr'] + 2)))]
        if test_rise.empty:
            fd = 0
            fpos = keywords['change_Nr']
        else:
            tmp2 = tmp.loc[((tmp[keywords['eval_on'][0]] <= th) &
                            (tmp['Nr'] >= keywords['change_Nr']))]
            if check_if_series(tmp2):
                fpos = tmp2['Nr']
                fd = fpos - keywords['change_Nr']
            elif tmp2.shape[0] == 1:
                fpos = tmp2['Nr'].iloc[0]
                fd = fpos - keywords['change_Nr']
            else:
                tmp2.set_index('Nr', inplace=True)
                tmp_iter = tmp2.iterrows()
                idx_old, _ = next(tmp_iter)
                fpos = 0
                for idx, _ in tmp_iter:
                    if idx == idx_old + 1:
                        fpos = idx_old
                        break
                if fpos > 0:
                    fd = fpos - keywords['change_Nr']
                else:
                    fpos = tmp2.index[0]
                    fd = fpos - keywords['change_Nr']
        tmp['fd'] = [np.NaN] * int(tmp.shape[0])
        tmp.loc[(tmp['Nr'] == fpos), 'fd'] = fd
        df_list.append(tmp)
    df_new = pd.concat(df_list, axis=0, ignore_index=False)

    gloss = add_to_glossary_eval(keywords['eval_on'])
    n_gloss_calced = True
    res = 0
    all_mean = []
    for i, it in enumerate(keywords['data_separators']):
        av_pars = [a for a in keywords['data_separators'] if a != it]
        df1 = df_new.groupby(it)
        grp_keys = df1.groups.keys()
        hist_list = []
        par_stats_list = []
        if n_gloss_calced:
            gloss = add_to_glossary(grp_keys, gloss)
            for it1 in av_pars:
                gloss = add_to_glossary(df_new[it1].unique().tolist(), gloss)
            n_gloss_calced = False
        for grp in grp_keys:
            tmp = df1.get_group(grp)
            nr_max = tmp['Nr'].max()
            possis = 1
            for p in keywords['it_parameters']:
                possis *= tmp[p].nunique()
            possis1 = max(3, int(round(0.01 * float(possis))))
            hist, bin_edges = np.histogram(tmp['fd'].dropna().values,
                                           bins=list(range(0, nr_max - keywords['change_Nr'] + 2)), density=False)
            fd_good = int(bin_edges[np.nonzero(hist >= possis1)[0][0]])
            possis2 = max(1, int(round(0.005 * float(possis))))
            hist1 = hist[hist >= possis2]
            edges1 = bin_edges[np.nonzero(hist >= possis2)]
            hist_list.append(pd.DataFrame(data={'fd': edges1, 'count': hist1}, columns=['fd', 'count']).set_index('fd'))
            par_stats_list.append(tmp.loc[tmp['fd'] == fd_good, keywords['it_parameters'] + ['fd']].describe())

        df_hist = pd.concat(hist_list, axis=1, keys=grp_keys, ignore_index=False)
        keywords['units'].append(('fd', '/\\# of frames',))

        # Plot histogram
        df_hist.columns = ['-'.join(map(str, a)) for a in df_hist.columns]
        base_name = 'histogram_frame_delay_vs_' + it + '_opts_' + \
                    short_concat_str(keywords['it_parameters'])
        hist_title_p01 = 'Histogram on Frame Delays for Reaching a Correct Calibration After an Abrupt ' + \
                         'Change in Extrinsics (' + \
                         get_short_scene_description(keywords['scene']) + ') vs '
        hist_title_p02 = replaceCSVLabels(it, True, True, True)
        hist_title_p03 = ' Over Data Partitions ' + combine_str_for_title(av_pars)
        hist_title_p1 = hist_title_p01 + hist_title_p02 + hist_title_p03
        hist_title_p2 = ' for Parameter Variations of ' + \
                        keywords['sub_title_it_pars'] + ' Based on ' + \
                        replaceCSVLabels(keywords['eval_on'][0], True, True, True)
        hist_title = hist_title_p1 + hist_title_p2
        tex_infos = {'title': hist_title,
                     'sections': [],
                     # Builds an index with hyperrefs on the beginning of the pdf
                     'make_index': True,
                     # If True, the figures are adapted to the page height if they are too big
                     'ctrl_fig_size': True,
                     # If true, a pdf is generated for every figure and inserted as image in a second run
                     'figs_externalize': False,
                     # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                     'fill_bar': True,
                     # Builds a list of abbrevations from a list of dicts
                     'abbreviations': gloss
                     }
        b_mean_name = 'data_' + base_name + '.csv'
        fb_mean_name = os.path.join(keywords['tdata_folder'], b_mean_name)
        with open(fb_mean_name, 'a') as f:
            f.write('# Histogram on Frame Delays for Reaching a Correct Calibration After an Abrupt '
                    'Change in Extrinsics (' + keywords['scene'] + ')  vs ' + it + ' over data partitions '
                    '-'.join(av_pars) + ' based on ' + keywords['eval_on'][0] + '\n')
            f.write('# Parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
            df_hist.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        nr_bins = df_hist.shape[0] * df_hist.shape[1]
        if nr_bins < 31:
            section_name = capitalizeFirstChar(strToLower(hist_title_p1))
            caption = capitalizeFirstChar(strToLower(hist_title))
            _, use_limits, use_log, exp_value = get_limits_log_exp(df_hist, True, True, False)
            section_name = split_large_titles(section_name, 80)
            enlarge_lbl_dist = check_legend_enlarge(df_hist, 'fd', len(df_hist.columns.values), 'xbar')
            exp_value = enl_space_title(exp_value, section_name, df_hist, 'fd',
                                        len(df_hist.columns.values), 'xbar')

            tex_infos['sections'].append({'file': os.path.join(keywords['rel_data_path'], b_mean_name),
                                          'name': section_name.replace('\\\\', ' '),
                                          'title': section_name,
                                          'title_rows': section_name.count('\\\\'),
                                          'fig_type': 'xbar',
                                          'plots': df_hist.columns.values,
                                          'label_y': 'count',  # Label of the value axis. For xbar it labels the x-axis
                                          # Label/column name of axis with bars. For xbar it labels the y-axis
                                          'label_x': replaceCSVLabels('fd') + findUnit('fd', keywords['units']),
                                          # Column name of axis with bars. For xbar it is the column for the y-axis
                                          'print_x': 'fd',
                                          # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                          'print_meta': False,
                                          'plot_meta': None,
                                          # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                          'rotate_meta': 0,
                                          'limits': None,
                                          # If None, no legend is used, otherwise use a list
                                          'legend': [' -- '.join([replaceCSVLabels(b) for b in a.split('-')]) for a in
                                                     df_hist.columns.values],
                                          'legend_cols': 1,
                                          'use_marks': False,
                                          # The x/y-axis values are given as strings if True
                                          'use_string_labels': False,
                                          'use_log_y_axis': use_log,
                                          'xaxis_txt_rows': 1,
                                          'enlarge_lbl_dist': enlarge_lbl_dist,
                                          'enlarge_title_space': exp_value,
                                          'large_meta_space_needed': False,
                                          'is_neg': False,
                                          'caption': caption
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
        else:
            for col in df_hist.columns.values:
                part = [a for a in col.split('-') if a != 'count'][0]
                section_name = capitalizeFirstChar(strToLower(hist_title_p01)) + \
                               strToLower(add_val_to_opt_str(hist_title_p02, part)) + strToLower(hist_title_p03)
                caption = section_name + strToLower(hist_title_p2)
                _, use_limits, use_log, exp_value = get_limits_log_exp(df_hist, True, True, False, None, col)
                section_name = split_large_titles(section_name, 80)
                enlarge_lbl_dist = check_legend_enlarge(df_hist, 'fd', 1, 'xbar')
                exp_value = enl_space_title(exp_value, section_name, df_hist, 'fd',
                                            1, 'xbar')

                tex_infos['sections'].append({'file': os.path.join(keywords['rel_data_path'], b_mean_name),
                                              'name': section_name.replace('\\\\', ' '),
                                              'title': section_name,
                                              'title_rows': section_name.count('\\\\'),
                                              'fig_type': 'xbar',
                                              'plots': [col],
                                              'label_y': 'count',  # Label of the value axis. For xbar it labels the x-axis
                                              # Label/column name of axis with bars. For xbar it labels the y-axis
                                              'label_x': replaceCSVLabels('fd') + findUnit('fd', keywords['units']),
                                              # Column name of axis with bars. For xbar it is the column for the y-axis
                                              'print_x': 'fd',
                                              # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                              'print_meta': False,
                                              'plot_meta': None,
                                              # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                              'rotate_meta': 0,
                                              'limits': None,
                                              # If None, no legend is used, otherwise use a list
                                              'legend': None,
                                              'legend_cols': 1,
                                              'use_marks': False,
                                              # The x/y-axis values are given as strings if True
                                              'use_string_labels': False,
                                              'use_log_y_axis': use_log,
                                              'xaxis_txt_rows': 1,
                                              'enlarge_lbl_dist': enlarge_lbl_dist,
                                              'enlarge_title_space': exp_value,
                                              'large_meta_space_needed': True,
                                              'is_neg': False,
                                              'caption': caption
                                              })

        base_out_name = 'tex_' + base_name
        template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
        rendered_tex = template.render(title=tex_infos['title'],
                                       make_index=tex_infos['make_index'],
                                       ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                       figs_externalize=tex_infos['figs_externalize'],
                                       fill_bar=tex_infos['fill_bar'],
                                       sections=tex_infos['sections'],
                                       abbreviations=tex_infos['abbreviations'])
        texf_name = base_out_name + '.tex'
        pdf_name = base_out_name + '.pdf'
        if keywords['build_pdf'][0]:
            res1 = compile_tex(rendered_tex,
                               keywords['tex_folder'],
                               texf_name,
                               tex_infos['make_index'],
                               os.path.join(keywords['pdf_folder'], pdf_name),
                               tex_infos['figs_externalize'])
        else:
            res1 = compile_tex(rendered_tex, keywords['tex_folder'], texf_name)
        if res1 != 0:
            res += abs(res1)
            warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

        # Plot parameter statistics
        par_stats = pd.concat(par_stats_list, axis=0, keys=grp_keys, ignore_index=False, names=[it])
        base_name = '_opts_' + short_concat_str(keywords['it_parameters']) + '_calib_frame_delays_vs_' + it
        title_p01 = 'Parameters ' + keywords['sub_title_it_pars'] + \
                    ' for Smallest Frame Delays of Reaching a Correct Calibration After an Abrupt ' + \
                    'Change in Extrinsics (' + \
                    get_short_scene_description(keywords['scene']) + ') vs '
        title_p1 = 'Statistics on ' + title_p01 + hist_title_p02 + hist_title_p03
        title_p2 = ' Based on ' + replaceCSVLabels(keywords['eval_on'][0], True, True, True)
        title = title_p1 + title_p2
        tex_infos = {'title': title,
                     'sections': [],
                     # Builds an index with hyperrefs on the beginning of the pdf
                     'make_index': True,
                     # If True, the figures are adapted to the page height if they are too big
                     'ctrl_fig_size': True,
                     # If true, a pdf is generated for every figure and inserted as image in a second run
                     'figs_externalize': False,
                     # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                     'fill_bar': True,
                     # Builds a list of abbrevations from a list of dicts
                     'abbreviations': gloss
                     }
        stats = [a for a in list(dict.fromkeys(par_stats.index.get_level_values(1))) if a != 'count']
        # fds = par_stats.xs('fd', axis=1, level=1, drop_level=True)
        fds = par_stats['fd'].xs('mean', axis=0, level=1, drop_level=True).tolist()
        all_mean.append(par_stats.xs('mean', axis=0, level=1, drop_level=True).drop('fd', axis=1))
        for st in stats:
            section_name = replace_stat_names(st) + ' Values of ' + title_p01 + hist_title_p02 + hist_title_p03
            section_name = capitalizeFirstChar(strToLower(section_name))
            caption = section_name + strToLower(title_p2)
            p_mean = par_stats.xs(st, axis=0, level=1, drop_level=True).copy(deep=True)
            p_mean.drop('fd', axis=1, inplace=True)
            p_mean = p_mean.T
            fd_cols = ['fd-' + str(a) for a in p_mean.columns]
            for i_fd, fdc in enumerate(fd_cols):
                p_mean[fdc] = ['delay=' + str(int(fds[i_fd])) + ' frames'] * int(p_mean.shape[0])
            p_mean['options_tex'] = [split_large_str(replaceCSVLabels(a), 20) for a in p_mean.index]
            max_txt_rows = 1
            for idx, val in p_mean['options_tex'].iteritems():
                txt_rows = str(val).count('\\\\') + 1
                if txt_rows > max_txt_rows:
                    max_txt_rows = txt_rows
            legend_main = replaceCSVLabels(it)
            legend = [add_val_to_opt_str(legend_main, a) for a in p_mean.columns
                      if 'fd-' not in str(a) and 'options_tex' != str(a)]
            plots = [a for a in p_mean.columns if str(a) != 'options_tex' and 'fd-' not in str(a)]
            base_name1 = replace_stat_names_col_tex(st) + base_name
            b_mean_name = 'data_' + base_name1 + '.csv'
            fb_mean_name = os.path.join(keywords['tdata_folder'], b_mean_name)
            with open(fb_mean_name, 'a') as f:
                f.write('# ' + replace_stat_names(st, False) + ' values of parameters ' +
                        '-'.join(keywords['it_parameters']) +
                        ' for smallest frame delays (fd) of reaching a correct calibration after an abrupt ' +
                        'change in extrinsics (' + keywords['scene'] + ') vs ' +
                        it + ' over data partitions ' + '-'.join(map(str, av_pars)) +
                        ' based on ' + keywords['eval_on'][0] + '\n')
                f.write('# Parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
                p_mean.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

            _, use_limits, use_log, exp_value = get_limits_log_exp(p_mean, True, True, False, ['options_tex'] + fd_cols)
            section_name = split_large_titles(section_name, 80)
            enlarge_lbl_dist = check_legend_enlarge(p_mean, 'options_tex', len(plots), 'xbar')
            exp_value = enl_space_title(exp_value, section_name, p_mean, 'options_tex',
                                        len(plots), 'xbar')

            tex_infos['sections'].append({'file': os.path.join(keywords['rel_data_path'], b_mean_name),
                                          'name': section_name.replace('\\\\', ' '),
                                          'title': section_name,
                                          'title_rows': section_name.count('\\\\'),
                                          'fig_type': 'xbar',
                                          'plots': plots,
                                          'label_y': 'Option value',  # Label of the value axis. For xbar it labels the x-axis
                                          # Label/column name of axis with bars. For xbar it labels the y-axis
                                          'label_x': 'Option',
                                          # Column name of axis with bars. For xbar it is the column for the y-axis
                                          'print_x': 'options_tex',
                                          # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                          'print_meta': True,
                                          'plot_meta': fd_cols,
                                          # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                          'rotate_meta': 0,
                                          'limits': None,
                                          # If None, no legend is used, otherwise use a list
                                          'legend': legend,
                                          'legend_cols': 1,
                                          'use_marks': False,
                                          # The x/y-axis values are given as strings if True
                                          'use_string_labels': True,
                                          'use_log_y_axis': use_log,
                                          'xaxis_txt_rows': max_txt_rows,
                                          'enlarge_lbl_dist': enlarge_lbl_dist,
                                          'enlarge_title_space': exp_value,
                                          'large_meta_space_needed': True,
                                          'is_neg': False,
                                          'caption': caption
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

        base_out_name = 'tex_stats' + base_name
        rendered_tex = template.render(title=tex_infos['title'],
                                       make_index=tex_infos['make_index'],
                                       ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                       figs_externalize=tex_infos['figs_externalize'],
                                       fill_bar=tex_infos['fill_bar'],
                                       sections=tex_infos['sections'],
                                       abbreviations=tex_infos['abbreviations'])
        texf_name = base_out_name + '.tex'
        pdf_name = base_out_name + '.pdf'
        if keywords['build_pdf'][1]:
            res1 = compile_tex(rendered_tex,
                               keywords['tex_folder'],
                               texf_name,
                               tex_infos['make_index'],
                               os.path.join(keywords['pdf_folder'], pdf_name),
                               tex_infos['figs_externalize'])
        else:
            res1 = compile_tex(rendered_tex, keywords['tex_folder'], texf_name)
        if res1 != 0:
            res += abs(res1)
            warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    df_list = []
    for df_it in all_mean:
        df_list.append(df_it.mean(axis=0).to_frame().T)
    df_means = pd.concat(df_list, axis=0, keys=keywords['data_separators'], names=['partition'], ignore_index=False)
    df_means.index = [a[0] for a in df_means.index]
    base_name = 'mean_opts_' + short_concat_str(keywords['it_parameters']) + '_for_partitions_' + \
                short_concat_str(keywords['data_separators'])
    b_mean_name = 'data_' + base_name + '.csv'
    fb_mean_name = os.path.join(keywords['res_folder'], b_mean_name)
    with open(fb_mean_name, 'a') as f:
        f.write('# Mean parameter values for mean values of partitions ' +
                '-'.join(map(str, keywords['data_separators'])) + '\n')
        f.write('# Parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
        df_means.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    df_mmean = df_means.mean(axis=0)
    main_parameter_name = keywords['res_par_name']  # 'USAC_opt_refine_min_time'
    # Check if file and parameters exist
    from usac_eval import check_par_file_exists, NoAliasDumper
    ppar_file, res = check_par_file_exists(main_parameter_name, keywords['res_folder'], res)
    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = df_mmean.to_numpy()
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
            alg_w[val] = float(alg_comb_bestl[i])
        yaml.dump({main_parameter_name: {'Algorithm': alg_w}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    if 'comp_res' in keywords and keywords['comp_res'] and isinstance(keywords['comp_res'], list):
        keywords['res'] = res
        res = compare_evaluations(**keywords)
    return res

def compare_evaluations(**keywords):
    if 'comp_res' in keywords and keywords['comp_res'] and isinstance(keywords['comp_res'], list):
        from statistics_and_plot import read_yaml_pars, short_concat_str
        pars_list = dict.fromkeys(keywords['it_parameters'])
        for k in pars_list.keys():
            pars_list[k] = []
        for it in keywords['comp_res']:
            pars = read_yaml_pars(it, keywords['res_folder'])
            if pars and 'Algorithm' in pars:
                for k, v in pars['Algorithm'].items():
                    for k1 in pars_list.keys():
                        if k == k1:
                            pars_list[k].append(v)
                            break
        if len(pars_list[keywords['it_parameters'][0]]) > 1:
            df_pars = pd.DataFrame(pars_list).describe()
            b_mean_name = 'data_mult_evals_stats_opts_' + short_concat_str(keywords['it_parameters']) + '.csv'
            fb_mean_name = os.path.join(keywords['res_folder'], b_mean_name)
            with open(fb_mean_name, 'a') as f:
                f.write(
                    '# Statistic over parameters from multiple evaluations\n')
                f.write('# Parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
                df_pars.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
        else:
            warnings.warn('Too less parameters for calculating statistics found in yaml file', UserWarning)
            keywords['res'] += 1
    return keywords['res']


def get_ml_acc(**keywords):
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
    if 'data_partitions' not in keywords:
        raise ValueError('data_partitions must be provided')
    for it in keywords['data_partitions']:
            if it not in individual_grps:
                raise ValueError(it + ' provided in data_partitions not found in dataframe')
    needed_cols = individual_grps + keywords['eval_columns']
    from statistics_and_plot import short_concat_str, \
        replaceCSVLabels, \
        add_to_glossary, \
        add_to_glossary_eval, \
        get_limits_log_exp, \
        combine_str_for_title, \
        enl_space_title, \
        categorical_sort
    eval_columns_init = deepcopy(keywords['eval_columns'])
    eval_cols1 = [a for a in eval_columns_init if 'mostLikely' not in a]
    eval_cols2 = [a for a in eval_columns_init if 'mostLikely' in a]
    data1 = keywords['data'].drop(eval_cols2, axis=1)
    data_ml = keywords['data'].drop(eval_cols1, axis=1)
    keywords = prepare_io(**keywords)
    res = 0
    if not eval_cols1 or not eval_cols2:
        raise ValueError('Some evaluation columns are missing')
    from usac_eval import combineRt
    b1 = combineRt(data1, False)
    b_ml = combineRt(data_ml, False)
    b_diff = b_ml - b1
    b_diff = b_diff.stack().reset_index()
    b_diff.rename(columns={b_diff.columns[-1]: 'Rt_diff2_ml'}, inplace=True)
    b_diff = b_diff.loc[b_diff['Rt_diff2_ml'].abs() > 1e-6]
    if b_diff.empty:
        f_name = os.path.join(keywords['res_folder'], 'out_compare_is_equal.txt')
        with open(f_name, 'w') as f:
            f.write('Error differences of most likely extrinsics and normal output are equal.' + '\n')
        return 0
    gloss = add_to_glossary_eval(keywords['eval_columns'] + ['Rt_diff', 'Rt_mostLikely_diff'])
    title_name = 'Mean Differences Between ' + \
                 replaceCSVLabels('Rt_diff', True, True, True) + ' and ' + \
                 replaceCSVLabels('Rt_mostLikely_diff', True, True, True) + ' vs ' + \
                 combine_str_for_title(keywords['data_partitions'])
    base_out_name0 = 'mean_diff_default-ml_rt_error_vs_'
    base_out_name = base_out_name0 + short_concat_str(keywords['data_partitions'])
    tex_infos = {'title': title_name,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations of a list of dicts
                 'abbreviations': None}
    for it in keywords['data_partitions']:
        drop_cols = [a for a in individual_grps if a != it]
        df = b_diff.drop(drop_cols, axis=1)
        df['negative'] = df['Rt_diff2_ml'].apply(lambda x: 1 if x < 0 else 0)
        df['positive'] = df['Rt_diff2_ml'].apply(lambda x: 1 if x > 0 else 0)
        df = df.groupby(it).describe()
        base_name = 'stats_rt_default-rt_ml_for_' + it
        b_mean_name = 'data_' + base_name + '.csv'
        fb_mean_name = os.path.join(keywords['res_folder'], b_mean_name)
        with open(fb_mean_name, 'a') as f:
            f.write('# Statistics on differences between most likely and default R&t errors vs ' + it + '\n')
            df.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
        df1 = df.xs('mean', axis=1, level=1, drop_level=True).drop(['negative', 'positive'], axis=1)
        if 'cat_sort' in keywords and keywords['cat_sort'] and \
                isinstance(keywords['cat_sort'], str) and keywords['cat_sort'] == it:
            categorical_sort(df1, it)
        gloss = add_to_glossary(df1.index.values, gloss)
        base_name = base_out_name0 + str(it)
        b_mean_name = 'data_' + base_name + '.csv'
        fb_mean_name = os.path.join(keywords['tdata_folder'], b_mean_name)
        with open(fb_mean_name, 'a') as f:
            f.write('# Mean differences (Rt_diff2_ml) between most likely and default R&t errors vs ' + str(it) + '\n')
            df1.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
        _, use_limits, use_log, exp_value = get_limits_log_exp(df1, True, True, False)
        is_numeric = pd.to_numeric(df1.reset_index()[it], errors='coerce').notnull().all()
        section_name = 'Mean Differences Between ' + \
                       replaceCSVLabels('Rt_diff', True, False, True) + ' and ' + \
                       replaceCSVLabels('Rt_mostLikely_diff', True, False, True) + ' Values vs ' + \
                       replaceCSVLabels(it, False, False, True)
        exp_value = enl_space_title(exp_value, section_name, df1, it,
                                    1, 'ybar')
        reltex_name = os.path.join(keywords['rel_data_path'], b_mean_name)
        tex_infos['sections'].append({'file': reltex_name,
                                      'name': section_name,
                                      # If caption is None, the field name is used
                                      'caption': None,
                                      'fig_type': 'ybar',
                                      'plots': ['Rt_diff2_ml'],
                                      'label_y': replaceCSVLabels('Rt_mostLikely_diff') + '-' +
                                                 replaceCSVLabels('Rt_diff'),
                                      'plot_x': str(it),
                                      'label_x': replaceCSVLabels(str(it)),
                                      'limits': use_limits,
                                      'legend': None,
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      'use_log_y_axis': use_log,
                                      'enlarge_title_space': exp_value,
                                      'use_string_labels': True if not is_numeric else False,
                                      'xaxis_txt_rows': 1,
                                      'enlarge_lbl_dist': None
                                      })

        df2_mean = df.xs('mean', axis=1, level=1, drop_level=True).drop('Rt_diff2_ml', axis=1)
        df2_cnt = df.xs('count', axis=1, level=1, drop_level=True).drop('Rt_diff2_ml', axis=1)
        df2_neg = df2_mean['negative'] * df2_cnt['negative']
        df2_pos = df2_mean['positive'] * df2_cnt['positive']
        df2 = df2_neg / (df2_neg + df2_pos)
        df2.rename('rat_defa_high', inplace=True)
        df2 = df2.to_frame()
        if 'cat_sort' in keywords and keywords['cat_sort'] and \
                isinstance(keywords['cat_sort'], str) and keywords['cat_sort'] == it:
            categorical_sort(df2, it)
        base_name = 'default_err_bigger_as_ml_ratio_vs_' + str(it)
        b_mean_name = 'data_' + base_name + '.csv'
        fb_mean_name = os.path.join(keywords['tdata_folder'], b_mean_name)
        with open(fb_mean_name, 'a') as f:
            f.write('# Ratio (rat_defa_high) of higher default R&t errors compared to most likely R&t errors vs ' +
                    str(it) + '\n')
            df2.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
        _, use_limits, use_log, exp_value = get_limits_log_exp(df2, True, True, False)
        is_numeric = pd.to_numeric(df2.reset_index()[it], errors='coerce').notnull().all()
        section_name = 'Ratio of higher errors ' + \
                       replaceCSVLabels('Rt_diff', False, False, True) + ' compared to ' + \
                       replaceCSVLabels('Rt_mostLikely_diff', False, False, True) + ' vs ' + \
                       replaceCSVLabels(it, False, False, True)
        exp_value = enl_space_title(exp_value, section_name, df2, it,
                                    1, 'ybar')
        reltex_name = os.path.join(keywords['rel_data_path'], b_mean_name)
        tex_infos['sections'].append({'file': reltex_name,
                                      'name': section_name,
                                      # If caption is None, the field name is used
                                      'caption': None,
                                      'fig_type': 'ybar',
                                      'plots': ['rat_defa_high'],
                                      'label_y': 'Ratio',
                                      'plot_x': str(it),
                                      'label_x': replaceCSVLabels(str(it)),
                                      'limits': use_limits,
                                      'legend': None,
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      'use_log_y_axis': use_log,
                                      'enlarge_title_space': exp_value,
                                      'use_string_labels': True if not is_numeric else False,
                                      'xaxis_txt_rows': 1,
                                      'enlarge_lbl_dist': None
                                      })

    tex_infos['abbreviations'] = gloss
    base_out_name1 = 'tex_' + base_out_name
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    texf_name = base_out_name1 + '.tex'
    pdf_name = base_out_name1 + '.pdf'
    res = 0
    if keywords['build_pdf'][0]:
        res = compile_tex(rendered_tex,
                          keywords['tex_folder'],
                          texf_name,
                          tex_infos['make_index'],
                          os.path.join(keywords['pdf_folder'], pdf_name),
                          tex_infos['figs_externalize'])
    else:
        res = compile_tex(rendered_tex, keywords['tex_folder'], texf_name)
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)
    return res


def calc_pose_stable_ratio(**keywords):
    if 'data_separators' not in keywords:
        raise ValueError('data_separators missing!')
    needed_columns = list(dict.fromkeys(keywords['eval_columns'] +
                                        keywords['it_parameters'] +
                                        keywords['x_axis_column'] +
                                        keywords['partitions'] +
                                        ['poseIsStable', 'Nr'] +
                                        keywords['data_separators']))
    df_grp = keywords['data'][needed_columns].groupby(keywords['data_separators'])
    grp_keys = df_grp.groups.keys()
    df_list = []
    for grp in grp_keys:
        tmp = df_grp.get_group(grp).copy(deep=True)
        max_nr = int(tmp['Nr'].max()) + 1
        nr_rows = tmp.shape[0]
        if max_nr < nr_rows:
            warnings.warn('Data is not completely seperated. '
                          'Ratios for stability will be calculated on multiple datasets!', UserWarning)
        nr_stable = float(tmp.loc[(tmp['poseIsStable'] != 0), 'poseIsStable'].shape[0])
        rat = nr_stable / float(nr_rows)
        tmp['tr'] = [rat] * int(nr_rows)
        if 'remove_partitions' in keywords and keywords['remove_partitions']:
            tmp.drop(['poseIsStable', 'Nr'] + keywords['remove_partitions'], axis=1, inplace=True)
        else:
            tmp.drop(['poseIsStable', 'Nr'], axis=1, inplace=True)
        df_list.append(tmp)
    keywords['data'] = pd.concat(df_list, ignore_index=False, axis=0)
    keywords['eval_columns'] += ['tr']
    return keywords


def get_best_stability_pars(**keywords):
    if 'data_separators' not in keywords:
        raise ValueError('data_separators missing!')
    from statistics_and_plot import replaceCSVLabels, \
        glossary_from_list, \
        add_to_glossary, \
        add_to_glossary_eval, \
        strToLower, \
        capitalizeFirstChar, \
        add_val_to_opt_str, \
        combine_str_for_title, \
        calc_limits, \
        split_large_titles, \
        check_if_neg_values, \
        calcNrLegendCols, \
        check_legend_enlarge
    keywords = prepare_io(**keywords)
    df_grp = keywords['data'].reset_index().groupby(keywords['data_separators'])
    grp_keys = df_grp.groups.keys()
    df_list = []
    for grp in grp_keys:
        tmp = df_grp.get_group(grp)
        tmp_mm = tmp.loc[[tmp.loc[:, ('tr', 'mean')].idxmin(), tmp.loc[:, ('tr', 'mean')].idxmax()], :]
        tmp_med = tmp_mm.xs('50%', axis=1, level=1, drop_level=True).copy(deep=True)
        for k, v in zip(keywords['data_separators'], grp):
            tmp_med[k] = [v] * int(tmp_med.shape[0])
        for it in keywords['it_parameters']:
            tmp_med[it] = tmp_mm.loc[:, (it, '')].to_list()
        tmp_med['R_diff2_ml'] = (tmp_med['R_mostLikely_diffAll'] - tmp_med['R_diffAll']).abs().to_list()
        tmp_med['t_diff2_ml'] = (tmp_med['t_mostLikely_angDiff_deg'] - tmp_med['t_angDiff_deg']).abs().to_list()
        tmp_med['Rt_diff2_ml'] = (tmp_med['R_diff2_ml'] + tmp_med['t_diff2_ml']).to_list()
        df_list.append(tmp_med)
    tmp2 = pd.concat(df_list, ignore_index=True, axis=0)
    tmp3 = tmp2.loc[tmp2.groupby(keywords['data_separators'])['Rt_diff2_ml'].idxmin()].copy(deep=True)
    if 'to_int_cols' in keywords and keywords['to_int_cols']:
        for it in keywords['to_int_cols']:
            tmp3.loc[:, it] = tmp3.loc[:, it].round(decimals=0).astype(int)
    it_pars_meta = [a for a in keywords['it_parameters'] if a != keywords['on_2nd_axis']]
    if len(it_pars_meta) > 1:
        tmp3['options_tex'] = [', '.join(['{:.3f}'.format(float(val)) for _, val in row.iteritems()])
                               for _, row in tmp3[it_pars_meta].iterrows()]
    else:
        tmp3['options_tex'] = ['{:.3f}'.format(float(val))
                               for _, val in tmp3[it_pars_meta[0]].iteritems()]
    is_numeric = pd.to_numeric(tmp3[keywords['data_separators'][1]], errors='coerce').notnull().all()
    gloss = glossary_from_list(tmp3[keywords['data_separators'][0]].to_list())
    gloss = add_to_glossary(tmp3[keywords['data_separators'][1]], gloss)
    gloss = add_to_glossary_eval(keywords['eval_columns'], gloss)

    sub_title_rest_it = combine_str_for_title(it_pars_meta)
    title1 = replaceCSVLabels('tr', True, True, True)
    title2 = ' Corresponding to Smallest Combined Median R \\& t Error Differences Between Default ' \
             'and Most Likely Pose Errors Based on Lowest and Highest Mean ' + \
             replaceCSVLabels('tr', False, True, True) + ' Values and Resulting Parameter '
    title3 = replaceCSVLabels(keywords['on_2nd_axis'], False, True, True)
    title = title1 + title2 + title3
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true, non-numeric entries can be provided for the x-axis
                 'nonnumeric_x': not is_numeric,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': gloss
                 }
    base_name = 'stable_pose_ratio_and_' + keywords['on_2nd_axis'] + '_vs_' + keywords['data_separators'][1]
    tmp3['dont_care'] = [0] * int(tmp3.shape[0])
    tmp3_grp = tmp3.groupby(keywords['data_separators'][0])
    grp_keys = tmp3_grp.groups.keys()
    for grp in grp_keys:
        tmp4 = tmp3_grp.get_group(grp)
        base_name1 = base_name + '_prop_' + keywords['data_separators'][0] + '-' + str(grp).replace('.', 'd')
        b_mean_name = 'data_' + base_name1 + '.csv'
        fb_mean_name = os.path.join(keywords['tdata_folder'], b_mean_name)
        section_name = capitalizeFirstChar(strToLower(title)) + ' for property ' + \
                       add_val_to_opt_str(replaceCSVLabels(keywords['data_separators'][0], False, False, True), grp)
        with open(fb_mean_name, 'a') as f:
            f.write('# ' + section_name + '\n')
            f.write('# Parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
            tmp4.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
        caption = capitalizeFirstChar(strToLower(title1)) + ' (bottom axis) ' + strToLower(title2) + ' (top axis) ' + \
                  strToLower(title3) + ' for property ' + \
                  add_val_to_opt_str(replaceCSVLabels(keywords['data_separators'][0], False, False, True), grp) + \
                  '. Values on top of bars correspond to parameters ' + sub_title_rest_it + '.'
        _, _, use_limits_l = calc_limits(tmp4, False, True, None, 'tr')
        is_neg_l = check_if_neg_values(tmp4, 'tr', False, use_limits_l)
        _, _, use_limits_r = calc_limits(tmp4, False, True, None, keywords['on_2nd_axis'])
        is_neg_r = check_if_neg_values(tmp4, keywords['on_2nd_axis'], False, use_limits_r)
        enlarge_lbl_dist = check_legend_enlarge(tmp4, keywords['data_separators'][1], 3, 'xbar')
        section_name = split_large_titles(section_name, 80)
        tex_infos['sections'].append({'file': os.path.join(keywords['rel_data_path'], b_mean_name),
                                      # Name of the whole section
                                      'name': section_name.replace('\\\\', ' '),
                                      # Title of the figure
                                      'title': None,
                                      'title_rows': section_name.count('\\\\'),
                                      'fig_type': 'xbar',
                                      # Column name for charts based on the left y-axis
                                      'plots_l': ['tr'],
                                      # Label of the left y-axis.
                                      'label_y_l': replaceCSVLabels('tr'),
                                      # Column name for charts based on the right y-axis
                                      'plots_r': [keywords['on_2nd_axis']],
                                      # Label of the right y-axis.
                                      'label_y_r': replaceCSVLabels(keywords['on_2nd_axis']),
                                      # Label of the x-axis.
                                      'label_x': replaceCSVLabels(keywords['data_separators'][1]),
                                      # Column name of the x-axis.
                                      'plot_x': keywords['data_separators'][1],
                                      # Enables/disables printing meta information at each data point
                                      'print_meta_l': False,
                                      # Meta information printed at each data point. Must contain same number of
                                      # column names as plots_l
                                      'plot_meta_l': None,
                                      # Are most values of plots at left axis negative?
                                      'is_neg_l': is_neg_l,
                                      # Is more space needed for plotting the meta information?
                                      'large_meta_space_needed_l': False,
                                      # Rotation angle for printed meta information (only 0, 45, and 90 degrees
                                      # are supported)
                                      'rotate_meta_l': 0,
                                      # Enables/disables printing meta information at each data point
                                      'print_meta_r': True,
                                      # Meta information printed at each data point. Must contain same number of
                                      # column names as plots_l
                                      'plot_meta_r': ['options_tex'],
                                      # Are most values of plots at left axis negative?
                                      'is_neg_r': is_neg_r,
                                      # Is more space needed for plotting the meta information?
                                      'large_meta_space_needed_r': True,
                                      # Rotation angle for printed meta information (only 0, 45, and 90 degrees
                                      # are supported)
                                      'rotate_meta_r': 0,
                                      # Maximum and/or minimum y value/s on the left y-axis
                                      'limits_l': use_limits_l,
                                      # Legend entries for the charts belonging to the left y-axis
                                      'legend_l': [replaceCSVLabels('tr')],
                                      # Maximum and/or minimum y value/s on the right y-axis
                                      'limits_r': use_limits_r,
                                      # Legend entries for the charts belonging to the right y-axis
                                      'legend_r': [replaceCSVLabels(keywords['on_2nd_axis'])],
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      'xaxis_txt_rows': 1,
                                      'caption': caption,
                                      'enlarge_lbl_dist': enlarge_lbl_dist
                                      })
        section_name = 'Smallest median R \\& t error differences between default and ' \
                       'most likely pose errors based on lowest and highest mean ' + \
                       replaceCSVLabels('tr', True, True, True) + ' Values and Resulting Parameter ' + \
                       strToLower(title3) + ' for property ' + \
                       add_val_to_opt_str(replaceCSVLabels(keywords['data_separators'][0], False, False, True), grp)
        caption = 'Smallest median R \\& t error differences between default and most ' \
                  'likely pose errors (bottom axis) ' + strToLower(title2) + ' (top axis) ' + \
                  strToLower(title3) + ' for property ' + \
                  add_val_to_opt_str(replaceCSVLabels(keywords['data_separators'][0], False, False, True), grp) + \
                  '. Values on top of bars correspond to parameters ' + sub_title_rest_it + '.'
        _, _, use_limits_l = calc_limits(tmp4, False, True, None, ['R_diff2_ml', 't_diff2_ml'])
        is_neg_l = check_if_neg_values(tmp4, ['R_diff2_ml', 't_diff2_ml'], False, use_limits_l)
        enlarge_lbl_dist = check_legend_enlarge(tmp4, keywords['data_separators'][1], 4, 'xbar')
        section_name = split_large_titles(section_name, 80)
        tex_infos['sections'].append({'file': os.path.join(keywords['rel_data_path'], b_mean_name),
                                      # Name of the whole section
                                      'name': section_name.replace('\\\\', ' '),
                                      # Title of the figure
                                      'title': None,
                                      'title_rows': section_name.count('\\\\'),
                                      'fig_type': 'xbar',
                                      # Column name for charts based on the left y-axis
                                      'plots_l': ['R_diff2_ml', 't_diff2_ml'],
                                      # Label of the left y-axis.
                                      'label_y_l': 'error difference',
                                      # Column name for charts based on the right y-axis
                                      'plots_r': [keywords['on_2nd_axis'], 'dont_care'],
                                      # Label of the right y-axis.
                                      'label_y_r': replaceCSVLabels(keywords['on_2nd_axis']),
                                      # Label of the x-axis.
                                      'label_x': replaceCSVLabels(keywords['data_separators'][1]),
                                      # Column name of the x-axis.
                                      'plot_x': keywords['data_separators'][1],
                                      # Enables/disables printing meta information at each data point
                                      'print_meta_l': False,
                                      # Meta information printed at each data point. Must contain same number of
                                      # column names as plots_l
                                      'plot_meta_l': None,
                                      # Are most values of plots at left axis negative?
                                      'is_neg_l': is_neg_l,
                                      # Is more space needed for plotting the meta information?
                                      'large_meta_space_needed_l': False,
                                      # Rotation angle for printed meta information (only 0, 45, and 90 degrees
                                      # are supported)
                                      'rotate_meta_l': 0,
                                      # Enables/disables printing meta information at each data point
                                      'print_meta_r': True,
                                      # Meta information printed at each data point. Must contain same number of
                                      # column names as plots_l
                                      'plot_meta_r': ['options_tex'],
                                      # Are most values of plots at left axis negative?
                                      'is_neg_r': is_neg_r,
                                      # Is more space needed for plotting the meta information?
                                      'large_meta_space_needed_r': True,
                                      # Rotation angle for printed meta information (only 0, 45, and 90 degrees
                                      # are supported)
                                      'rotate_meta_r': 0,
                                      # Maximum and/or minimum y value/s on the left y-axis
                                      'limits_l': use_limits_l,
                                      # Legend entries for the charts belonging to the left y-axis
                                      'legend_l': ['R error difference', '$\\bm{t}$ error difference'],
                                      # Maximum and/or minimum y value/s on the right y-axis
                                      'limits_r': use_limits_r,
                                      # Legend entries for the charts belonging to the right y-axis
                                      'legend_r': [replaceCSVLabels(keywords['on_2nd_axis'])],
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      'xaxis_txt_rows': 1,
                                      'caption': caption,
                                      'enlarge_lbl_dist': enlarge_lbl_dist
                                      })
        tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
        section_name = 'Smallest combined median R \\& t error differences between default and ' \
                       'most likely pose errors based on lowest and highest mean ' + \
                       replaceCSVLabels('tr', True, True, True) + ' Values and Resulting Parameter ' + \
                       strToLower(title3) + ' for property ' + \
                       add_val_to_opt_str(replaceCSVLabels(keywords['data_separators'][0], False, False, True), grp)
        caption = 'Smallest combined median R \\& t error differences between default and most ' \
                  'likely pose errors (bottom axis) ' + strToLower(title2) + ' (top axis) ' + \
                  strToLower(title3) + ' for property ' + \
                  add_val_to_opt_str(replaceCSVLabels(keywords['data_separators'][0], False, False, True), grp) + \
                  '. Values on top of bars correspond to parameters ' + sub_title_rest_it + '.'
        _, _, use_limits_l = calc_limits(tmp4, False, True, None, 'Rt_diff2_ml')
        is_neg_l = check_if_neg_values(tmp4, 'Rt_diff2_ml', False, use_limits_l)
        enlarge_lbl_dist = check_legend_enlarge(tmp4, keywords['data_separators'][1], 3, 'xbar')
        section_name = split_large_titles(section_name, 80)
        tex_infos['sections'].append({'file': os.path.join(keywords['rel_data_path'], b_mean_name),
                                      # Name of the whole section
                                      'name': section_name.replace('\\\\', ' '),
                                      # Title of the figure
                                      'title': None,
                                      'title_rows': section_name.count('\\\\'),
                                      'fig_type': 'xbar',
                                      # Column name for charts based on the left y-axis
                                      'plots_l': ['Rt_diff2_ml'],
                                      # Label of the left y-axis.
                                      'label_y_l': 'error difference',
                                      # Column name for charts based on the right y-axis
                                      'plots_r': [keywords['on_2nd_axis']],
                                      # Label of the right y-axis.
                                      'label_y_r': replaceCSVLabels(keywords['on_2nd_axis']),
                                      # Label of the x-axis.
                                      'label_x': replaceCSVLabels(keywords['data_separators'][1]),
                                      # Column name of the x-axis.
                                      'plot_x': keywords['data_separators'][1],
                                      # Enables/disables printing meta information at each data point
                                      'print_meta_l': False,
                                      # Meta information printed at each data point. Must contain same number of
                                      # column names as plots_l
                                      'plot_meta_l': None,
                                      # Are most values of plots at left axis negative?
                                      'is_neg_l': is_neg_l,
                                      # Is more space needed for plotting the meta information?
                                      'large_meta_space_needed_l': False,
                                      # Rotation angle for printed meta information (only 0, 45, and 90 degrees
                                      # are supported)
                                      'rotate_meta_l': 0,
                                      # Enables/disables printing meta information at each data point
                                      'print_meta_r': True,
                                      # Meta information printed at each data point. Must contain same number of
                                      # column names as plots_l
                                      'plot_meta_r': ['options_tex'],
                                      # Are most values of plots at left axis negative?
                                      'is_neg_r': is_neg_r,
                                      # Is more space needed for plotting the meta information?
                                      'large_meta_space_needed_r': True,
                                      # Rotation angle for printed meta information (only 0, 45, and 90 degrees
                                      # are supported)
                                      'rotate_meta_r': 0,
                                      # Maximum and/or minimum y value/s on the left y-axis
                                      'limits_l': use_limits_l,
                                      # Legend entries for the charts belonging to the left y-axis
                                      'legend_l': ['error difference'],
                                      # Maximum and/or minimum y value/s on the right y-axis
                                      'limits_r': use_limits_r,
                                      # Legend entries for the charts belonging to the right y-axis
                                      'legend_r': [replaceCSVLabels(keywords['on_2nd_axis'])],
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      'xaxis_txt_rows': 1,
                                      'caption': caption,
                                      'enlarge_lbl_dist': enlarge_lbl_dist
                                      })
    template = ji_env.get_template('usac-testing_2D_plots_2y_axis.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   nonnumeric_x=tex_infos['nonnumeric_x'],
                                   sections=tex_infos['sections'],
                                   fill_bar=True,
                                   abbreviations=tex_infos['abbreviations'])
    texf_name = base_name + '.tex'
    pdf_name = base_name + '.pdf'
    from statistics_and_plot import compile_tex
    if keywords['build_pdf'][0]:
        res = compile_tex(rendered_tex,
                          keywords['tex_folder'],
                          texf_name,
                          tex_infos['make_index'],
                          os.path.join(keywords['pdf_folder'], pdf_name),
                          tex_infos['figs_externalize'])
    else:
        res = compile_tex(rendered_tex, keywords['tex_folder'], texf_name)
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    mean_pars = tmp3[keywords['it_parameters']].mean(axis=0)
    main_parameter_name = keywords['res_par_name']  # 'USAC_opt_refine_min_time'
    # Check if file and parameters exist
    from usac_eval import check_par_file_exists, NoAliasDumper
    ppar_file, res = check_par_file_exists(main_parameter_name, keywords['res_folder'], res)
    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = mean_pars.to_numpy()
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
            alg_w[val] = float(alg_comb_bestl[i])
        yaml.dump({main_parameter_name: {'Algorithm': alg_w}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    return res