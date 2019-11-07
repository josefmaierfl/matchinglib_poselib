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
from pandas.api.types import is_numeric_dtype
from usac_eval import ji_env, get_time_fixed_kp, insert_opt_lbreak, prepare_io
import inspect

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
           'eval_cols_lname': [replaceCSVLabels(a, False, False, True) for a in vars['eval_columns']],
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
    if 'data_separators' not in vars:
        raise ValueError('data_separators are necessary.')
    if ('lname_pre' in vars) != ('eval_pre' in vars):
        raise ValueError('Either both or none of \'lname_pre\' and \'eval_pre\' have to be provided')
    if 'partitions' in vars:
        for key in vars['partitions']:
            if key not in vars['data_separators']:
                raise ValueError('All partition names must be included in the data separators.')
        if 'xy_axis_columns' in vars:
            if len(vars['data_separators']) != (len(vars['partitions']) + 2):
                raise ValueError('Wrong number of data separators.')
        elif 'x_axis_column' in vars:
            if len(vars['data_separators']) != (len(vars['partitions']) + 1):
                raise ValueError('Wrong number of data separators.')
    if 'eval_on' not in vars:
        raise ValueError('Information (column name) for which evaluation is performed must be provided.')
    if 'diff_by' not in vars:
        raise ValueError('No information provided on which consecutive data basis the diff should be performed')

    if 'keepEval' in vars and vars['keepEval']:
        for i in vars['keepEval']:
            if i not in vars['eval_columns']:
                raise ValueError('Label ' + i + ' not found in \'eval_columns\'')

    from statistics_and_plot import use_log_axis
    if 'additional_data' in vars and vars['additional_data']:
        needed_cols = list(dict.fromkeys(vars['eval_columns'] +
                                         vars['it_parameters'] +
                                         vars['data_separators'] +
                                         vars['additional_data']))
        # additional_data_types = vars['data'][vars['additional_data']].dtypes
    else:
        needed_cols = list(dict.fromkeys(vars['eval_columns'] +
                                         vars['it_parameters'] +
                                         vars['data_separators']))
    df = vars['data'][needed_cols]
    grpd_cols = [a for a in vars['data_separators'] if a != vars['diff_by']] + vars['it_parameters']
    df_grp = df.groupby(grpd_cols)
    grp_keys = df_grp.groups.keys()
    data_list = []
    if 'eval_pre' in vars:
        eval_columns_diff = [vars['eval_pre'] + a + '_diff' for a in vars['eval_columns']]
    else:
        eval_columns_diff = [a + '_diff' for a in vars['eval_columns']]
    if 'keepEval' in vars:
        if 'eval_pre' in vars:
            eval_columns_diff1 = eval_columns_diff + [vars['eval_pre'] + a for a in vars['keepEval']]
        else:
            eval_columns_diff1 = eval_columns_diff + vars['keepEval']
    else:
        eval_columns_diff1 = eval_columns_diff

    eval_log = {}
    eval_log_glob = {}
    eval_cols_log_scaling = []
    for evalv in eval_columns_diff1:
        eval_log[evalv] = []
        eval_log_glob[evalv] = []
    for grp in grp_keys:
        tmp = df_grp.get_group(grp)
        data_types = {vars['diff_by']: tmp[vars['diff_by']].dtype}
        data_types.update([(a, tmp[vars['eval_columns'][i]].dtype) for i, a in enumerate(eval_columns_diff)])
        if 'keepEval' in vars and vars['keepEval']:
            for i1 in vars['keepEval']:
                if 'eval_pre' in vars:
                    data_types[vars['eval_pre'] + i1] = tmp[i1].dtype
                else:
                    data_types[i1] = tmp[i1].dtype
        if 'additional_data' in vars and vars['additional_data']:
            for i1 in vars['additional_data']:
                data_types[i1] = tmp[i1].dtype
        data_types[vars['diff_by']] = tmp[vars['diff_by']].dtype
        add_cols = [a for a in grpd_cols if a != vars['diff_by']]
        data_types.update([(a, tmp[a].dtype) for a in add_cols])
        tmp.set_index(vars['diff_by'], inplace=True)
        row_iterator = tmp.iterrows()
        i_last, last = next(row_iterator)
        tmp1 = []
        for i, row in row_iterator:
            if i < i_last:
                warnings.warn('Groupings for calculating frame to frame diffs are not unique! '
                              'Skipping data.', UserWarning)
                break
            tmp1.append(row[vars['eval_columns']] - last[vars['eval_columns']])
            tmp1[-1].index = eval_columns_diff
            if 'keepEval' in vars and vars['keepEval']:
                for i1 in vars['keepEval']:
                    if 'eval_pre' in vars:
                        tmp1[-1][vars['eval_pre'] + i1] = row[i1]
                    else:
                        tmp1[-1][i1] = row[i1]
            if 'additional_data' in vars and vars['additional_data']:
                for i1 in vars['additional_data']:
                    tmp1[-1][i1] = row[i1]
            tmp1[-1][vars['diff_by']] = i
            tmp1[-1] = tmp1[-1].append(row[add_cols])
            last = row
            i_last = i
        data_list.append(pd.concat(tmp1, axis=1).T)
        data_list[-1] = data_list[-1].astype(data_types, copy=False)

        for evalv in eval_columns_diff1:
            ev_min = data_list[-1][evalv].min()
            eval_log_glob[evalv].append(True if ev_min > 0 else False)
            eval_log[evalv].append(use_log_axis(ev_min, data_list[-1][evalv].max()))
    data_new = pd.concat(data_list, ignore_index=True)
    for evalv in eval_columns_diff1:
        if any(eval_log[evalv]) and all(eval_log_glob):
            eval_cols_log_scaling.append(True)
        else:
            eval_cols_log_scaling.append(False)

    # data_new.columns = [a + '_diff' if a in vars['eval_columns'] else a for a in data_new.columns]
    if 'eval_pre' in vars:
        units = [(vars['eval_pre'] + a[0] + '_diff', a[1]) for a in vars['units']]
    else:
        units = [(a[0] + '_diff', a[1]) for a in vars['units']]
    if 'keepEval' in vars and vars['keepEval']:
        units1 = [a for a in vars['units'] if a[0] in vars['keepEval']]
        if 'eval_pre' in vars:
            units1 = [(vars['eval_pre'] + a[0], a[1]) for a in units1]
        units += units1

    from statistics_and_plot import replaceCSVLabels
    if 'lname_pre' in vars:
        eval_cols_lname = [vars['lname_pre'] + ' ' +
                           replaceCSVLabels(a.replace(vars['eval_pre'], ''), False, False, True)
                           for a in eval_columns_diff1]
    else:
        eval_cols_lname = [replaceCSVLabels(a, False, False, True) for a in eval_columns_diff1]

    ret = {'data': data_new,
           'it_parameters': vars['it_parameters'],
           'eval_columns': eval_columns_diff1,
           'eval_cols_lname': eval_cols_lname,
           'eval_cols_log_scaling': eval_cols_log_scaling,
           'units': units,
           'eval_init_input': None}
    if 'partitions' in vars:
        if 'x_axis_column' in vars:
            ret['x_axis_column'] = []
            for key in vars['data_separators']:
                if key not in vars['partitions']:
                    ret['x_axis_column'].append(key)
        elif 'xy_axis_columns' in vars:
            ret['xy_axis_columns'] = []
            for key in vars['data_separators']:
                if key not in vars['partitions']:
                    ret['xy_axis_columns'].append(key)
        ret['partitions'] = vars['partitions']
    else:
        if 'x_axis_column' in vars:
            ret['x_axis_column'] = vars['x_axis_column']
        elif 'xy_axis_columns' in vars:
            ret['xy_axis_columns'] = vars['xy_axis_columns']

    return ret


def get_mean_data_parts(df, nr_parts, key_name='Rt_diff2'):
    img_min = df['Nr'].min()
    img_max = df['Nr'].max()
    ir = img_max - img_min
    if ir <= 0:
        return {}, False
    elif ir <= nr_parts:
        nr_parts = ir + 1
    ir_part = round(ir / float(nr_parts) + 1e-6, 0)
    if isinstance(img_max, int):
        parts = [[img_min + int(a * ir_part), img_min + int((a + 1) * ir_part)] for a in range(0, nr_parts)]
    elif isinstance(img_max, float):
        parts = [[img_min + float(a * ir_part), img_min + float((a + 1) * ir_part)] for a in range(0, nr_parts)]
    else:
        parts = [np.array([img_min + a * ir_part, img_min + (a + 1) * ir_part]).astype(img_max.dtype)
                 for a in range(0, nr_parts)]
    parts[-1][1] = img_max + 1
    if parts[-1][1] - parts[-1][0] <= 0:
        parts.pop()
    data_parts = []
    mean_dd = []
    # print('Limit:', len(inspect.stack()), 'Min:', img_min, 'Max:',
    #       img_max, 'ir_part:', ir_part, 'nr_parts:', nr_parts, 'parts:', parts)
    for sl in parts:
        if sl[1] - sl[0] <= 1:
            tmp = df.set_index('Nr')
            data_parts.append(tmp.loc[[sl[0]],:])
            data_parts[-1].reset_index(inplace=True)
            mean_dd.append(data_parts[-1][key_name].values[0])
        else:
            data_parts.append(df.loc[(df['Nr'] >= sl[0]) & (df['Nr'] < sl[1])])
            # A negative value indicates a decreasing error value and a positive number an increasing error
            # if data_parts[-1].shape[0] < 3:
            #     print('Smaller')
            # elif data_parts[-1].isnull().values.any():
            #     print('nan found')
            mean_dd.append(data_parts[-1][key_name].mean())
    data = {'data_parts': data_parts, 'mean_dd': mean_dd}
    return data, True


def get_converge_img(df, nr_parts, th_diff2=0.33, th_diff3=0.02, key_name='Rt_diff2'):
    data, succ = get_mean_data_parts(df, nr_parts, key_name)
    if not succ:
        return df, False
    data_parts = data['data_parts']
    # A negative value indicates a decreasing error value and a positive number an increasing error
    mean_dd = data['mean_dd']

    error_gets_smaller = [False] * len(data_parts)
    nr_parts1 = nr_parts
    while not any(error_gets_smaller) and nr_parts1 < 10:
        for i, val in enumerate(mean_dd):
            if val < 0:
                error_gets_smaller[i] = True
        if not any(error_gets_smaller) and nr_parts1 < 10:
            nr_parts1 += 1
            data, succ = get_mean_data_parts(df, nr_parts1, key_name)
            if not succ:
                return df, False
            data_parts = data['data_parts']
            mean_dd = data['mean_dd']
            error_gets_smaller = [False] * len(data_parts)
        else:
            break
    if not any(error_gets_smaller):
        df1, succ = get_converge_img(data_parts[0], nr_parts, th_diff2, th_diff3)
        if not succ:
            return df1, False
        else:
            data, succ = get_mean_data_parts(df1, nr_parts, key_name)
            if not succ:
                return df1, False
            data_parts = data['data_parts']
            mean_dd = data['mean_dd']
            error_gets_smaller = [False] * len(data_parts)
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
        return get_converge_img(data_parts[sel_parts[0]], nr_parts, th_diff2, th_diff3, key_name)
    elif last != 0:
        return get_converge_img(data_parts[min(last, l1)], nr_parts, th_diff2, th_diff3, key_name)
    else:
        return get_converge_img(data_parts[0], nr_parts, th_diff2, th_diff3, key_name)


def eval_corr_pool_converge(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    if 'eval_columns' not in keywords:
        raise ValueError('Missing parameter eval_columns')
    if 'partition_x_axis' not in keywords:
        raise ValueError('Missing parameter eval_columns')
    if keywords['partition_x_axis'] not in keywords['partitions']:
        raise ValueError('Partition ' + keywords['partition_x_axis'] + ' not found in partitions')
    needed_evals = ['poolSize', 'poolSize_diff', 'R_diffAll_diff', 't_angDiff_deg_diff', 'R_diffAll', 't_angDiff_deg']
    if not all([a in keywords['eval_columns'] for a in needed_evals]):
        raise ValueError('Some specific entries within parameter eval_columns is missing')

    keywords = prepare_io(**keywords)
    from statistics_and_plot import replaceCSVLabels, \
        glossary_from_list, \
        add_to_glossary, \
        add_to_glossary_eval, \
        split_large_titles, \
        strToLower, \
        tex_string_coding_style, \
        calcNrLegendCols, \
        capitalizeFirstChar, \
        findUnit, \
        compile_tex, \
        get_limits_log_exp, \
        check_legend_enlarge, \
        enl_space_title
    partition_title = ''
    nr_partitions = len(keywords['partitions'])
    for i, val in enumerate(keywords['partitions']):
        partition_title += replaceCSVLabels(val, True, True, True)
        if nr_partitions <= 2:
            if i < nr_partitions - 1:
                partition_title += ' and '
        else:
            if i < nr_partitions - 2:
                partition_title += ', '
            elif i < nr_partitions - 1:
                partition_title += ', and '
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

    tmp, keywords = combine_rt_diff2(tmp, keywords)
    print_evals = ['Rt_diff2'] + needed_evals
    print_evals1 = [a for a in print_evals if a != 'poolSize']

    grpd_cols = keywords['partitions'] + \
                [a for a in keywords['xy_axis_columns'] if a != 'Nr'] + \
                keywords['it_parameters']
    df_grp = tmp.groupby(grpd_cols)
    grp_keys = df_grp.groups.keys()
    data_list = []
    # mult = 5
    # while mult > 1:
    #     try:
    #         sys.setrecursionlimit(mult * sys.getrecursionlimit())
    #         break
    #     except:
    #         mult -= 1
    for grp in grp_keys:
        tmp1 = df_grp.get_group(grp)
        tmp2, succ = get_converge_img(tmp1, 3, 0.33, 0.05)
        data_list.append(tmp2)
    data_new = pd.concat(data_list, ignore_index=True)

    tex_infos = {'title': 'Correspondence Pool Sizes \\& Error Values for Converging Differences from Frame to ' +
                          'Frame of R \\& t Errors '
                          ' for Parameters ' + keywords['sub_title_it_pars'] +
                          ' and Properties ' + partition_title,
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
                 'abbreviations': None
                 }
    df_grp = data_new.groupby(keywords['partitions'])
    grp_keys = df_grp.groups.keys()
    gloss_not_calced = True
    t_main_name = 'converge_poolSizes_with_inlrat'
    if len(keywords['it_parameters']) > 1:
        itpars_name = '-'.join(keywords['it_parameters'])
    else:
        itpars_name = keywords['it_parameters'][0]
    for grp in grp_keys:
        tmp1 = df_grp.get_group(grp)
        tmp1 = tmp1.drop(keywords['partitions'] + ['Nr'], axis=1)
        if gloss_not_calced:
            eval_names = tmp1.columns.values
        tmp1.set_index(keywords['it_parameters'] + [a for a in keywords['xy_axis_columns'] if a != 'Nr'], inplace=True)
        tmp1 = tmp1.unstack(level=-1)
        if len(keywords['it_parameters']) > 1:
            if gloss_not_calced:
                gloss = glossary_from_list([str(b) for a in tmp1.index for b in a])
                gloss = add_to_glossary_eval([a for a in tmp1.columns], gloss)
                # tex_infos['abbreviations'] = gloss
                gloss_not_calced = False
            it_idxs = ['-'.join(map(str, a)) for a in tmp1.index]
            tmp1.index = it_idxs
        else:
            it_idxs = [str(a) for a in tmp1.index]
            if gloss_not_calced:
                gloss = glossary_from_list(it_idxs)
                gloss = add_to_glossary_eval(eval_names, gloss)
                # tex_infos['abbreviations'] = gloss
                gloss_not_calced = False
        gloss = add_to_glossary(list(grp), gloss)
        comb_cols = ['-'.join(map(str, a)) for a in tmp1.columns]
        comb_cols1 = ['/'.join(map(str, a)) for a in tmp1.columns]
        comb_cols = [a.replace('.', 'd') for a in comb_cols]
        tmp1.columns = comb_cols

        tmp1['tex_it_pars'] = insert_opt_lbreak(it_idxs)
        max_txt_rows = 1
        for idx, val in tmp1['tex_it_pars'].iteritems():
            txt_rows = str(val).count('\\\\') + 1
            if txt_rows > max_txt_rows:
                max_txt_rows = txt_rows

        t_main_name1 = t_main_name + '_part_' + \
                       '_'.join([keywords['partitions'][i][:min(4, len(keywords['partitions'][i]))] + '-' +
                                 a[:min(3, len(a))] for i, a in enumerate(map(str, grp))]) + '_for_opts_' + itpars_name
        t_main_name1 = t_main_name1.replace('.', 'd')
        t_mean_name = 'data_' + t_main_name1 + '.csv'
        ft_mean_name = os.path.join(keywords['tdata_folder'], t_mean_name)
        with open(ft_mean_name, 'a') as f:
            f.write('# Correspondence pool sizes for converging differences from frame to '
                    'frame of R & t errors and their inlier ratio '
                    'for data partition ' + '_'.join([keywords['partitions'][i] + '-' +
                                                      a for i, a in enumerate(map(str, grp))]) + '\n')
            f.write('# Different parameters: ' + itpars_name + '\n')
            tmp1.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        for ev in print_evals:
            surpr_ev = [a for a in print_evals if ev in a and ev != a]
            use_cols = []
            for a in comb_cols:
                if ev in a:
                    take_it = True
                    for b in surpr_ev:
                        if b in a:
                            take_it = False
                            break
                    if take_it:
                        use_cols.append(a)
            if not use_cols:
                continue
            side_eval = replaceCSVLabels([a for a in keywords['xy_axis_columns'] if a != 'Nr'][0])
            close_equ = False
            if '$' == side_eval[-1]:
                close_equ = True
                side_eval = side_eval[:-1] + '='
            elif '}' == side_eval[-1]:
                side_eval += '='
            else:
                side_eval += ' equal to '
            use_cols1 = []
            for a in comb_cols1:
                if ev in a:
                    take_it = True
                    for b in surpr_ev:
                        if b in a:
                            take_it = False
                            break
                    if take_it:
                        use_cols1.append(a)
            use_cols1 = [round(float(b), 6) for a in use_cols1 for b in a.split('/') if ev not in b]
            use_cols1 = [side_eval + str(a) + '$' if close_equ else side_eval + str(a) for a in use_cols1]

            _, use_limits, use_log, exp_value = get_limits_log_exp(tmp1, True, True, False, None, use_cols)
            # is_numeric = pd.to_numeric(tmp.reset_index()[keywords['xy_axis_columns'][0]], errors='coerce').notnull().all()
            reltex_name = os.path.join(keywords['rel_data_path'], t_mean_name)
            fig_name = capitalizeFirstChar(replaceCSVLabels(ev, True, False, True)) + \
                       ' for converging differences from frame to frame of R \\& t errors\\\\for parameters ' + \
                       strToLower(keywords['sub_title_it_pars']) + ' and properties '
            for i1, (part, val2) in enumerate(zip(keywords['partitions'], grp)):
                fig_name += replaceCSVLabels(part) + ' equal to ' + str(val2)
                if nr_partitions <= 2:
                    if i1 < nr_partitions - 1:
                        fig_name += ' and '
                else:
                    if i1 < nr_partitions - 2:
                        fig_name += ', '
                    elif i1 < nr_partitions - 1:
                        fig_name += ', and '
            fig_name = split_large_titles(fig_name)
            exp_value = enl_space_title(exp_value, fig_name, tmp1, 'tex_it_pars',
                                        len(use_cols), 'ybar')
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': fig_name,
                                          # If caption is None, the field name is used
                                          'caption': fig_name.replace('\\\\', ' '),
                                          'fig_type': 'ybar',
                                          'plots': use_cols,
                                          'label_y': replaceCSVLabels(ev) + findUnit(str(ev), keywords['units']),
                                          'plot_x': 'tex_it_pars',
                                          'label_x': 'Parameter',
                                          'limits': use_limits,
                                          'legend': use_cols1,
                                          'legend_cols': 1,
                                          'use_marks': False,
                                          'use_log_y_axis': use_log,
                                          'xaxis_txt_rows': max_txt_rows,
                                          'enlarge_lbl_dist': None,
                                          'enlarge_title_space': exp_value,
                                          'use_string_labels': True,
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    tex_infos['abbreviations'] = gloss
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    if keywords['build_pdf'][0]:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex,
                              keywords['tex_folder'],
                              texf_name,
                              tex_infos['make_index'],
                              os.path.join(keywords['pdf_folder'], pdf_name),
                              tex_infos['figs_externalize']))
    else:
        res = abs(compile_tex(rendered_tex, keywords['tex_folder'], texf_name, False))
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    grpd_cols = keywords['partitions'] + keywords['it_parameters']
    df_grp = data_new.groupby(grpd_cols)
    grp_keys = df_grp.groups.keys()
    data_list = []
    for grp in grp_keys:
        tmp1 = df_grp.get_group(grp)
        if tmp1.shape[0] < 4:
            if tmp1.shape[0] > 2:
                poolsize_med = tmp1['poolSize'].median()
                data_list.append(tmp1.loc[tmp1['poolSize'] == poolsize_med])
            else:
                poolsize_mean = mean_with_strings(tmp1)
                data_list.append(poolsize_mean)
        else:
            hist, bin_edges = np.histogram(tmp1['poolSize'].values, bins='auto', density=True)
            idx = np.argmax(hist)
            take_edges = bin_edges[idx:(idx + 2)]
            tmp2 = tmp1.loc[((tmp1['poolSize'] >= take_edges.item(0)) & (tmp1['poolSize'] <= take_edges.item(1)))]
            if tmp2.shape[0] > 2:
                poolsize_med = tmp2['poolSize'].median()
                data_list.append(tmp2.loc[tmp2['poolSize'] == poolsize_med])
            elif tmp2.shape[0] > 1:
                poolsize_mean = mean_with_strings(tmp2)
                data_list.append(poolsize_mean)
            else:
                data_list.append(tmp2)
    data_new2 = pd.concat(data_list, ignore_index=True)
    data_new2.drop(keywords['xy_axis_columns'], axis=1, inplace=True)
    if len(keywords['it_parameters']) > 1:
        data_new2 = data_new2.set_index(keywords['it_parameters']).T
        itpars_cols = ['-'.join(a) for a in data_new2.columns]
        data_new2.columns = itpars_cols
        data_new2.columns.name = itpars_name
        data_new2 = data_new2.T.reset_index()
    else:
        itpars_cols = data_new2[itpars_name].values
    itpars_cols = list(dict.fromkeys(itpars_cols))
    dataseps = [a for a in keywords['partitions'] if a != keywords['partition_x_axis']] + [itpars_name]
    l1 = len(dataseps)
    data_new2.set_index([keywords['partition_x_axis']] + dataseps, inplace=True)
    comb_cols = None
    for i in range(0, l1):
        data_new2 = data_new2.unstack(level=-1)
        comb_cols = ['-'.join(map(str, a)) for a in data_new2.columns]
        data_new2.columns = comb_cols
    t_main_name = 'converge_poolSizes_vs_' + itpars_name + '_' + keywords['partition_x_axis']
    t_mean_name = 'data_' + t_main_name + '.csv'
    ft_mean_name = os.path.join(keywords['tdata_folder'], t_mean_name)
    with open(ft_mean_name, 'a') as f:
        f.write('# Most likely correspondence pool sizes for converging differences from frame to '
                'frame of R & t errors vs data partition ' + keywords['partition_x_axis'] + '\n')
        f.write('# Different parameters: ' + itpars_name + '\n')
        data_new2.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    tex_infos = {'title': 'Most Likely Correspondence Pool Sizes for Converging Differences from Frame to ' +
                          'Frame of R \\& t Errors '
                          ' for Parameters ' + keywords['sub_title_it_pars'] +
                          ' vs Property ' + replaceCSVLabels(keywords['partition_x_axis'], False, True, True),
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
    use_plots = [a for a in comb_cols if 'poolSize' in a and 'poolSize_diff' not in a]
    data_new3 = data_new2[use_plots]
    _, use_limits, use_log, exp_value = get_limits_log_exp(data_new3, True, True)
    is_numeric = pd.to_numeric(data_new3.reset_index()[keywords['partition_x_axis']], errors='coerce').notnull().all()
    enlarge_lbl_dist = check_legend_enlarge(data_new2, keywords['partition_x_axis'], len(use_plots), 'xbar')
    reltex_name = os.path.join(keywords['rel_data_path'], t_mean_name)
    fig_name = 'Most likely correspondence pool sizes for converging differences from ' \
               'frame to frame of R \\& t errors\\\\for parameters ' + \
               strToLower(keywords['sub_title_it_pars']) + ' vs property ' + \
               replaceCSVLabels(keywords['partition_x_axis'], False, False, True)
    fig_name = split_large_titles(fig_name)
    exp_value = enl_space_title(exp_value, fig_name, data_new3, keywords['partition_x_axis'],
                                len(use_plots), 'xbar')
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': fig_name,
                                  # If caption is None, the field name is used
                                  'caption': fig_name.replace('\\\\', ' '),
                                  'fig_type': 'xbar',
                                  'plots': use_plots,
                                  'label_y': replaceCSVLabels('poolSize') + findUnit('poolSize', keywords['units']),
                                  'plot_x': keywords['partition_x_axis'],
                                  'label_x': replaceCSVLabels(keywords['partition_x_axis']),
                                  'limits': use_limits,
                                  'legend': [tex_string_coding_style(a) for a in use_plots],
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  'use_log_y_axis': use_log,
                                  'xaxis_txt_rows': 1,
                                  'enlarge_lbl_dist': enlarge_lbl_dist,
                                  'enlarge_title_space': exp_value,
                                  'use_string_labels': True if not is_numeric else False,
                                  })
    # tex_infos['sections'].append({'file': reltex_name,
    #                               'name': fig_name.replace('\\\\', ' '),
    #                               'title': fig_name,
    #                               'title_rows': fig_name.count('\\\\'),
    #                               'fig_type': 'xbar',
    #                               'plots': use_plots,
    #                               # Label of the value axis. For xbar it labels the x-axis
    #                               'label_y': replaceCSVLabels('poolSize') + findUnit('poolSize', keywords['units']),
    #                               # Label/column name of axis with bars. For xbar it labels the y-axis
    #                               'label_x': replaceCSVLabels(keywords['partition_x_axis']),
    #                               # Column name of axis with bars. For xbar it is the column for the y-axis
    #                               'print_x': keywords['partition_x_axis'],
    #                               # Set print_meta to True if values from column plot_meta should be printed next to each bar
    #                               'print_meta': False,
    #                               'plot_meta': None,
    #                               # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
    #                               'rotate_meta': 0,
    #                               'limits': use_limits,
    #                               # If None, no legend is used, otherwise use a list
    #                               'legend': [tex_string_coding_style(a) for a in use_plots],
    #                               'legend_cols': 1,
    #                               'use_marks': False,
    #                               # The x/y-axis values are given as strings if True
    #                               'use_string_labels': True if not is_numeric else False,
    #                               'use_log_y_axis': use_log,
    #                               'xaxis_txt_rows': 1,
    #                               'enlarge_lbl_dist': enlarge_lbl_dist,
    #                               'enlarge_title_space': exp_value,
    #                               'large_meta_space_needed': False,
    #                               'caption': fig_name.replace('\\\\', ' ')
    #                               })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    if keywords['build_pdf'][1]:
        pdf_name = base_out_name + '.pdf'
        res1 = abs(compile_tex(rendered_tex,
                               keywords['tex_folder'],
                               texf_name,
                               tex_infos['make_index'],
                               os.path.join(keywords['pdf_folder'], pdf_name),
                               tex_infos['figs_externalize']))
    else:
        res1 = abs(compile_tex(rendered_tex, keywords['tex_folder'], texf_name, False))
    if res1 != 0:
        res += res1
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    data_new3 = data_new2.mean(axis=0)
    sel_parts =[[b for b in comb_cols if a in b and ('R_diffAll' in b or 't_angDiff_deg' in b)
                 and not ('R_diffAll_diff' in b or 't_angDiff_deg_diff' in b)] for a in itpars_cols]
    sel_parts2 = [[c for b in sel_parts for c in b if ('R_diffAll' in c or 't_angDiff_deg' in c) and a in c] for a in
                  itpars_cols]
    sel_parts4 = [[[b for b in a if 'R_diffAll' in b], [b for b in a if 't_angDiff_deg' in b]] for a in sel_parts2]
    rem_parts = [[c for c in list(dict.fromkeys(b.split('-'))) if c != 'R_diffAll' and c not in
                  [e1 for d in itpars_cols for e1 in d.split('-')]] for a in sel_parts4 for b in a[0]]
    rem_parts = list(dict.fromkeys([b for a in rem_parts for b in a]))
    sel_parts1 = [b for a in itpars_cols for b in comb_cols if a in b and 'poolSize' in b and 'poolSize_diff' not in b]
    sel_parts3 = [[b for b in sel_parts1 if 'poolSize' in b and a in b] for a in itpars_cols]
    for i, (val1, val2) in enumerate(zip(sel_parts4, sel_parts3)):
        hlp3 = []
        for rp in rem_parts:
            hlp = None
            hlp1 = None
            hlp2 = None
            for rh in val1[0]:
                if rp in rh:
                    take_it = True
                    for rp1 in rh.split('-'):
                        if rp in rp1 and rp1 != rp:
                            take_it = False
                    if not take_it:
                        continue
                    hlp = data_new3[rh]
                    break
            for rh in val1[1]:
                if rp in rh:
                    take_it = True
                    for rp1 in rh.split('-'):
                        if rp in rp1 and rp1 != rp:
                            take_it = False
                    if not take_it:
                        continue
                    hlp1 = data_new3[rh]
                    break
            for rh in val2:
                if rp in rh:
                    take_it = True
                    for rp1 in rh.split('-'):
                        if rp in rp1 and rp1 != rp:
                            take_it = False
                    if not take_it:
                        continue
                    hlp2 = data_new3[rh]
                    break
            if hlp and hlp1 and hlp2:
                hlp3.append((hlp + hlp1) * hlp2)
        data_new3[itpars_cols[i]] = sum(hlp3)
    best_alg = data_new3[itpars_cols].idxmin()
    idx = [a for a in comb_cols if best_alg in a and 'R_diffAll' in a and 'R_diffAll_diff' not in a]
    mean_r_error = float(data_new3[idx].mean())
    idx = [a for a in comb_cols if best_alg in a and 't_angDiff_deg' in a and 't_angDiff_deg_diff' not in a]
    mean_t_error = float(data_new3[idx].mean())
    idx = [a for a in comb_cols if best_alg in a and 'poolSize' in a and 'poolSize_diff' not in a]
    mean_poolSize = int(data_new3[idx].mean())
    main_parameter_name = keywords['res_par_name']  # 'USAC_opt_refine_min_time'
    # Check if file and parameters exist
    from usac_eval import check_par_file_exists, NoAliasDumper
    ppar_file, res = check_par_file_exists(main_parameter_name, keywords['res_folder'], res)
    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = best_alg.split('-')
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithm': alg_w,
                                         'mean_conv_pool_size': mean_poolSize,
                                         'mean_R_error': mean_r_error,
                                         'mean_t_error': mean_t_error}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    df_grp = tmp.groupby(keywords['partitions'])
    grp_keys = df_grp.groups.keys()
    t_main_name = 'double_Rt_diff_vs_poolSize'
    tex_infos = {'title': 'Differences from Frame to ' +
                          'Frame of R \\& t Errors for Parameters ' + keywords['sub_title_it_pars'] +
                          ' in Addition to Properties ' + partition_title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': gloss
                 }
    for grp in grp_keys:
        tmp1 = df_grp.get_group(grp)
        tmp1 = tmp1.drop(keywords['partitions'], axis=1)
        tmp1 = tmp1.groupby(keywords['it_parameters'])
        grp_keys1 = tmp1.groups.keys()
        grp_list = []
        for grp1 in grp_keys1:
            tmp2 = tmp1.get_group(grp1)
            #take mean for same corrPool size
            tmp2 = tmp2.sort_values(by='poolSize')
            row_iterator = tmp2.iterrows()
            i_last, last = next(row_iterator)
            mean_rows = []
            for i, row in row_iterator:
                if last['poolSize'] == row['poolSize']:
                    if mean_rows:
                        if mean_rows[-1][-1] == i_last:
                            mean_rows[-1].append(i)
                        else:
                            mean_rows.append([i_last, i])
                    else:
                        mean_rows.append([i_last, i])
                last = row
                i_last = i
            if mean_rows:
                mean_rows_data = []
                for val in mean_rows:
                    mean_rows_data.append(tmp2.loc[val, :].mean(axis=0))
                    for val1 in keywords['it_parameters']:
                        mean_rows_data[-1][val1] = tmp2.loc[val[0], val1]
                    for val1 in keywords['xy_axis_columns']:
                        mean_rows_data[-1][val1] = tmp2.loc[val[0], val1]
                mean_rows1 = []
                for val in mean_rows:
                    mean_rows1 += val
                tmp2.drop(mean_rows1, axis=0, inplace=True)
                tmp3 = pd.concat(mean_rows_data, axis=1).T
                tmp2 = pd.concat([tmp2, tmp3], ignore_index=True, sort=True)
                tmp2 = tmp2.sort_values(by='poolSize')
            grp_list.append(tmp2)
        tmp1 = pd.concat(grp_list, ignore_index=True, sort=True, axis=0)
        if len(keywords['it_parameters']) > 1:
            tmp1 = tmp1.set_index(keywords['it_parameters']).T
            itpars_cols = ['-'.join(a) for a in tmp1.columns]
            tmp1.columns = itpars_cols
            tmp1.columns.name = itpars_name
            tmp1 = tmp1.T.reset_index()
        else:
            itpars_cols = tmp1[itpars_name].values
        itpars_cols = list(dict.fromkeys(itpars_cols))
        tmp1.set_index(keywords['xy_axis_columns'] + [itpars_name], inplace=True)
        tmp1 = tmp1.unstack(level=-1)
        tmp1.reset_index(inplace=True)
        tmp1.drop(keywords['xy_axis_columns'], axis=1, inplace=True, level=0)
        col_names = ['-'.join(a) for a in tmp1.columns]
        tmp1.columns = col_names
        # surpr_ev = [b for a in print_evals1 for b in print_evals1 if a in b and a != b]
        sections = []
        for ev in print_evals1:
            surpr_ev = [a for a in print_evals1 if ev in a and a != ev]
            sections1 = []
            for a in col_names:
                if ev in a:
                    take_it = True
                    for b in surpr_ev:
                        if b in a:
                            take_it = False
                            break
                    if take_it:
                        sections1.append(a)
            if sections1:
                sections.append(sections1)
        # sections = [[b for b in col_names if a in b] for a in print_evals1]
        # x_axis = [[b for d in itpars_cols if d in c for b in col_names
        #            if d in b and 'poolSize' in b and 'poolSize_diff' not in b] for a in sections
        #           for c in a]
        x_axis = []
        for a in sections:
            hlp = []
            for c in a:
                break_it = False
                for d in itpars_cols:
                    if d in c:
                        for b in col_names:
                            if d in b and 'poolSize' in b and 'poolSize_diff' not in b:
                                hlp.append(b)
                                break_it = True
                                break
                        if break_it:
                            break
            if hlp:
                x_axis.append(hlp)
        if len(sections) != len(x_axis):
            warnings.warn('Unable to extract column names for x- and y-axis. There might be similar names. '
                          'Aborting calculation of next figure.', UserWarning)
            return 1

        it_split = []
        it_split_x = []
        for it in itpars_cols:
            evals1 = []
            it_split_x1 = []
            for i, ev in enumerate(sections):
                for i1, ev1 in enumerate(ev):
                    if it in ev1:
                        evals1.append(ev1)
                        it_split_x1.append(x_axis[i][i1])
            it_split.append(evals1)
            it_split_x.append(list(dict.fromkeys(it_split_x1)))
        cols_list = []
        for it_x, it in zip(it_split_x, it_split):
            tmpi = tmp1.loc[:, it_x + it]
            tmpi = tmpi.sort_values(by=it_x).reset_index(drop=True)
            cols_list.append(tmpi)
        tmp1 = pd.concat(cols_list, axis=1, ignore_index=False)

        t_main_name1 = t_main_name + '_part_' + \
                       '_'.join([keywords['partitions'][i][:min(4, len(keywords['partitions'][i]))] + '-' +
                                 a[:min(3, len(a))] for i, a in enumerate(map(str, grp))]) + '_for_opts_' + itpars_name
        t_main_name1 = t_main_name1.replace('.', 'd')
        t_mean_name = 'data_' + t_main_name1 + '.csv'
        ft_mean_name = os.path.join(keywords['tdata_folder'], t_mean_name)
        with open(ft_mean_name, 'a') as f:
            f.write('# Differences from frame to frame for R & t errors vs correspondence pool sizes '
                    'for data partition ' + '_'.join([keywords['partitions'][i] + '-' +
                                                      a for i, a in enumerate(map(str, grp))]) + '\n')
            f.write('# Different parameters: ' + itpars_name + '\n')
            tmp1.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

        for ev, x, ev_name in zip(sections, x_axis, print_evals1):
            tmp2 = tmp1.loc[:, ev + x]
            _, use_limits, use_log, exp_value = get_limits_log_exp(tmp2, False, False, False, None, ev)
            reltex_name = os.path.join(keywords['rel_data_path'], t_mean_name)

            partition_part = ''
            for i, (val_name, val) in enumerate(zip(keywords['partitions'], grp)):
                partition_part += replaceCSVLabels(val_name, False, False, True)
                if '$' == partition_part[-1]:
                    partition_part = partition_part[:-1] + '=' + str(val) + '$'
                elif '}' == partition_part[-1]:
                    partition_part += '=' + str(val)
                else:
                    partition_part += ' equal to ' + str(val)
                if nr_partitions <= 2:
                    if i < nr_partitions - 1:
                        partition_part += ' and '
                else:
                    if i < nr_partitions - 2:
                        partition_part += ', '
                    elif i < nr_partitions - 1:
                        partition_part += ', and '

            legend_entries = [a for b in ev for a in itpars_cols if a in b]
            fig_name = capitalizeFirstChar(replaceCSVLabels(ev_name, False, False, True)) +  \
                       ' vs correspondence pool sizes\\\\for parameters ' + \
                       strToLower(keywords['sub_title_it_pars']) + ' and properties ' + \
                       partition_part
            fig_name = split_large_titles(fig_name)
            exp_value = enl_space_title(exp_value, fig_name, tmp2, x,
                                        len(ev), 'sharp plot')
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': fig_name,
                                          # If caption is None, the field name is used
                                          'caption': fig_name.replace('\\\\', ' '),
                                          'fig_type': 'sharp plot',
                                          'plots': ev,
                                          'label_y': replaceCSVLabels(ev_name) + findUnit(ev_name,
                                                                                          keywords['units']),
                                          'plot_x': x,
                                          'label_x': replaceCSVLabels('poolSize'),
                                          'limits': use_limits,
                                          'legend': [tex_string_coding_style(a) for a in legend_entries],
                                          'legend_cols': 1,
                                          'use_marks': False,
                                          'use_log_y_axis': use_log,
                                          'enlarge_title_space': exp_value,
                                          'enlarge_lbl_dist': None,
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    template = ji_env.get_template('usac-testing_2D_plots_mult_x_cols.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    if keywords['build_pdf'][2]:
        pdf_name = base_out_name + '.pdf'
        res1 = abs(compile_tex(rendered_tex,
                               keywords['tex_folder'],
                               texf_name,
                               tex_infos['make_index'],
                               os.path.join(keywords['pdf_folder'], pdf_name),
                               tex_infos['figs_externalize']))
    else:
        res1 = abs(compile_tex(rendered_tex, keywords['tex_folder'], texf_name, False))
    if res1 != 0:
        res += res1
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    return res


def combine_rt_diff2(df, keywords):
    if 'eval_pre' in keywords:
        comb_vars = [keywords['eval_pre'] + 'R_diffAll', keywords['eval_pre'] + 't_angDiff_deg']
        comb_diff_vars = [keywords['eval_pre'] + 'R_diffAll_diff', keywords['eval_pre'] + 't_angDiff_deg_diff']
        key_out = keywords['eval_pre'] + 'Rt_diff2'
    else:
        comb_vars = ['R_diffAll', 't_angDiff_deg']
        comb_diff_vars = ['R_diffAll_diff', 't_angDiff_deg_diff']
        key_out = 'Rt_diff2'
    tmp_mm = df[comb_diff_vars]
    tmp3 = (df[comb_vars[0]] * tmp_mm[comb_diff_vars[0]] / df[comb_vars[0]].abs() +
            df[comb_vars[1]] * tmp_mm[comb_diff_vars[1]] / df[comb_vars[1]].abs()) / 2
    min_vals = tmp3.min()
    max_vals = tmp3.max()
    r_vals = max_vals - min_vals
    if np.isclose(r_vals, 0, atol=1e-06):
        df[key_out] = tmp3
    else:
        df[key_out] = tmp3 / r_vals
    if 'units' in keywords:
        keywords['units'].append((key_out, '/\\textdegree'))

    return df, keywords


def mean_with_strings(df):
    return df.agg(mean_str, axis=0).to_frame().T


def mean_str(col):
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.unique()[0] if col.nunique() == 1 else col.iloc[0]


def calc_rt_diff_n_matches(**keywords):
    # Calculate new poolSize values which are near their original value but equal to other poolSize values from other
    # evaluations to be able to use the same poolSize values for visualization (x-axis) of different evaluations
    if 'partitions' in keywords:
        if 'x_axis_column' in keywords:
            needed_columns = keywords['eval_columns'] + keywords['it_parameters'] + \
                             keywords['x_axis_column'] + keywords['partitions']
        elif 'xy_axis_columns' in keywords:
            needed_columns = keywords['eval_columns'] + keywords['it_parameters'] + \
                             keywords['xy_axis_columns'] + keywords['partitions']
        else:
            needed_columns = keywords['eval_columns'] + keywords['it_parameters'] + \
                             keywords['partitions']
    elif 'x_axis_column' in keywords:
        needed_columns = keywords['eval_columns'] + keywords['it_parameters'] + keywords['x_axis_column']
    elif 'xy_axis_columns' in keywords:
        needed_columns = keywords['eval_columns'] + keywords['it_parameters'] + keywords['xy_axis_columns']
    else:
        needed_columns = keywords['eval_columns'] + keywords['it_parameters']
    if 'data_separators' in keywords:
        needed_columns += keywords['data_separators']
        needed_columns = list(dict.fromkeys(needed_columns))
    data = keywords['data'].loc[:, needed_columns]
    if 'partitions' in keywords:
        data = data.groupby(keywords['partitions'])
        grp_keys = data.groups.keys()
        grp_list = []
        for grp in grp_keys:
            tmp = data.get_group(grp)
            bin_edges = np.histogram_bin_edges(tmp['poolSize'].values, bins='auto')
            first = None
            for be in np.nditer(bin_edges):
                if not first:
                    first = be
                else:
                    mean = round((be - first) / 2, 0)
                    tmp.loc[((tmp['poolSize'] > first) & (tmp['poolSize'] <= be)), 'poolSize'] = mean
                    first = be
            grp_list.append(tmp)
        df = pd.concat(grp_list, ignore_index=True)
    else:
        bin_edges = np.histogram_bin_edges(data['poolSize'].values, bins='auto')
        first = None
        for be in np.nditer(bin_edges):
            if not first:
                first = float(be)
            else:
                mean = round(first + (be - first) / 2, 0)
                data.loc[((data['poolSize'] > first) & (data['poolSize'] <= be)), 'poolSize'] = mean
                first = float(be)
        df = data
    ret = {'data': df,
           'eval_columns': keywords['eval_columns'],
           'it_parameters': keywords['it_parameters'],
           'eval_init_input': ['poolSize']}
    if 'partitions' in keywords:
        ret['partitions'] = keywords['partitions']
    if 'data_separators' in keywords:
        ret['data_separators'] = keywords['data_separators']
    if 'units' in keywords:
        ret['units'] = keywords['units']
    if 'eval_on' in keywords:
        ret['eval_on'] = keywords['eval_on']
    if 'diff_by' in keywords:
        ret['diff_by'] = keywords['diff_by']
    if 'keepEval' in keywords:
        ret['keepEval'] = keywords['keepEval']
    if 'x_axis_column' in keywords:
        ret['x_axis_column'] = keywords['x_axis_column']
    elif 'xy_axis_columns' in keywords:
        ret['xy_axis_columns'] = keywords['xy_axis_columns']
    return ret


def calc_diff_stat_rt_diff_n_matches(**keywords):
    vars = calc_rt_diff_n_matches(**keywords)
    accum_all = False
    if 'partitions' in vars:
        for key in vars['partitions']:
            if key not in vars['data_separators']:
                raise ValueError('All partition names must be included in the data separators.')
        if ('x_axis_column' in vars and len(vars['data_separators']) != (len(vars['partitions']) + 1)) or \
                ('xy_axis_columns' in vars and len(vars['data_separators']) != (len(vars['partitions']) + 2)):
            raise ValueError('Wrong number of data separators.')
    elif 'x_axis_column' in vars and 'data_separators' in vars and len(vars['data_separators']) > 1:
        raise ValueError('Only one data separator is allowed.')
    elif 'xy_axis_columns' in vars and len(vars['data_separators']) != 2:
        raise ValueError('Only two data separators are allowed.')
    if 'x_axis_column' in vars:
        x_axis_column = vars['x_axis_column']
        if 'data_separators' not in vars or not vars['data_separators']:
            accum_all = True
    elif 'xy_axis_columns' in vars:
        x_axis_column = vars['xy_axis_columns']
    else:
        raise ValueError('Missing x-axis column names')
    if 'eval_on' not in vars:
        raise ValueError('Information (column name) for which evaluation this is performed must be provided.')
    from statistics_and_plot import replace_stat_names, replace_stat_names_col_tex
    if accum_all:
        needed_cols = vars['eval_columns'] + vars['it_parameters'] + x_axis_column
    else:
        needed_cols = vars['eval_columns'] + vars['it_parameters'] + x_axis_column + vars['data_separators']
    needed_cols = list(dict.fromkeys(needed_cols))
    df = vars['data'][needed_cols]
    if accum_all:
        grpd_cols = vars['it_parameters']
    else:
        grpd_cols = vars['eval_on'] + vars['it_parameters']
    stats = df.groupby(grpd_cols).describe()
    # errvalnames = stats.columns.values  # Includes statistic name and error value names
    # grp_names = stats.index.names  # As used when generating the groups
    algpars = vars
    data_parts = []
    stat_values = list(dict.fromkeys(stats.columns.get_level_values(1).tolist()))
    for it in stat_values:
        if it != 'count':
            tmp = stats.xs(it, axis=1, level=1, drop_level=True).reset_index()
            algpars['data'] = tmp
            algpars['eval_pre'] = replace_stat_names_col_tex(it) + '_'
            algpars['lname_pre'] = replace_stat_names(it)
            data_parts.append(calc_rt_diff2_frame_to_frame(**algpars))
            data_parts[-1]['data'].set_index(vars['it_parameters'] + vars['eval_on'], inplace=True)
    data = pd.concat([a['data'] for a in data_parts], ignore_index=False, axis=1)
    data.reset_index(inplace=True)
    eval_columns = []
    eval_cols_lname = []
    eval_cols_log_scaling = []
    units = []
    for it in data_parts:
        eval_columns += it['eval_columns']
        eval_cols_lname += it['eval_cols_lname']
        eval_cols_log_scaling += it['eval_cols_log_scaling']
        units += it['units']

    stat_values_tex = [replace_stat_names_col_tex(a) + '_' for a in stat_values if a != 'count']
    evals_for_gloss = []
    for a in eval_columns:
        for b in stat_values_tex:
            if b in a:
                evals_for_gloss.append(a.replace(b, ''))
                break
    evals_for_gloss = list(dict.fromkeys(evals_for_gloss))

    ret = {'data': data,
           'it_parameters': vars['it_parameters'],
           'eval_columns': eval_columns,
           'eval_cols_lname': eval_cols_lname,
           'eval_cols_log_scaling': eval_cols_log_scaling,
           'units': units,
           'evals_for_gloss': evals_for_gloss,
           'eval_init_input': None}
    if 'partitions' in vars:
        if 'x_axis_column' in vars:
            ret['x_axis_column'] = []
            for key in vars['data_separators']:
                if key not in vars['partitions']:
                    ret['x_axis_column'].append(key)
        elif 'xy_axis_columns' in vars:
            ret['xy_axis_columns'] = []
            for key in vars['data_separators']:
                if key not in vars['partitions']:
                    ret['xy_axis_columns'].append(key)
        ret['partitions'] = vars['partitions']
    else:
        if 'x_axis_column' in vars:
            ret['x_axis_column'] = vars['x_axis_column']
        elif 'xy_axis_columns' in vars:
            ret['xy_axis_columns'] = vars['xy_axis_columns']

    return ret


def eval_corr_pool_converge_vs_x(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    if 'eval_columns' not in keywords:
        raise ValueError('Missing parameter eval_columns')
    if 'evals_for_gloss' not in keywords:
        raise ValueError('Missing parameter evals_for_gloss')
    needed_evals_p = [['mean_R_diffAll_diff', 'mean_t_angDiff_deg_diff', 'mean_R_diffAll', 'mean_t_angDiff_deg'],
                      ['median_R_diffAll_diff', 'median_t_angDiff_deg_diff', 'median_R_diffAll',
                       'median_t_angDiff_deg']]
    needed_evals = [b for a in needed_evals_p for b in a]
    if not all([a in keywords['eval_columns'] for a in needed_evals]):
        raise ValueError('Some specific entries within parameter eval_columns is missing')

    keywords = prepare_io(**keywords)
    from statistics_and_plot import replaceCSVLabels, \
        add_to_glossary, \
        add_to_glossary_eval, \
        split_large_titles, \
        strToLower, \
        tex_string_coding_style, \
        calcNrLegendCols, \
        capitalizeFirstChar, \
        findUnit, \
        compile_tex, \
        get_limits_log_exp, \
        enl_space_title

    base_ev = ['Rt_diff2', 'R_diffAll_diff', 't_angDiff_deg_diff', 'R_diffAll', 't_angDiff_deg', 'poolSize']

    needed_cols = needed_evals + keywords['x_axis_column'] + keywords['it_parameters']
    tmp = keywords['data'].loc[:, needed_cols]

    df_nr = tmp.groupby(keywords['it_parameters'])
    grp_keys = df_nr.groups.keys()
    data_list = []
    for grp in grp_keys:
        tmp1 = df_nr.get_group(grp)
        tmp1 = tmp1.sort_values(by=keywords['x_axis_column'])
        tmp1['Nr'] = list(range(int(tmp1.shape[0])))
        data_list.append(tmp1)
    tmp_new = pd.concat(data_list, axis=0, ignore_index=False)

    keywords['eval_pre'] = 'mean_'
    tmp_new, keywords = combine_rt_diff2(tmp_new, keywords)
    keywords['eval_pre'] = 'median_'
    tmp_new, keywords = combine_rt_diff2(tmp_new, keywords)
    print_evals = [['mean_Rt_diff2'] + needed_evals_p[0],
                   ['median_Rt_diff2'] + needed_evals_p[1]]
    print_evals_tmp = ['mean_Rt_diff2', 'median_Rt_diff2'] + needed_evals
    gloss_evals = list(dict.fromkeys([a.replace('mean_', '')
                                      if 'mean_' in a else a.replace('median_', '') for a in print_evals_tmp]))
    gloss = add_to_glossary_eval(gloss_evals)

    df_grp = tmp_new.groupby(keywords['it_parameters'])
    grp_keys = df_grp.groups.keys()
    data_list = []
    # mult = 5
    # while mult > 1:
    #     try:
    #         sys.setrecursionlimit(mult * sys.getrecursionlimit())
    #         break
    #     except:
    #         mult -= 1
    for grp in grp_keys:
        tmp1 = df_grp.get_group(grp)
        tmp2, succ = get_converge_img(tmp1, 3, 0.33, 0.05, 'mean_Rt_diff2')
        tmp2.drop(['Nr'] + print_evals[1], axis=1, inplace=True)
        tmp2['stat_type'] = ['mean'] * int(tmp2.shape[0])
        tmp2.columns = [a.replace('mean_', '') for a in tmp2.columns]
        data_list.append(tmp2)
        tmp3, succ = get_converge_img(tmp1, 3, 0.33, 0.05, 'median_Rt_diff2')
        tmp3.drop(['Nr'] + print_evals[0], axis=1, inplace=True)
        tmp3['stat_type'] = ['median'] * int(tmp3.shape[0])
        tmp3.columns = [a.replace('median_', '') for a in tmp3.columns]
        data_list.append(tmp3)
    data_new = pd.concat(data_list, ignore_index=True)

    tex_infos = {'title': 'Most Likely Correspondence Pool Sizes \\& Error Values for Converging Differences ' +
                          'from Frame to Frame of R \\& t Errors '
                          ' for Parameters ' + keywords['sub_title_it_pars'],
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
                 'abbreviations': None
                 }
    t_main_name = 'ml_converge_poolSizes'
    if len(keywords['it_parameters']) > 1:
        itpars_name = '-'.join(keywords['it_parameters'])
    else:
        itpars_name = keywords['it_parameters'][0]
    p_pre = ['mean', 'median']

    tmp1 = data_new.set_index(keywords['it_parameters'])
    if len(keywords['it_parameters']) > 1:
        gloss = add_to_glossary([str(b) for a in tmp1.index for b in a], gloss)
        it_idxs = ['-'.join(map(str, a)) for a in tmp1.index]
        tmp1.index = it_idxs
    else:
        it_idxs = [str(a) for a in tmp1.index]
        gloss = add_to_glossary(it_idxs, gloss)
    tmp1 = tmp1.reset_index().set_index(['stat_type', itpars_name]).unstack(level=-1)
    comb_cols = ['-'.join(a) for a in tmp1.columns]
    tmp1.columns = comb_cols

    t_main_name1 = t_main_name + '_for_opts_' + itpars_name
    t_mean_name = 'data_' + t_main_name1 + '.csv'
    ft_mean_name = os.path.join(keywords['tdata_folder'], t_mean_name)
    with open(ft_mean_name, 'a') as f:
        f.write('# Most likely correspondence pool sizes for converging differences from frame to '
                'frame of mean and median R & t errors ' + '\n')
        f.write('# Different parameters: ' + itpars_name + '\n')
        tmp1.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    use_cols = []
    for a in base_ev:
        use_cols1 = []
        for b in comb_cols:
            if a in b:
                not_found = True
                for c in b.split('-'):
                    if a == c:
                        not_found = False
                        break
                if not_found:
                    continue
                else:
                    use_cols1.append(b)
        if use_cols1:
            use_cols.append(use_cols1)

    for i, ev in enumerate(use_cols):
        legend = []
        for a in ev:
            for b in it_idxs:
                if b in a:
                    legend.append(tex_string_coding_style(b))
                    break
        _, use_limits, use_log, exp_value = get_limits_log_exp(tmp1, True, True, False, None, ev)
        reltex_name = os.path.join(keywords['rel_data_path'], t_mean_name)
        fig_name = capitalizeFirstChar(replaceCSVLabels(base_ev[i], True, False, True)) + \
                   ' for converging differences from frame to frame of mean and median R \\& t errors' \
                   '\\\\for parameters ' + \
                   strToLower(keywords['sub_title_it_pars'])
        fig_name = split_large_titles(fig_name)
        exp_value = enl_space_title(exp_value, fig_name, tmp1, 'stat_type',
                                    len(ev), 'ybar')
        tex_infos['sections'].append({'file': reltex_name,
                                      'name': fig_name,
                                      # If caption is None, the field name is used
                                      'caption': fig_name.replace('\\\\', ' '),
                                      'fig_type': 'ybar',
                                      'plots': ev,
                                      'label_y': replaceCSVLabels(base_ev[i]) +
                                                 findUnit(str(base_ev[i]), keywords['units']),
                                      'plot_x': 'stat_type',
                                      'label_x': 'Used statistic',
                                      'limits': use_limits,
                                      'legend': legend,
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      'use_log_y_axis': use_log,
                                      'xaxis_txt_rows': 1,
                                      'enlarge_lbl_dist': None,
                                      'enlarge_title_space': exp_value,
                                      'use_string_labels': True,
                                      })
        tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    tex_infos['abbreviations'] = gloss
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    if keywords['build_pdf'][0]:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex,
                              keywords['tex_folder'],
                              texf_name,
                              tex_infos['make_index'],
                              os.path.join(keywords['pdf_folder'], pdf_name),
                              tex_infos['figs_externalize']))
    else:
        res = abs(compile_tex(rendered_tex, keywords['tex_folder'], texf_name, False))
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    sel_parts = [b for a in use_cols[3:] for b in a]
    data_new3 = tmp1[sel_parts].mean(axis=0)
    it_idxs = list(dict.fromkeys(it_idxs))
    for it in it_idxs:
        data_new3[it] = data_new3[[a for a in sel_parts if it in a and 'R_diffAll' in a][0]] + \
                        data_new3[[a for a in sel_parts if it in a and 't_angDiff_deg' in a][0]]
    alg_idx = str(data_new3[it_idxs].idxmin())
    rdiff = float(data_new3[[a for a in sel_parts if alg_idx in a and 'R_diffAll' in a][0]])
    tdiff = float(data_new3[[a for a in sel_parts if alg_idx in a and 't_angDiff_deg' in a][0]])
    poolSize = int(data_new3[[a for a in sel_parts if alg_idx in a and 'poolSize' in a][0]])

    main_parameter_name = keywords['res_par_name']  # 'USAC_opt_refine_min_time'
    # Check if file and parameters exist
    from usac_eval import check_par_file_exists, NoAliasDumper
    ppar_file, res = check_par_file_exists(main_parameter_name, keywords['res_folder'], res)
    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = alg_idx.split('-')
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithm': alg_w,
                                         'mean_conv_pool_size': poolSize,
                                         'mean_R_error': rdiff,
                                         'mean_t_error': tdiff}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    return res


def eval_mean_time_poolcorrs(**keywords):
    if 'eval_columns' not in keywords:
        raise ValueError('Missing parameter eval_columns')
    if 'it_parameters' not in keywords:
        raise ValueError('Missing parameter it_parameters')

    grp_itp = 'stereoParameters_maxPoolCorrespondences'
    if 'stereoParameters_maxPoolCorrespondences' not in keywords['it_parameters']:
        raise ValueError('Missing column name stereoParameters_maxPoolCorrespondences in parameter it_parameters')
    if len(keywords['it_parameters']) < 2:
        raise ValueError('Too less it_parameters')

    keywords = prepare_io(**keywords)
    from statistics_and_plot import replaceCSVLabels, \
        glossary_from_list, \
        add_to_glossary_eval, \
        split_large_titles, \
        findUnit, \
        compile_tex, \
        get_limits_log_exp, \
        enl_space_title

    df_grp = keywords['data'].xs('mean', axis=1, level=1,
                                 drop_level=True).reset_index().drop([a for a in keywords['it_parameters']
                                                                      if grp_itp not in a],
                                                                     axis=1).groupby(grp_itp).mean()

    it_idxs = [str(a) for a in df_grp.index]
    gloss = glossary_from_list(it_idxs)
    gloss = add_to_glossary_eval([a for a in df_grp.columns], gloss)
    errvalnames = df_grp.columns.values
    df_grp['tex_it_pars'] = insert_opt_lbreak(it_idxs)
    max_txt_rows = 1
    for idx, val in df_grp['tex_it_pars'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows:
            max_txt_rows = txt_rows

    title = 'Mean Execution Times over the Last 30 Stereo Frames out of 150 Frames for Comparison of Different ' + \
            replaceCSVLabels(grp_itp, True, True, True)

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
    t_main_name = 'mean_time_for_opt_' + grp_itp
    t_mean_name = 'data_' + t_main_name + '.csv'
    ft_mean_name = os.path.join(keywords['tdata_folder'], t_mean_name)
    with open(ft_mean_name, 'a') as f:
        f.write('# Mean execution times over the last 30 stereo frames out of 150 frames\n')
        f.write('# Different parameters: ' + grp_itp + '\n')
        df_grp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
    for ev in errvalnames:
        _, use_limits, use_log, exp_value = get_limits_log_exp(df_grp, True, True, False, 'tex_it_pars',
                                                               ev if len(errvalnames) > 1 else None)
        reltex_name = os.path.join(keywords['rel_data_path'], t_mean_name)
        fig_name = 'Mean execution times over the last 30 stereo frames ' \
                   'out of 150 frames for comparison of different ' + \
                    replaceCSVLabels(grp_itp, True, False, True)
        fig_name = split_large_titles(fig_name)
        exp_value = enl_space_title(exp_value, fig_name, df_grp, 'tex_it_pars',
                                    1, 'ybar')
        tex_infos['sections'].append({'file': reltex_name,
                                      'name': fig_name,
                                      # If caption is None, the field name is used
                                      'caption': fig_name.replace('\\\\', ' '),
                                      'fig_type': 'ybar',
                                      'plots': [ev],
                                      'label_y': replaceCSVLabels(ev) +
                                                 findUnit(str(ev), keywords['units']),
                                      'plot_x': 'tex_it_pars',
                                      'label_x': replaceCSVLabels(grp_itp),
                                      'limits': use_limits,
                                      'legend': None,
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      'use_log_y_axis': use_log,
                                      'xaxis_txt_rows': max_txt_rows,
                                      'enlarge_lbl_dist': None,
                                      'enlarge_title_space': exp_value,
                                      'use_string_labels': True,
                                      })

    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    if keywords['build_pdf'][0]:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex,
                              keywords['tex_folder'],
                              texf_name,
                              tex_infos['make_index'],
                              os.path.join(keywords['pdf_folder'], pdf_name),
                              tex_infos['figs_externalize']))
    else:
        res = abs(compile_tex(rendered_tex, keywords['tex_folder'], texf_name, False))
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)
    return res


def eval_mean_time_pool_3D_dist(**keywords):
    if 'eval_columns' not in keywords:
        raise ValueError('Missing parameter eval_columns')
    if 'it_parameters' not in keywords:
        raise ValueError('Missing parameter it_parameters')
    if len(keywords['it_parameters']) < 2:
        raise ValueError('Too less it_parameters')

    keywords = prepare_io(**keywords)
    from statistics_and_plot import replaceCSVLabels, \
        glossary_from_list, \
        add_to_glossary_eval, \
        split_large_titles, \
        findUnit, \
        compile_tex, \
        get_limits_log_exp, \
        enl_space_title, \
        add_to_glossary

    title = 'Mean Execution Times over the Last 30 Stereo Frames out of 150 Frames for Comparison of Different ' + \
            keywords['sub_title_it_pars']

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
                 'abbreviations': None
                 }

    df_list = []
    gloss_not_calced = True
    t_main_name = 'mean_time_for_opt_'
    gloss = None
    for grp_itp in keywords['it_parameters']:
        df_grp = keywords['data'].xs('mean', axis=1, level=1,
                                     drop_level=True).reset_index().drop([a for a in keywords['it_parameters']
                                                                          if grp_itp not in a],
                                                                         axis=1).groupby(grp_itp).mean()

        it_idxs = [str(a) for a in df_grp.index]
        if gloss_not_calced:
            gloss = glossary_from_list(it_idxs)
            gloss = add_to_glossary_eval([a for a in df_grp.columns], gloss)
            gloss_not_calced = False
        else:
            gloss = add_to_glossary(it_idxs, gloss)
        errvalnames = df_grp.columns.values
        df_grp['tex_it_pars'] = insert_opt_lbreak(it_idxs)
        max_txt_rows = 1
        for idx, val in df_grp['tex_it_pars'].iteritems():
            txt_rows = str(val).count('\\\\') + 1
            if txt_rows > max_txt_rows:
                max_txt_rows = txt_rows
        t_main_name1 = t_main_name + grp_itp
        t_mean_name = 'data_' + t_main_name1 + '.csv'
        ft_mean_name = os.path.join(keywords['tdata_folder'], t_mean_name)
        with open(ft_mean_name, 'a') as f:
            f.write('# Mean execution times over the last 30 stereo frames out of 150 frames\n')
            f.write('# Different parameters: ' + grp_itp + '\n')
            df_grp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
        df_list.append(df_grp.reset_index().drop([grp_itp, 'tex_it_pars'], axis=1))
        for ev in errvalnames:
            _, use_limits, use_log, exp_value = get_limits_log_exp(df_grp, True, True, False, 'tex_it_pars',
                                                                   ev if len(errvalnames) > 1 else None)
            reltex_name = os.path.join(keywords['rel_data_path'], t_mean_name)
            fig_name = 'Mean execution times over the last 30 stereo frames ' \
                       'out of 150 frames for comparison of different ' + \
                        replaceCSVLabels(grp_itp, True, False, True)
            fig_name = split_large_titles(fig_name)
            exp_value = enl_space_title(exp_value, fig_name, df_grp, 'tex_it_pars',
                                        1, 'ybar')
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': fig_name,
                                          # If caption is None, the field name is used
                                          'caption': fig_name.replace('\\\\', ' '),
                                          'fig_type': 'ybar',
                                          'plots': [ev],
                                          'label_y': replaceCSVLabels(ev) +
                                                     findUnit(str(ev), keywords['units']),
                                          'plot_x': 'tex_it_pars',
                                          'label_x': replaceCSVLabels(grp_itp),
                                          'limits': use_limits,
                                          'legend': None,
                                          'legend_cols': 1,
                                          'use_marks': False,
                                          'use_log_y_axis': use_log,
                                          'xaxis_txt_rows': max_txt_rows,
                                          'enlarge_lbl_dist': None,
                                          'enlarge_title_space': exp_value,
                                          'use_string_labels': True,
                                          })
    tex_infos['abbreviations'] = gloss
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    base_out_name = 'tex_' + t_main_name + '-'.join(keywords['it_parameters'])
    texf_name = base_out_name + '.tex'
    if keywords['build_pdf'][0]:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex,
                              keywords['tex_folder'],
                              texf_name,
                              tex_infos['make_index'],
                              os.path.join(keywords['pdf_folder'], pdf_name),
                              tex_infos['figs_externalize']))
    else:
        res = abs(compile_tex(rendered_tex, keywords['tex_folder'], texf_name, False))
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    df = pd.concat(df_list).describe()
    t_main_name1 = 'statistics_over_mean_time_for_opts_' + '-'.join(keywords['it_parameters'])
    t_mean_name = 'data_' + t_main_name1 + '.csv'
    ft_mean_name = os.path.join(keywords['res_folder'], t_mean_name)
    with open(ft_mean_name, 'a') as f:
        f.write('# Statistic over mean execution times over the last 30 stereo frames out of 150 frames\n')
        f.write('# Different parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
        df.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    return res
