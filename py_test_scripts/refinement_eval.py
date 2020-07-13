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

Description: Calculates the best performing parameter values with given testing data as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np, math
# import modin.pandas as pd
import pandas as pd
#from jinja2 import Template as ji
import jinja2 as ji
import ruamel.yaml as yaml
from usac_eval import ji_env, get_time_fixed_kp, insert_opt_lbreak, prepare_io

def filter_nr_kps_calc_t(**vars):
    tmp = vars['data'].loc[vars['data']['nrTP'] == '100to1000'].copy(deep=True)
    linref = tmp['linRefinement_us']
    ba = tmp['bundleAdjust_us']
    tmp['linRef_BA_us'] = linref + ba
    return tmp

def filter_nr_kps_calc_t_all(**vars):
    tmp = vars['data'].loc[vars['data']['nrTP'] == '500'].copy(deep=True)
    linref = tmp['linRefinement_us']
    ba = tmp['bundleAdjust_us']
    sac = tmp['robEstimationAndRef_us']
    tmp['linRef_BA_sac_us'] = linref + ba + sac
    return tmp

def get_best_comb_scenes_1(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    from usac_eval import pars_calc_single_fig_partitions
    ret = pars_calc_single_fig_partitions(**keywords)
    # if len(ret['partitions']) != 2:
    #     raise ValueError('Only a number of 2 partitions is allowed in function get_best_comb_scenes_1')
    b_new_index = ret['b'].stack().reset_index()
    if len(ret['it_parameters'])  > 1:
        it_pars_name = '-'.join(ret['it_parameters'])
    else:
        it_pars_name = ret['it_parameters'][0]
    b_mean_l = []
    drops = []
    for i, val in enumerate(ret['partitions']):
        drops.append([val1 for i1, val1 in enumerate(ret['partitions']) if i1 != i])
        drops[-1] += [ret['grp_names'][-1]]
        b_mean_l.append(b_new_index.drop(drops[-1], axis=1).groupby(ret['it_parameters'] + [val]).mean())
    drops.append(ret['partitions'])
    b_mean_l.append(b_new_index.drop(drops[-1], axis=1).groupby(ret['it_parameters'] +
                                                                        [ret['grp_names'][-1]]).mean())
    data_parts = ret['partitions'] + [ret['grp_names'][-1]]
    data_it_indices = []
    data_it_b_columns = []
    column_legend = []
    b_mean_name = []
    part_name_title = []
    is_numeric = []
    from statistics_and_plot import replaceCSVLabels, \
        tex_string_coding_style, \
        calcNrLegendCols, \
        strToLower, \
        compile_tex, \
        split_large_titles, \
        get_limits_log_exp, \
        enl_space_title, \
        check_if_neg_values, \
        handle_nans, \
        short_concat_str, \
        check_file_exists_rename
    if 'error_type_text' in keywords:
        title_text = 'Mean ' + keywords['error_type_text'] + ' Over Different Properties ' +\
                     ' for Parameter Variations of ' + ret['sub_title_it_pars']
    else:
        title_text = 'Mean Combined R \\& t Errors Over Different Properties ' +\
                     ' for Parameter Variations of ' + ret['sub_title_it_pars']
    tex_infos = {'title': title_text,
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
    for i, (df, dp) in enumerate(zip(b_mean_l, data_parts)):
        tmp = df.reset_index().set_index(ret['it_parameters'])
        if len(ret['it_parameters']) > 1:
            data_it_indices.append(['-'.join(map(str, a)) for a in tmp.index])
            tmp.index = data_it_indices[-1]
            tmp.index.name = it_pars_name
        else:
            data_it_indices.append([str(a) for a in tmp.index])
        is_numeric.append(pd.to_numeric(tmp[dp], errors='coerce').notnull().all())
        tmp = tmp.reset_index().set_index([dp] + [it_pars_name]).unstack(level=-1)
        column_legend.append([tex_string_coding_style(b) for a in tmp.columns for b in a if b in data_it_indices[-1]])
        if 'error_col_name' in keywords:
            err_name = keywords['error_col_name'] + '_mean'
        else:
            err_name = 'b_mean'
        data_it_b_columns.append(['-'.join([str(b) if b in data_it_indices[-1] else err_name
                                           for b in a]) for a in tmp.columns])
        tmp.columns = data_it_b_columns[-1]

        if 'file_name_err_part' in keywords:
            b_mean_name_tmp = 'data_mean_' + keywords['file_name_err_part'] + '_over_' + \
                              short_concat_str(list(map(str, drops[i]))) + '_vs_' + str(dp) +\
                              '_for_opts_' + short_concat_str(ret['it_parameters']) + '.csv'
        else:
            b_mean_name_tmp = 'data_mean_RTerrors_over_' + short_concat_str(list(map(str, drops[i]))) + '_vs_' + \
                              str(dp) + '_for_opts_' + short_concat_str(ret['it_parameters']) + '.csv'
        b_mean_name.append(b_mean_name_tmp)
        fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name[-1])
        fb_mean_name = check_file_exists_rename(fb_mean_name)
        b_mean_name[-1] = os.path.basename(fb_mean_name)
        with open(fb_mean_name, 'a') as f:
            if 'error_type_text' in keywords:
                f.write('# Mean' + strToLower(keywords['error_type_text']) + '(' + err_name + ')' +
                        ' over properties ' + '-'.join(map(str, drops[i])) +
                        ' compared to ' + str(dp) + ' for options ' +
                        '-'.join(ret['it_parameters']) + '\n')
            else:
                f.write('# Mean combined R & t errors (b_mean) over properties ' + '-'.join(map(str, drops[i])) +
                        ' compared to ' + str(dp) + ' for options ' +
                        '-'.join(ret['it_parameters']) + '\n')
            tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        part_name_title.append('')
        len_drops = len(drops[i])
        for i1, val in enumerate(drops[i]):
            part_name_title[-1] += replaceCSVLabels(val, True, False, True)
            if (len_drops <= 2):
                if i1 < len_drops - 1:
                    part_name_title[-1] += ' and '
            else:
                if i1 < len_drops - 2:
                    part_name_title[-1] += ', '
                elif i1 < len_drops - 1:
                    part_name_title[-1] += ', and '

        _, use_limits, use_log, exp_value = get_limits_log_exp(tmp)
        reltex_name = os.path.join(ret['rel_data_path'], b_mean_name[-1])
        if 'error_type_text' in keywords:
            fig_name = 'Mean ' + strToLower(keywords['error_type_text']) + \
                       ' over\\\\properties ' + part_name_title[-1] + \
                       '\\\\vs ' + replaceCSVLabels(str(dp), True, False, True) + \
                       ' for parameter variations of\\\\' + strToLower(ret['sub_title_it_pars'])
            label_y = 'mean ' + strToLower(keywords['error_type_text'])
        else:
            fig_name = 'Mean combined R \\& t errors over\\\\properties ' + part_name_title[-1] + \
                       '\\\\vs ' + replaceCSVLabels(str(dp), True, False, True) + \
                       ' for parameter variations of\\\\' + strToLower(ret['sub_title_it_pars'])
            label_y = 'mean R \\& t error $e_{R\\bm{t}}$'

        nr_plots = len(data_it_b_columns[-1])
        exp_value_o = exp_value
        if nr_plots <= 20:
            nr_plots_i = [nr_plots]
        else:
            pp = math.floor(nr_plots / 20)
            nr_plots_i = [20] * int(pp)
            rp = nr_plots - pp * 20
            if rp > 0:
                nr_plots_i += [nr_plots - pp * 20]
        pcnt = 0
        for i1, it1 in enumerate(nr_plots_i):
            ps = data_it_b_columns[-1][pcnt: pcnt + it1]
            cl = column_legend[-1][pcnt: pcnt + it1]
            pcnt += it1
            if nr_plots > 20:
                sec_name1 = fig_name + ' -- part ' + str(i1 + 1)
            else:
                sec_name1 = fig_name

            is_neg = check_if_neg_values(tmp, ps, use_log, use_limits)
            sec_name1 = split_large_titles(sec_name1)
            exp_value = enl_space_title(exp_value_o, sec_name1, tmp, dp,
                                        len(ps), 'smooth')
            x_rows = handle_nans(tmp, ps, not is_numeric[-1], 'smooth')
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': sec_name1.replace('\\\\', ' '),
                                          'title': sec_name1,
                                          'title_rows': sec_name1.count('\\\\'),
                                          'fig_type': 'smooth',
                                          'plots': ps,
                                          'label_y': label_y,  # Label of the value axis. For xbar it labels the x-axis
                                          # Label/column name of axis with bars. For xbar it labels the y-axis
                                          'label_x': replaceCSVLabels(str(dp)),
                                          # Column name of axis with bars. For xbar it is the column for the y-axis
                                          'print_x': str(dp),
                                          # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                          'print_meta': False,
                                          'plot_meta': None,
                                          # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                          'rotate_meta': 0,
                                          'limits': use_limits,
                                          # If None, no legend is used, otherwise use a list
                                          'legend': cl,
                                          'legend_cols': None,
                                          'use_marks': ret['use_marks'],
                                          # The x/y-axis values are given as strings if True
                                          'use_string_labels': True if not is_numeric[-1] else False,
                                          'use_log_y_axis': use_log,
                                          'xaxis_txt_rows': 1,
                                          'enlarge_lbl_dist': None,
                                          'enlarge_title_space': exp_value,
                                          'large_meta_space_needed': False,
                                          'is_neg': is_neg,
                                          'nr_x_if_nan': x_rows,
                                          'caption': sec_name1.replace('\\\\', ' ')
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    if 'file_name_err_part' in keywords:
        t_main_name = 'mean_' + keywords['file_name_err_part'] + '_over_different_properties_vs_1_property'
    else:
        t_main_name = 'mean_RTerrors_over_different_properties_vs_1_property'
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    if ret['build_pdf'][1]:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex,
                              ret['tex_folder'],
                              texf_name,
                              tex_infos['make_index'],
                              os.path.join(ret['pdf_folder'], pdf_name),
                              tex_infos['figs_externalize']))
    else:
        res = abs(compile_tex(rendered_tex, ret['tex_folder'], texf_name, False))
    if res != 0:
        ret['res'] += res
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    if 'error_type_text' in keywords:
        title_text = 'Minimum Mean ' + keywords['error_type_text'] + ' Over Different Properties ' +\
                     ' for Parameter Variations of ' + ret['sub_title_it_pars']
    else:
        title_text = 'Minimum Mean Combined R \\& t Errors Over Different Properties ' +\
                     ' for Parameter Variations of ' + ret['sub_title_it_pars']
    tex_infos = {'title': title_text,
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
    data_parts_min = []
    if 'error_col_name' in keywords:
        err_name = keywords['error_col_name'] + '_min'
    else:
        err_name = 'b_min'
    for df, dp, dit, fn, pnt, isn, dps in zip(b_mean_l,
                                              data_parts,
                                              data_it_indices,
                                              b_mean_name,
                                              part_name_title,
                                              is_numeric,
                                              drops):
        tmp = df.reset_index().set_index(ret['it_parameters'])
        if len(ret['it_parameters']) > 1:
            tmp.index = dit
            tmp.index.name = it_pars_name
        tmp.rename(columns={tmp.columns[1]: err_name}, inplace=True)
        tmp.reset_index(inplace=True)
        data_parts_min.append(tmp.loc[tmp.groupby(dp)[err_name].idxmin()].set_index(dp))
        # data_parts_min[-1]['tex_it_pars'] = insert_opt_lbreak(data_parts_min[-1][it_pars_name].tolist())
        data_parts_min[-1]['tex_it_pars'] = data_parts_min[-1][it_pars_name].apply(lambda x: tex_string_coding_style(x))
        data_f_name = fn.replace('data_mean','data_min_mean')
        fb_mean_name = os.path.join(ret['tdata_folder'], data_f_name)
        fb_mean_name = check_file_exists_rename(fb_mean_name)
        data_f_name = os.path.basename(fb_mean_name)
        with open(fb_mean_name, 'a') as f:
            if 'error_type_text' in keywords:
                f.write('# Minimum mean' + strToLower(keywords['error_type_text']) + '(' + err_name + ')' +
                        ' and their corresponding options over properties ' +
                        ' compared to ' + str(dp) + ' for options ' +
                        '-'.join(ret['it_parameters']) + '\n')
            else:
                f.write('# Minimum mean combined R & t errors (b_min) and their '
                        'corresponding options over properties ' +
                        '-'.join(map(str, dps)) +
                        ' compared to ' + str(dp) + ' for options ' +
                        '-'.join(ret['it_parameters']) + '\n')
            data_parts_min[-1].to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        _, use_limits, use_log, exp_value = get_limits_log_exp(data_parts_min[-1], True, True, False,
                                                               [it_pars_name, 'tex_it_pars'])
        is_neg = check_if_neg_values(data_parts_min[-1], err_name, use_log, use_limits)
        reltex_name = os.path.join(ret['rel_data_path'], data_f_name)
        if 'error_type_text' in keywords:
            fig_name = 'Minimum mean ' + strToLower(keywords['error_type_text']) + \
                       ' and their corresponding parameter over\\\\properties ' + \
                       pnt + '\\\\vs ' + replaceCSVLabels(str(dp), True, False, True) + \
                       ' for parameter variations of\\\\' + strToLower(ret['sub_title_it_pars'])
            caption = 'Minimum mean combined ' + strToLower(keywords['error_type_text']) + \
                      ' (corresponding parameter on top of each bar) over properties ' + \
                      pnt + ' vs ' + replaceCSVLabels(str(dp), True) + \
                      ' for parameter variations of ' + strToLower(ret['sub_title_it_pars'])
            label_y = 'min. mean ' + strToLower(keywords['error_type_text'])
        else:
            fig_name = 'Minimum mean combined R \\& t errors and their ' \
                       'corresponding parameter over\\\\properties ' + \
                       pnt + '\\\\vs ' + replaceCSVLabels(str(dp), True, False, True) + \
                       ' for parameter variations of\\\\' + strToLower(ret['sub_title_it_pars'])
            caption = 'Minimum mean combined R \\& t errors ' \
                      '(corresponding parameter on top of each bar) over properties ' + \
                      pnt + ' vs ' + replaceCSVLabels(str(dp), True) + \
                      ' for parameter variations of ' + strToLower(ret['sub_title_it_pars'])
            label_y = 'min. mean R \\& t error $e_{R\\bm{t}}$'
        fig_name = split_large_titles(fig_name)
        exp_value = enl_space_title(exp_value, fig_name, data_parts_min[-1], dp,
                                    1, 'ybar')
        x_rows = handle_nans(data_parts_min[-1], err_name, not isn, 'ybar')
        tex_infos['sections'].append({'file': reltex_name,
                                      'name': fig_name.replace('\\\\', ' '),
                                      'title': fig_name,
                                      'title_rows': fig_name.count('\\\\'),
                                      'fig_type': 'ybar',
                                      'plots': [err_name],
                                      'label_y': label_y,
                                      # Label of the value axis. For xbar it labels the x-axis
                                      # Label/column name of axis with bars. For xbar it labels the y-axis
                                      'label_x': replaceCSVLabels(str(dp)),
                                      # Column name of axis with bars. For xbar it is the column for the y-axis
                                      'print_x': str(dp),
                                      # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                      'print_meta': True,
                                      'plot_meta': ['tex_it_pars'],
                                      # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                      'rotate_meta': 45,
                                      'limits': use_limits,
                                      # If None, no legend is used, otherwise use a list
                                      'legend': None,
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      # The x/y-axis values are given as strings if True
                                      'use_string_labels': True if not isn else False,
                                      'use_log_y_axis': use_log,
                                      'xaxis_txt_rows': 1,
                                      'enlarge_lbl_dist': None,
                                      'enlarge_title_space': exp_value,
                                      'large_meta_space_needed': True,
                                      'is_neg': is_neg,
                                      'nr_x_if_nan': x_rows,
                                      'caption': caption
                                      })
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    if 'file_name_err_part' in keywords:
        t_main_name = 'min_mean_' + keywords['file_name_err_part'] + '_over_different_properties_vs_1_property'
    else:
        t_main_name = 'min_mean_RTerrors_over_different_properties_vs_1_property'
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    if ret['build_pdf'][2]:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex,
                              ret['tex_folder'],
                              texf_name,
                              tex_infos['make_index'],
                              os.path.join(ret['pdf_folder'], pdf_name),
                              tex_infos['figs_externalize']))
    else:
        res = abs(compile_tex(rendered_tex, ret['tex_folder'], texf_name, False))
    if res != 0:
        ret['res'] += res
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    data_min = pd.concat(data_parts_min, ignore_index=True)#, sort=False, copy=False)
    data_min_c = data_min[it_pars_name].value_counts()
    if data_min_c.count() > 1:
        if data_min_c.iloc[0] == data_min_c.iloc[1]:
            data_min_c = data_min_c.loc[data_min_c == data_min_c.iloc[0]]
            data_min2 = data_min.loc[data_min[it_pars_name].isin(data_min_c.index.values)]
            data_min2 = data_min2.loc[data_min2[err_name].idxmin()]
            alg = str(data_min2[it_pars_name])
            b_min = float(data_min2[err_name])
        else:
            data_min2 = data_min.loc[data_min[it_pars_name] == data_min_c.index.values[0]]
            if len(data_min2.shape) == 1:
                alg = str(data_min2[it_pars_name])
                b_min = float(data_min2[err_name])
            else:
                alg = str(data_min2[it_pars_name].iloc[0])
                b_min = float(data_min2[err_name].mean())
    else:
        alg = str(data_min[it_pars_name].iloc[0])
        b_min = float(data_min[err_name].mean())

    main_parameter_name = keywords['res_par_name']
    # Check if file and parameters exist
    from usac_eval import check_par_file_exists, NoAliasDumper
    ppar_file, ret['res'] = check_par_file_exists(main_parameter_name, ret['res_folder'], ret['res'])

    import eval_mutex as em
    em.init_lock()
    em.acquire_lock()
    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = alg.split('-')
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithms': alg_w,
                                         err_name: b_min}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    em.release_lock()

    return ret['res']


def estimate_alg_time_fixed_kp_agg(**vars):
    if 'res_par_name' not in vars:
        raise ValueError('Missing parameter res_par_name')
    tmp, col_name = get_time_fixed_kp(**vars)

    tmp.set_index(vars['it_parameters'], inplace=True)
    from statistics_and_plot import glossary_from_list, get_limits_log_exp, enl_space_title, short_concat_str
    if len(vars['it_parameters']) > 1:
        gloss = glossary_from_list([str(b) for a in tmp.index for b in a])
        par_cols = ['-'.join(map(str, a)) for a in tmp.index]
        tmp.index = par_cols
        it_pars_cols_name = '-'.join(map(str, vars['it_parameters']))
        tmp.index.name = it_pars_cols_name
    else:
        gloss = glossary_from_list([str(a) for a in tmp.index])
        it_pars_cols_name = vars['it_parameters'][0]
        par_cols = [a for a in tmp.index]
    tmp['pars_tex'] = insert_opt_lbreak(par_cols)
    max_txt_rows = 1
    for idx, val in tmp['pars_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows:
            max_txt_rows = txt_rows

    tmp_min = tmp.loc[[tmp[col_name].idxmin(axis=0)]].reset_index()

    vars = prepare_io(**vars)
    from statistics_and_plot import compile_tex, strToLower, split_large_titles, handle_nans, check_file_exists_rename
    t_main_name = 'mean_time_for_' + \
                  str(int(vars['nr_target_kps'])) + 'kpts_for_opts_' + \
                  short_concat_str(list(map(str, vars['it_parameters'])))
    t_mean_name = 'data_' + t_main_name + '.csv'
    ft_mean_name = os.path.join(vars['tdata_folder'], t_mean_name)
    ft_mean_name = check_file_exists_rename(ft_mean_name)
    t_mean_name = os.path.basename(ft_mean_name)
    with open(ft_mean_name, 'a') as f:
        f.write('# Mean execution times extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    title = 'Mean Execution Times for Parameter Variations of ' + vars['sub_title_it_pars'] + \
            ' Extrapolated for ' + str(int(vars['nr_target_kps'])) + ' Keypoints'
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': gloss
                 }
    _, use_limits, use_log, exp_value = get_limits_log_exp(tmp, True, True, False, None, col_name)
    reltex_name = os.path.join(vars['rel_data_path'], t_mean_name)
    fig_name = 'Mean execution times for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
               '\\\\extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    fig_name = split_large_titles(fig_name)
    exp_value = enl_space_title(exp_value, fig_name, tmp, 'pars_tex',
                                1, 'xbar')
    x_rows = handle_nans(tmp, col_name, True, 'xbar')
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': fig_name.replace('\\\\', ' '),
                                  'title': fig_name,
                                  'title_rows': fig_name.count('\\\\'),
                                  'fig_type': 'xbar',
                                  'plots': [col_name],
                                  'label_y': 'mean execution time/$\\mu s$',
                                  # Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'Parameter',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'pars_tex',
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': False,
                                  'plot_meta': None,
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 0,
                                  'limits': use_limits,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': None,
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': True,
                                  'use_log_y_axis': use_log,
                                  'xaxis_txt_rows': max_txt_rows,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': exp_value,
                                  'large_meta_space_needed': False,
                                  'is_neg': False,
                                  'nr_x_if_nan': x_rows,
                                  'caption': fig_name.replace('\\\\', ' ')
                                  })
    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    if any(vars['build_pdf']):
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex,
                              vars['tex_folder'],
                              texf_name,
                              False,
                              os.path.join(vars['pdf_folder'], pdf_name),
                              tex_infos['figs_externalize']))
    else:
        res = abs(compile_tex(rendered_tex, vars['tex_folder'], texf_name, False))
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    main_parameter_name = vars['res_par_name']#'USAC_opt_refine_min_time'
    # Check if file and parameters exist
    from usac_eval import check_par_file_exists, NoAliasDumper
    ppar_file, res = check_par_file_exists(main_parameter_name, vars['res_folder'], res)

    import eval_mutex as em
    em.init_lock()
    em.acquire_lock()
    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = tmp_min[it_pars_cols_name][0].split('-')
        min_t = float(tmp_min[col_name][0])
        if len(vars['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(vars['it_parameters']):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithms': alg_w,
                                         't_min': min_t}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    em.release_lock()

    return res


def combineK(data):
    #Get R and t mean and standard deviation values
    stat_K1 = data['K1_cxyfxfyNorm'].unstack()
    stat_K2 = data['K2_cxyfxfyNorm'].unstack()
    stat_K12_mean = (stat_K1['mean'] + stat_K2['mean']) / 2
    stat_K12_std = (stat_K1['std'] + stat_K2['std']) / 2
    comb_stat_K12 = stat_K12_mean + stat_K12_std

    return comb_stat_K12


def pars_calc_single_fig_K(**keywords):
    from statistics_and_plot import short_concat_str, check_file_exists_rename
    if len(keywords) < 3:
        raise ValueError('Wrong number of arguments for function pars_calc_single_fig_K')
    if 'data' not in keywords:
        raise ValueError('Missing data argument of function pars_calc_single_fig_K')
    data = keywords['data']
    ret = {}
    if 'res_folder' not in keywords:
        raise ValueError('Missing res_folder argument of function pars_calc_single_fig_K')
    ret['res_folder'] = keywords['res_folder']
    ret['use_marks'] = False
    if 'use_marks' not in keywords:
        print('No information provided if marks should be used: Disabling marks')
    else:
        ret['use_marks'] = keywords['use_marks']
    ret['build_pdf'] = (False, True,)
    if 'build_pdf' in keywords:
        ret['build_pdf'] = keywords['build_pdf']
    if len(ret['build_pdf']) < 2:
        raise ValueError('Wrong number of arguments for build_pdf')
    ret['pdf_folder'] = None
    if any(ret['build_pdf']):
        ret['pdf_folder'] = os.path.join(ret['res_folder'], 'pdf')
        try:
            os.mkdir(ret['pdf_folder'])
        except FileExistsError:
            # print('Folder', ret['pdf_folder'], 'for storing pdf files already exists')
            pass
    ret['tex_folder'] = os.path.join(ret['res_folder'], 'tex')
    try:
        os.mkdir(ret['tex_folder'])
    except FileExistsError:
        # print('Folder', ret['tex_folder'], 'for storing tex files already exists')
        pass
    ret['tdata_folder'] = os.path.join(ret['tex_folder'], 'data')
    try:
        os.mkdir(ret['tdata_folder'])
    except FileExistsError:
        # print('Folder', ret['tdata_folder'], 'for storing data files already exists')
        pass
    ret['rel_data_path'] = os.path.relpath(ret['tdata_folder'], ret['tex_folder'])
    ret['grp_names'] = data.index.names
    ret['it_parameters'] = keywords['it_parameters']
    ret['dataf_name_main'] = str(ret['grp_names'][-1]) + '_for_options_' + short_concat_str(keywords['it_parameters'])
    ret['dataf_name'] = ret['dataf_name_main'] + '.csv'
    ret['b'] = combineK(data)
    ret['b'] = ret['b'].T
    from statistics_and_plot import glossary_from_list, add_to_glossary_eval, enl_space_title
    if len(keywords['it_parameters']) > 1:
        ret['gloss'] = glossary_from_list([str(b) for a in ret['b'].columns for b in a])
        ret['b'].columns = ['-'.join(map(str, a)) for a in ret['b'].columns]
        ret['b'].columns.name = '-'.join(keywords['it_parameters'])
    else:
        ret['gloss'] = glossary_from_list([str(a) for a in ret['b'].columns])
    if 'K1_cxyfxfyNorm' in data.columns:
        ret['gloss'] = add_to_glossary_eval('K12_cxyfxfyNorm', ret['gloss'])
    else:
        raise ValueError('Combined Rt error column is missing.')
    ret['gloss'] = add_to_glossary_eval(keywords['eval_columns'] +
                                        keywords['x_axis_column'], ret['gloss'])

    b_name = 'data_Kerrors_vs_' + ret['dataf_name']
    fb_name = os.path.join(ret['tdata_folder'], b_name)
    fb_name = check_file_exists_rename(fb_name)
    b_name = os.path.basename(fb_name)
    with open(fb_name, 'a') as f:
        f.write('# Combined camera matrix errors vs ' + str(ret['grp_names'][-1]) + '\n')
        f.write('# Parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
        ret['b'].to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
    ret['sub_title'] = ''
    nr_it_parameters = len(keywords['it_parameters'])
    from statistics_and_plot import tex_string_coding_style, \
        compile_tex, \
        calcNrLegendCols, \
        replaceCSVLabels, \
        split_large_titles, \
        get_limits_log_exp, \
        handle_nans
    for i, val in enumerate(keywords['it_parameters']):
        ret['sub_title'] += replaceCSVLabels(val, True, True, True)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                ret['sub_title'] += ' and '
        else:
            if i < nr_it_parameters - 2:
                ret['sub_title'] += ', '
            elif i < nr_it_parameters - 1:
                ret['sub_title'] += ', and '
    tex_infos = {'title': 'Combined Camera Matrix Errors vs ' +
                          replaceCSVLabels(str(ret['grp_names'][-1]), True, True, True) +
                          ' for Parameter Variations of ' + ret['sub_title'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': ret['gloss']
                 }
    _, use_limits, use_log, exp_value = get_limits_log_exp(ret['b'])
    is_numeric = pd.to_numeric(ret['b'].reset_index()[ret['grp_names'][-1]], errors='coerce').notnull().all()
    x_rows = handle_nans(ret['b'], list(ret['b'].columns.values), not is_numeric, 'smooth')
    section_name = 'Combined camera matrix errors $e_{\\mli{K1,2}}$ vs ' +\
                   replaceCSVLabels(str(ret['grp_names'][-1]), True, False, True) +\
                   ' for parameter variations of\\\\' + ret['sub_title']
    section_name = split_large_titles(section_name)
    exp_value = enl_space_title(exp_value, section_name, ret['b'], ret['grp_names'][-1],
                                len(list(ret['b'].columns.values)), 'smooth')
    reltex_name = os.path.join(ret['rel_data_path'], b_name)
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': section_name,
                                  # If caption is None, the field name is used
                                  'caption': 'Combined camera matrix errors $e_{\\mli{K1,2}}$ vs ' +
                                             replaceCSVLabels(str(ret['grp_names'][-1]), True) +
                                             ' for parameter variations of ' + ret['sub_title'],
                                  'fig_type': 'smooth',
                                  'plots': list(ret['b'].columns.values),
                                  'label_y': 'Combined camera matrix error',
                                  'plot_x': str(ret['grp_names'][-1]),
                                  'label_x': replaceCSVLabels(str(ret['grp_names'][-1])),
                                  'limits': use_limits,
                                  'legend': [tex_string_coding_style(a) for a in list(ret['b'].columns.values)],
                                  'legend_cols': None,
                                  'use_marks': ret['use_marks'],
                                  'use_log_y_axis': use_log,
                                  'xaxis_txt_rows': 1,
                                  'nr_x_if_nan': x_rows,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': exp_value,
                                  'use_string_labels': True if not is_numeric else False
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    base_out_name = 'tex_Kerrors_vs_' + ret['dataf_name_main']
    texf_name = base_out_name + '.tex'
    if ret['build_pdf'][0]:
        pdf_name = base_out_name + '.pdf'
        ret['res'] = abs(compile_tex(rendered_tex,
                                     ret['tex_folder'],
                                     texf_name,
                                     False,
                                     os.path.join(ret['pdf_folder'], pdf_name),
                                     tex_infos['figs_externalize']))
    else:
        ret['res'] = abs(compile_tex(rendered_tex, ret['tex_folder'], texf_name, False))
    if ret['res'] != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)
    return ret


def get_best_comb_inlrat_k(**keywords):
    from statistics_and_plot import check_file_exists_rename
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    ret = pars_calc_single_fig_K(**keywords)
    b_mean = ret['b'].mean(axis=0)
    b_mean_best = b_mean.idxmin()
    alg_best = str(b_mean_best)
    b_best = float(b_mean.loc[b_mean_best])

    b_mean = b_mean.reset_index()
    b_mean.columns = ['options', 'ke_mean']
    # Insert a tex line break for long options
    b_mean['options_tex'] = insert_opt_lbreak(ret['b'].columns)
    max_txt_rows = 1
    for idx, val in b_mean['options_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows:
            max_txt_rows = txt_rows
    b_mean_name = 'data_mean_Kerrors_over_all_' + ret['dataf_name']
    fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
    fb_mean_name = check_file_exists_rename(fb_mean_name)
    b_mean_name = os.path.basename(fb_mean_name)
    with open(fb_mean_name, 'a') as f:
        f.write('# Mean combined camera matrix errors over all ' + str(ret['grp_names'][-1]) + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
        b_mean.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    # Get data for tex file generation
    if len(ret['b'].columns) > 10:
        fig_type = 'xbar'
    else:
        fig_type = 'ybar'
    from statistics_and_plot import replaceCSVLabels, check_if_neg_values, handle_nans
    is_neg = check_if_neg_values(b_mean, 'ke_mean', False, None)
    x_rows = handle_nans(b_mean, 'ke_mean', True, fig_type)
    tex_infos = {'title': 'Mean Combined Camera Matrix Errors over all ' +
                          replaceCSVLabels(str(ret['grp_names'][-1]), True, True, True) +
                          ' for Parameter Variations of ' + ret['sub_title'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': ret['gloss']
                 }
    section_name = 'Mean combined camera matrix errors $e_{\\mli{K1,2}}$ over all ' + \
                   replaceCSVLabels(str(ret['grp_names'][-1]), True, False, True)
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                  'name': section_name,
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': fig_type,
                                  'plots': ['ke_mean'],
                                  'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'Options',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'options_tex',
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': False,
                                  'plot_meta': [],
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 0,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': None,
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': True,
                                  'use_log_y_axis': False,
                                  'xaxis_txt_rows': max_txt_rows,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': False,
                                  'large_meta_space_needed': False,
                                  'is_neg': is_neg,
                                  'nr_x_if_nan': x_rows,
                                  'caption': 'Mean combined camera matrix errors $e_{\\mli{K1,2}}$ (error bars) over all ' +
                                             replaceCSVLabels(str(ret['grp_names'][-1]), True) + '.'
                                  })
    from usac_eval import compile_2D_bar_chart, check_par_file_exists, NoAliasDumper
    ret['res'] = compile_2D_bar_chart('tex_mean_K-errors_' + ret['grp_names'][-1], tex_infos, ret)

    main_parameter_name = keywords['res_par_name']#'USAC_opt_refine_ops_inlrat'
    # Check if file and parameters exist
    ppar_file, ret['res'] = check_par_file_exists(main_parameter_name, ret['res_folder'], ret['res'])

    import eval_mutex as em
    em.init_lock()
    em.acquire_lock()
    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = alg_best.split('-')
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithms': alg_w,
                                         'ke_best_val': b_best}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    em.release_lock()
    return ret['res']
