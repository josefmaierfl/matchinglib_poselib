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
    if 'data_separators' not in keywords:
        raise ValueError('data_separators missing!')
    needed_cols = list(dict.fromkeys(individual_grps + keywords['data_separators'] +
                                     ['Nr', 'accumCorrs'] + keywords['eval_columns']))
    if 'addit_cols' in keywords and keywords['addit_cols']:
        drop_cols = [a for a in needed_cols if a not in individual_grps and
                     a not in keywords['eval_columns'] and
                     a not in keywords['addit_cols']]
    else:
        drop_cols = [a for a in needed_cols if a not in individual_grps and a not in keywords['eval_columns']]
    # Split in sequences
    df_grp = keywords['data'].loc[:, needed_cols].copy(deep=True).groupby(keywords['data_separators'])
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
            tmp1 = tmp.loc[idx[0]:idx[1], :].copy(deep=True)
            row_cnt = tmp1.shape[0]
            accumCorrs_max = tmp1['accumCorrs'].max()
            tmp1['accumCorrs_max'] = [accumCorrs_max] * row_cnt
            tmp1.drop(drop_cols, axis=1, inplace=True)
            df_list2.append(tmp1)
        df_list.append(pd.concat(df_list2, ignore_index=False))
    tmp = pd.concat(df_list, ignore_index=False)
    keywords['it_parameters'] = list(dict.fromkeys(keywords['it_parameters'] + ['accumCorrs_max']))
    keywords['data'] = tmp
    return keywords


def get_mean_y_vs_x_it(**keywords):
    if 'used_x_axis' not in keywords:
        raise ValueError('used_x_axis missing!')
    if 'cols_for_mean' not in keywords:
        raise ValueError('cols_for_mean missing!')
    if 'partitions' in keywords:
        if 'x_axis_column' in keywords:
            individual_grps = keywords['partitions'] + keywords['x_axis_column']
            in_type = 2
        elif 'xy_axis_columns' in keywords:
            individual_grps = keywords['partitions'] + keywords['xy_axis_columns']
            in_type = 3
        else:
            raise ValueError('Either x_axis_column or xy_axis_columns must be provided')
    elif 'x_axis_column' in keywords:
        individual_grps = keywords['x_axis_column']
        in_type = 0
    elif 'xy_axis_columns' in keywords:
        individual_grps = keywords['xy_axis_columns']
        in_type = 1
    else:
        raise ValueError('Either x_axis_column or xy_axis_columns and it_parameters must be provided')

    partitions = [a for a in individual_grps if a != keywords['used_x_axis'] and a not in keywords['cols_for_mean']]
    from statistics_and_plot import replaceCSVLabels, \
        handle_nans, \
        short_concat_str, \
        capitalizeFirstChar, \
        strToLower, \
        get_limits_log_exp, \
        split_large_labels, \
        split_large_titles, \
        check_legend_enlarge, \
        enl_space_title, \
        calcNrLegendCols, \
        insert_str_option_values, \
        check_file_exists_rename
    if in_type == 0:
        from usac_eval import pars_calc_single_fig
        ret = pars_calc_single_fig(**keywords)
        b = ret['b'].reset_index()
    elif in_type == 1:
        from usac_eval import pars_calc_multiple_fig
        ret = pars_calc_multiple_fig(**keywords)
        ret['b'] = ret['b'].set_index(list(ret['b'].columns.values)[0:2])
        drop_cols = [a for a in list(ret['b'].columns.values)
                     if 'nr_rep_for_pgf_x' == a or 'nr_rep_for_pgf_y' == a or '_lbl' in a]
        if drop_cols:
            ret['b'].drop(drop_cols, axis=1, inplace=True)
        b = ret['b'].reset_index()
    elif in_type == 2:
        from usac_eval import pars_calc_single_fig_partitions
        ret = pars_calc_single_fig_partitions(**keywords)
        ret['sub_title'] = ret['sub_title_it_pars']
        ret['b'] = ret['b'].stack().reset_index()
        ret['b'].rename(columns={ret['b'].columns[-1]: 'Rt_diff'}, inplace=True)
        ret['b'].set_index(keywords['it_parameters'], inplace=True)
        if len(keywords['it_parameters']) > 1:
            cols_idx = ['-'.join(map(str, a)) for a in ret['b'].index]
            ret['b'].index = cols_idx
            idx_name = '-'.join(keywords['it_parameters'])
            ret['b'].index.name = idx_name
        else:
            idx_name = keywords['it_parameters'][0]
            cols_idx = list(ret['b'].index.values)
        ret['b'] = ret['b'].reset_index().set_index(keywords['partitions'] + keywords['x_axis_column'] + [idx_name])
        ret['b'] = ret['b'].unstack()
        ret['b'].columns = ret['b'].columns.get_level_values(1)
        b = ret['b'].reset_index()
    elif in_type == 3:
        from usac_eval import pars_calc_multiple_fig_partitions
        ret = pars_calc_multiple_fig_partitions(**keywords)
        ret['sub_title'] = ret['sub_title_it_pars']
        ret['b'] = ret['b'].stack().reset_index()
        ret['b'].rename(columns={ret['b'].columns[-1]: 'Rt_diff'}, inplace=True)
        ret['b'].set_index(keywords['it_parameters'], inplace=True)
        if len(keywords['it_parameters']) > 1:
            cols_idx = ['-'.join(map(str, a)) for a in ret['b'].index]
            ret['b'].index = cols_idx
            idx_name = '-'.join(keywords['it_parameters'])
            ret['b'].index.name = idx_name
        else:
            idx_name = keywords['it_parameters'][0]
            cols_idx = list(ret['b'].index.values)
        ret['b'] = ret['b'].reset_index().set_index(keywords['partitions'] + keywords['xy_axis_columns'] + [idx_name])
        ret['b'] = ret['b'].unstack()
        ret['b'].columns = ret['b'].columns.get_level_values(1)
        b = ret['b'].reset_index()

    title = 'Mean Combined R \\& t Errors vs ' + replaceCSVLabels(keywords['used_x_axis'], True, True, True) + \
            ' for Parameter Variations of ' + ret['sub_title']
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
    base_name = 'mean_Rt_error_vs_' + keywords['used_x_axis'] + '_pars_' + \
                short_concat_str(keywords['it_parameters'])

    if partitions:
        df_grp = b.drop(keywords['cols_for_mean'], axis=1).groupby(partitions)
        grp_keys = df_grp.groups.keys()
    else:
        df_grp = b.drop(keywords['cols_for_mean'], axis=1)
        grp_keys = [None]
    for grp in grp_keys:
        if grp is not None:
            df = df_grp.get_group(grp).copy(deep=True)
            base_name1 = base_name + '_part_' + \
                         '-'.join([a[0:min(len(a), 6)] + str(b) for a, b in zip(partitions, grp)])
        else:
            df = df_grp
            base_name1 = base_name
        df = df.groupby(keywords['used_x_axis']).mean()

        b_mean_name = 'data_' + base_name1 + '.csv'
        fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
        fb_mean_name = check_file_exists_rename(fb_mean_name)
        b_mean_name = os.path.basename(fb_mean_name)
        with open(fb_mean_name, 'a') as f:
            f.write('# Combined R & t errors vs ' + keywords['used_x_axis'] + '\n')
            if grp is not None:
                f.write('# Data partition: ' + '-'.join([a + str(b) for a, b in zip(partitions, grp)]) + '\n')
            f.write('# Parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
            df.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        section_name = capitalizeFirstChar(strToLower(title))
        caption = 'Mean combined R \\& t Errors vs ' + \
                  replaceCSVLabels(keywords['used_x_axis'], True, False, True)
        if grp is not None:
            section_name += ' for data partition ' + insert_str_option_values(partitions, grp)
            caption += ' for data partition ' + insert_str_option_values(partitions, grp)
        caption += '. Legend: ' + ' -- '.join([replaceCSVLabels(a, False, False, True)
                                               for a in keywords['it_parameters']])
        _, use_limits, use_log, exp_value = get_limits_log_exp(df, True, True, False)
        is_numeric = pd.to_numeric(df.reset_index()[keywords['used_x_axis']], errors='coerce').notnull().all()
        mat_si = df.shape
        nr_bars = mat_si[0] * mat_si[1]
        if nr_bars > 100:
            fig_type = 'sharp plot'
        elif nr_bars > 60:
            fig_type = 'smooth'
        elif nr_bars > 24:
            fig_type = 'xbar'
        else:
            fig_type = 'ybar'
        x_rows = handle_nans(df, list(df.columns.values), not is_numeric, fig_type)
        label_x = replaceCSVLabels(keywords['used_x_axis'])
        label_x, _ = split_large_labels(df, keywords['used_x_axis'], len(df.columns.values), fig_type, False, label_x)
        section_name = split_large_titles(section_name, 80)
        enlarge_lbl_dist = check_legend_enlarge(df, keywords['used_x_axis'], len(df.columns.values), fig_type,
                                                label_x.count('\\') + 1, not is_numeric)
        exp_value = enl_space_title(exp_value, section_name, df, keywords['used_x_axis'],
                                    len(df.columns.values), fig_type)

        tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                      'name': section_name.replace('\\\\', ' '),
                                      'title': section_name,
                                      'title_rows': section_name.count('\\\\'),
                                      'fig_type': fig_type,
                                      'plots': df.columns.values,
                                      'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                      # Label/column name of axis with bars. For xbar it labels the y-axis
                                      'label_x': label_x,
                                      # Column name of axis with bars. For xbar it is the column for the y-axis
                                      'print_x': keywords['used_x_axis'],
                                      # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                      'print_meta': False,
                                      'plot_meta': None,
                                      # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                      'rotate_meta': 0,
                                      'limits': None,
                                      # If None, no legend is used, otherwise use a list
                                      'legend': [' -- '.join([replaceCSVLabels(b) for b in a.split('-')]) for a in
                                                 df.columns.values],
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      # The x/y-axis values are given as strings if True
                                      'use_string_labels': not is_numeric,
                                      'use_log_y_axis': use_log,
                                      'xaxis_txt_rows': 1,
                                      'enlarge_lbl_dist': enlarge_lbl_dist,
                                      'enlarge_title_space': exp_value,
                                      'large_meta_space_needed': False,
                                      'is_neg': False,
                                      'nr_x_if_nan': x_rows,
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
    pdf_name = base_name + '.pdf'
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
    return ret['res']


def filter_calc_t_all_rt_change_type(**vars):
    from robustness_eval import get_rt_change_type
    data = get_rt_change_type(**vars)
    tmp_usac = data.loc[data['stereoRef'] == 'disabled']
    tmp_auto = data.loc[data['stereoRef'] == 'enabled']
    tmp_usac_li = tmp_usac.loc[(tmp_usac['linRefinement_us'] > 0)]
    if not tmp_usac_li.empty:
        tmp_usac = tmp_usac_li.copy(deep=True)
        linref = tmp_usac['linRefinement_us']
    else:
        linref = None
    tmp_usac_ba = tmp_usac.loc[(tmp_usac['bundleAdjust_us'] > 0)]
    if not tmp_usac_ba.empty:
        tmp_usac = tmp_usac_ba.copy(deep=True)
        ba = tmp_usac['bundleAdjust_us']
    else:
        ba = None
    tmp_usac_sac = tmp_usac.loc[(tmp_usac['robEstimationAndRef_us'] > 0)]
    if not tmp_usac_sac.empty:
        tmp_usac = tmp_usac_sac.copy(deep=True)
        sac = tmp_usac['robEstimationAndRef_us']
    else:
        sac = None
    if linref is not None and ba is not None and sac is not None:
        tmp_usac['exec_time'] = linref + ba + sac
    elif linref is not None and ba is not None:
        tmp_usac['exec_time'] = linref + ba
    elif linref is not None and sac is not None:
        tmp_usac['exec_time'] = linref + sac
    elif ba is not None and sac is not None:
        tmp_usac['exec_time'] = ba + sac
    elif linref is not None:
        tmp_usac['exec_time'] = linref
    elif ba is not None:
        tmp_usac['exec_time'] = ba
    elif sac is not None:
        tmp_usac['exec_time'] = sac
    else:
        raise ValueError('No valid timestamps found for execution of USAC')

    tmp_auto = tmp_auto.loc[(tmp_auto['stereoRefine_us'] > 0)]
    if tmp_auto.empty:
        raise ValueError('No valid timestamps found for stereo refinement')
    tmp_auto['exec_time'] = tmp_auto['stereoRefine_us'].to_list()
    tmp = pd.concat([tmp_usac, tmp_auto], ignore_index=False, axis=0)
    return tmp


def estimate_alg_time_fixed_kp(**keywords):
    if 'partitions' in keywords:
        if 'x_axis_column' in keywords:
            individual_grps = keywords['partitions'] + keywords['x_axis_column']
        elif 'xy_axis_columns' in keywords:
            individual_grps = keywords['partitions'] + keywords['xy_axis_columns']
        else:
            raise ValueError('Either x_axis_column or xy_axis_columns must be provided')
    elif 'x_axis_column' in keywords:
        individual_grps = keywords['x_axis_column']
    elif 'xy_axis_columns' in keywords:
        individual_grps = keywords['xy_axis_columns']
    else:
        raise ValueError('Either x_axis_column or xy_axis_columns and it_parameters must be provided')
    if 't_data_separators' not in keywords:
        no_sep = True
    else:
        no_sep = False
    from statistics_and_plot import tex_string_coding_style, \
        compile_tex, \
        calcNrLegendCols, \
        replaceCSVLabels, \
        strToLower, \
        split_large_titles, \
        get_limits_log_exp, \
        enl_space_title, \
        short_concat_str, \
        split_large_str, \
        check_legend_enlarge, \
        check_file_exists_rename
    tmp, col_name = get_time_fixed_kp(**keywords)

    keywords = prepare_io(**keywords)

    tmp.set_index(keywords['it_parameters'], inplace=True)
    tmp = tmp.T
    from statistics_and_plot import glossary_from_list, add_to_glossary, add_to_glossary_eval, handle_nans
    if len(keywords['it_parameters']) > 1:
        gloss = glossary_from_list([str(b) for a in tmp.columns for b in a])
        par_cols = ['-'.join(map(str, a)) for a in tmp.columns]
        tmp.columns = par_cols
        it_pars_cols_name = '-'.join(map(str, keywords['it_parameters']))
        tmp.columns.name = it_pars_cols_name
    else:
        gloss = glossary_from_list([str(a) for a in tmp.columns])
        it_pars_cols_name = keywords['it_parameters'][0]
    gloss = add_to_glossary_eval(individual_grps, gloss)
    if no_sep:
        xaxis = ['options_tex']
    else:
        xaxis = individual_grps

    title = 'Mean Execution Times for Parameter Variations of ' + keywords['sub_title_it_pars'] + \
            ' Extrapolated for ' + str(int(keywords['nr_target_kps'])) + ' Keypoints'
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': False,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': gloss
                 }
    for col in xaxis:
        if no_sep:
            tmp1 = tmp.T.reset_index()
            tmp1[col] = [split_large_str(tex_string_coding_style(val))
                         for _, val in tmp1[it_pars_cols_name].iteritems()]
            max_txt_rows = 1
            for idx, val in tmp1[col].iteritems():
                txt_rows = str(val).count('\\\\') + 1
                if txt_rows > max_txt_rows:
                    max_txt_rows = txt_rows
            tmp1.set_index(col, inplace=True)
            tmp1.drop(it_pars_cols_name, axis=1, inplace=True)
            label_x = 'options'
            legend = None
        else:
            if len(xaxis) > 1:
                drop_cols = [a for a in xaxis if a != col]
                tmp1 = tmp.T.reset_index().drop(drop_cols, axis=1).groupby([col, it_pars_cols_name]).mean().unstack()
            else:
                tmp1 = tmp.T.reset_index().set_index([col, it_pars_cols_name]).unstack()
            tmp1.columns = [h for g in tmp1.columns for h in g if h != col_name]
            label_x = replaceCSVLabels(str(col))
            legend = [tex_string_coding_style(a) for a in list(tmp1.columns.values)]
            max_txt_rows = 1
        t_main_name = 'mean_time_for_' + \
                      str(int(keywords['nr_target_kps'])) + 'kpts_vs_' + str(col) + '_for_opts_' + \
                      short_concat_str(keywords['it_parameters'])
        t_mean_name = 'data_' + t_main_name + '.csv'
        ft_mean_name = os.path.join(keywords['tdata_folder'], t_mean_name)
        ft_mean_name = check_file_exists_rename(ft_mean_name)
        t_mean_name = os.path.basename(ft_mean_name)
        with open(ft_mean_name, 'a') as f:
            f.write('# Mean execution times extrapolated for ' +
                    str(int(keywords['nr_target_kps'])) + ' keypoints vs ' + str(col) + '\n')
            f.write('# Row (column options) parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
            tmp1.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        _, use_limits, use_log, exp_value = get_limits_log_exp(tmp1)
        is_numeric = pd.to_numeric(tmp1.reset_index()[col], errors='coerce').notnull().all()
        reltex_name = os.path.join(keywords['rel_data_path'], t_mean_name)
        fig_name = 'Mean execution times vs ' + replaceCSVLabels(col, False, False, True) + \
                   ' for parameter variations of\\\\' + \
                   strToLower(keywords['sub_title_it_pars']) + \
                   '\\\\extrapolated for ' + str(int(keywords['nr_target_kps'])) + ' keypoints'
        fig_name = split_large_titles(fig_name)
        exp_value = enl_space_title(exp_value, fig_name, tmp1, col,
                                    len(list(tmp1.columns.values)), 'ybar')
        enlarge_lbl_dist = check_legend_enlarge(tmp1, col, len(list(tmp1.columns.values)), 'ybar')
        x_rows = handle_nans(tmp1, list(tmp1.columns.values), not is_numeric, 'ybar')
        tex_infos['sections'].append({'file': reltex_name,
                                      'name': fig_name,
                                      # If caption is None, the field name is used
                                      'caption': fig_name.replace('\\\\', ' '),
                                      'fig_type': 'ybar',
                                      'plots': list(tmp1.columns.values),
                                      'label_y': 'mean execution times/$\\mu s$',
                                      'plot_x': str(col),
                                      'label_x': label_x,
                                      'limits': use_limits,
                                      'legend': legend,
                                      'legend_cols': None,
                                      'use_marks': False,
                                      'use_log_y_axis': use_log,
                                      'xaxis_txt_rows': max_txt_rows,
                                      'nr_x_if_nan': x_rows,
                                      'enlarge_lbl_dist': enlarge_lbl_dist,
                                      'enlarge_title_space': exp_value,
                                      'use_string_labels': not is_numeric,
                                      })
        if not no_sep:
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    t_main_name = 'mean_time_for_' + \
                  str(int(keywords['nr_target_kps'])) + 'kpts_vs_' + short_concat_str(map(str, xaxis)) + '_for_opts_' + \
                  short_concat_str(keywords['it_parameters'])
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    if keywords['build_pdf'][0]:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex,
                              keywords['tex_folder'],
                              texf_name,
                              False,
                              os.path.join(keywords['pdf_folder'], pdf_name),
                              tex_infos['figs_externalize']))
    else:
        res = abs(compile_tex(rendered_tex, keywords['tex_folder'], texf_name, False))
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)
    return res


def accum_corrs_sequs_time_model(**keywords):
    keywords = get_accum_corrs_sequs(**keywords)
    keywords['data_separators'] = keywords['t_data_separators']
    from usac_eval import calc_Time_Model
    keywords = calc_Time_Model(**keywords)
    return keywords