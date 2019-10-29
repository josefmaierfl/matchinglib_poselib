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

def get_rt_change_type(**keywords):
    if 'data_seperators' not in keywords:
        raise ValueError('it_parameters missing!')

    df_grp = keywords['data'].groupby(keywords['data_seperators'])
    grp_keys = df_grp.groups.keys()
    df_list = []
    change_j_occ = {'jrt': 0, 'jra': 0, 'jta': 0, 'jrx': 0, 'jry': 0, 'jrz': 0, 'jtx': 0, 'jty': 0, 'jtz': 0}
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
            else:
                rxc = tmp1['R_GT_n_diff_roll_deg'].fillna(0).abs().sum() > 1e-3
                ryc = tmp1['R_GT_n_diff_pitch_deg'].fillna(0).abs().sum() > 1e-3
                rzc = tmp1['R_GT_n_diff_yaw_deg'].fillna(0).abs().sum() > 1e-3
                txc = tmp1['t_GT_n_elemDiff_tx'].fillna(0).abs().sum() > 1e-4
                tyc = tmp1['t_GT_n_elemDiff_ty'].fillna(0).abs().sum() > 1e-4
                tzc = tmp1['t_GT_n_elemDiff_tz'].fillna(0).abs().sum() > 1e-4
                if rxc and ryc and rzc and txc and tyc and tzc:
                    tmp1['rt_change_type'] = ['jrt'] * int(tmp1.shape[0])
                    change_j_occ['jrt'] += 1
                elif rxc and ryc and rzc:
                    tmp1['rt_change_type'] = ['jra'] * int(tmp1.shape[0])
                    change_j_occ['jra'] += 1
                elif txc and tyc and tzc:
                    tmp1['rt_change_type'] = ['jta'] * int(tmp1.shape[0])
                    change_j_occ['jta'] += 1
                elif rxc:
                    tmp1['rt_change_type'] = ['jrx'] * int(tmp1.shape[0])
                    change_j_occ['jrx'] += 1
                elif ryc:
                    tmp1['rt_change_type'] = ['jry'] * int(tmp1.shape[0])
                    change_j_occ['jry'] += 1
                elif rzc:
                    tmp1['rt_change_type'] = ['jrz'] * int(tmp1.shape[0])
                    change_j_occ['jrz'] += 1
                elif txc:
                    tmp1['rt_change_type'] = ['jtx'] * int(tmp1.shape[0])
                    change_j_occ['jtx'] += 1
                elif tyc:
                    tmp1['rt_change_type'] = ['jty'] * int(tmp1.shape[0])
                    change_j_occ['jty'] += 1
                elif tzc:
                    tmp1['rt_change_type'] = ['jtz'] * int(tmp1.shape[0])
                    change_j_occ['jtz'] += 1
                else:
                    max_val = change_j_occ[max(change_j_occ, key=(lambda key: change_j_occ[key]))]
                    min_val = max_val
                    min_key = 'nv'
                    for key, value in change_j_occ.items():
                        if value > 0 and value < min_val:
                            min_key = key
                        elif value > 0 and value == min_val:
                            min_key = 'nv'
                    tmp1['rt_change_type'] = [min_key] * int(tmp1.shape[0])# no variation
            df_list.append(tmp1)
    df_new = pd.concat(df_list, axis=0, ignore_index=False)
    if 'filter_scene' in keywords:
        df_new = df_new.loc[df_new['rt_change_type'].str.contains(keywords['filter_scene'], regex=False)]
    return df_new


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
        check_if_series, \
        strToLower
    ret = pars_calc_single_fig_partitions(**keywords)
    b_min = ret['b'].stack().reset_index()
    b_min.rename(columns={b_min.columns[-1]: 'b_min'}, inplace=True)
    b_min = b_min.loc[b_min.groupby(ret['partitions'] + keywords['x_axis_column'])['b_min'].idxmin()]
    b_min1 = b_min.set_index(ret['it_parameters'])
    if len(ret['it_parameters']) > 1:
        b_min1.index = ['-'.join(map(str, a)) for a in b_min1.index]
        it_pars_name = '-'.join(map(str, ret['it_parameters']))
        b_min1.index.name = it_pars_name
    else:
        it_pars_name = ret['it_parameters'][0]
    b_min_grp = b_min1.reset_index().set_index(keywords['x_axis_column']).groupby(ret['partitions'])
    grp_keys = b_min_grp.groups.keys()
    base_name = 'min_RTerrors_vs_' + keywords['x_axis_column'][0] + '_and_corresp_opts_' + \
                short_concat_str(ret['it_parameters']) + '_for_part_' + ret['dataf_name_partition']
    tex_infos = {'title': 'Smallest Combined R \\& t Errors and Their Corresponding ' + \
                          ' Parameters ' + ret['sub_title_it_pars'] + \
                          ' vs ' + replaceCSVLabels(keywords['x_axis_column'][0], True, True, True) + \
                          ' for Different ' + ret['sub_title_partitions'],
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
        tmp['options_tex'] = [', '.join(['{:.3f}'.format(float(b)) for b in a.split('-')])
                              for a in tmp[it_pars_name].values]
        b_mean_name = 'data_' + base_name + \
                      (str(grp) if len(ret['partitions']) == 1 else '-'.join(map(str, grp))) + '.csv'
        fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
        with open(fb_mean_name, 'a') as f:
            f.write('# Minimum combined R & t errors (b_min) and corresponding option vs ' +
                    keywords['x_axis_column'][0] +
                    ' for partition ' + '-'.join(map(str, ret['partitions'])) + ' = ' +
                    (str(grp) if len(ret['partitions']) == 1 else '-'.join(map(str, grp))) + '\n')
            f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
            tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        section_name = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ and their ' + \
                       ' corresponding options vs ' + \
                       replaceCSVLabels(keywords['x_axis_column'][0], False, False, True) + \
                       ' for property ' + insert_str_option_values(ret['partitions'], grp)
        caption = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ and their ' + \
                  ' corresponding options on top of each bar separated by a comma in the order ' + \
                  strToLower(ret['sub_title_it_pars']) + ' vs ' + \
                  replaceCSVLabels(keywords['x_axis_column'][0], False, False, True) + \
                  ' for the ' + insert_str_option_values(ret['partitions'], grp)
        _, use_limits, use_log, exp_value = get_limits_log_exp(tmp, True, True, False, ['options_tex', it_pars_name])
        is_numeric = pd.to_numeric(tmp.reset_index()[keywords['x_axis_column'][0]], errors='coerce').notnull().all()
        section_name = split_large_titles(section_name)
        exp_value = enl_space_title(exp_value, section_name, tmp, keywords['x_axis_column'],
                                    1, 'ybar')
        tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                      'name': section_name.replace('\\\\', ' '),
                                      'title': section_name,
                                      'title_rows': section_name.count('\\\\'),
                                      'fig_type': 'ybar',
                                      'plots': ['b_min'],
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
    if ret['build_pdf'][0]:
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
        f.write('# Statistic over minimum combined R & t errors (b_min) and statistic of corresponding parameters ' +
                ' over all ' + keywords['x_axis_column'][0] + ' for different ' +
                '-'.join(map(str, ret['partitions'])) + '\n')
        f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
        par_stat.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    b_mean = par_stat.xs('mean', axis=1, level=1, drop_level=True).copy(deep=True)
    b_mean['options_tex'] = [', '.join(['{:.3f}'.format(float(val)) for _, val in row.iteritems()])
                             for _, row in b_mean[ret['it_parameters']].iterrows()]

    base_name = 'mean_min_RTerrors_and_mean_corresp_opts_' + \
                short_concat_str(ret['it_parameters']) + '_for_part_' + ret['dataf_name_partition']
    tex_infos = {'title': 'Mean Values of Smallest Combined R \\& t Errors and Their Corresponding ' + \
                          ' Mean Parameters ' + ret['sub_title_it_pars'] + \
                          ' over all ' + replaceCSVLabels(keywords['x_axis_column'][0], True, True, True) + \
                          ' for Different ' + ret['sub_title_partitions'],
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
        f.write('# Mean values over minimum combined R & t errors (b_min) and corresponding mean parameters ' +
                ' over all ' + keywords['x_axis_column'][0] + ' for different ' +
                '-'.join(map(str, ret['partitions'])) + '\n')
        f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
        b_mean.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    section_name = 'Mean values of smallest combined R \\& t errors and their corresponding\\\\' + \
                   ' mean parameters ' + strToLower(ret['sub_title_it_pars']) + \
                   ' over all ' + replaceCSVLabels(keywords['x_axis_column'][0], True, False, True) + \
                   ' for different ' + strToLower(ret['sub_title_partitions'])
    caption = 'Mean values of smallest combined R \\& t errors and their ' + \
              ' corresponding mean parameters on top of each bar separated by a comma in the order ' + \
              strToLower(ret['sub_title_it_pars']) + ' over all ' + \
              replaceCSVLabels(keywords['x_axis_column'][0], True, False, True)
    _, use_limits, use_log, exp_value = get_limits_log_exp(b_mean, True, True, False, ['options_tex'] +
                                                           ret['it_parameters'])
    is_numeric = pd.to_numeric(b_mean.reset_index()[ret['partitions'][0]], errors='coerce').notnull().all()
    section_name = split_large_titles(section_name)
    exp_value = enl_space_title(exp_value, section_name, b_mean, ret['partitions'][0],
                                1, 'ybar')
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'ybar',
                                  'plots': ['b_min'],
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
                                         'mean_Rt_error': float(b_mmean['b_min'])}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    return ret['res']


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
        check_if_series, \
        strToLower, \
        check_legend_enlarge, \
        calcNrLegendCols
    ret = pars_calc_multiple_fig_partitions(**keywords)
    b_min = ret['b'].stack().reset_index()
    b_min.rename(columns={b_min.columns[-1]: 'b_min'}, inplace=True)
    b_min = b_min.loc[b_min.groupby(ret['partitions'] + keywords['xy_axis_columns'])['b_min'].idxmin()]
    b_min1 = b_min.set_index(ret['it_parameters'])
    if len(ret['it_parameters']) > 1:
        b_min1.index = ['-'.join(map(str, a)) for a in b_min1.index]
        it_pars_name = '-'.join(map(str, ret['it_parameters']))
        b_min1.index.name = it_pars_name
    else:
        it_pars_name = ret['it_parameters'][0]
    b_min_grp = b_min1.reset_index().set_index(keywords['xy_axis_columns']).groupby(ret['partitions'])
    grp_keys = b_min_grp.groups.keys()
    base_name = 'min_RTerrors_vs_' + keywords['xy_axis_columns'][0] + '_and_' + \
                keywords['xy_axis_columns'][1] + '_with_corresp_opts_' + \
                short_concat_str(ret['it_parameters']) + '_for_part_' + ret['dataf_name_partition']
    tex_infos = {'title': 'Smallest Combined R \\& t Errors and Their Corresponding ' + \
                          ' Parameters ' + ret['sub_title_it_pars'] + \
                          ' vs ' + replaceCSVLabels(keywords['xy_axis_columns'][0], True, True, True) + \
                          ' and ' + replaceCSVLabels(keywords['xy_axis_columns'][1], True, True, True) + \
                          ' for Different ' + ret['sub_title_partitions'],
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
        tmp['options_tex'] = [', '.join(['{:.3f}'.format(float(b)) for b in a.split('-')])
                              for a in tmp[it_pars_name].values]
        # tmp = tmp.reset_index().set_index([it_pars_name] + keywords['xy_axis_columns'])
        tmp = tmp.unstack()
        tmp.columns = ['-'.join(map(str, a)) for a in tmp.columns]
        # tmp = tmp.reset_index().set_index(keywords['xy_axis_columns'][0])
        plots = [a for a in tmp.columns if 'b_min' in a]
        meta_cols = [a for a in tmp.columns if 'options_tex' in a]
        it_pars_names = [a for a in tmp.columns if it_pars_name in a]
        b_mean_name = 'data_' + base_name + \
                      (str(grp) if len(ret['partitions']) == 1 else '-'.join(map(str, grp))) + '.csv'
        fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
        with open(fb_mean_name, 'a') as f:
            f.write('# Minimum combined R & t errors (b_min) and corresponding option vs ' +
                    keywords['xy_axis_columns'][0] + ' and ' + keywords['xy_axis_columns'][1] +
                    ' for partition ' + '-'.join(map(str, ret['partitions'])) + ' = ' +
                    (str(grp) if len(ret['partitions']) == 1 else '-'.join(map(str, grp))) + '\n')
            f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
            tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        section_name = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ and their ' + \
                       'corresponding options vs ' + \
                       replaceCSVLabels(keywords['xy_axis_columns'][0], False, False, True) + ' and ' + \
                       replaceCSVLabels(keywords['xy_axis_columns'][1], False, False, True) + \
                       ' for property ' + insert_str_option_values(ret['partitions'], grp)
        caption = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ and their ' + \
                  'corresponding options on top of each bar separated by a comma in the order ' + \
                  strToLower(ret['sub_title_it_pars']) + ' vs ' + \
                  replaceCSVLabels(keywords['xy_axis_columns'][0], False, False, True) + ' and ' + \
                  replaceCSVLabels(keywords['xy_axis_columns'][1], False, False, True) + \
                  ' for the ' + insert_str_option_values(ret['partitions'], grp)
        _, use_limits, use_log, exp_value = get_limits_log_exp(tmp, True, True, False, it_pars_names + meta_cols)
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
                                      'rotate_meta': 90,
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
    if ret['build_pdf'][0]:
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
        f.write('# Statistic over minimum combined R & t errors (b_min) and statistic of corresponding parameters ' +
                ' over all ' + keywords['x_axis_column'][0] + ' for different ' +
                '-'.join(map(str, ret['partitions'])) + '\n')
        f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
        par_stat.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    b_mean = par_stat.xs('mean', axis=1, level=1, drop_level=True).copy(deep=True)
    b_mean['options_tex'] = [', '.join(['{:.3f}'.format(float(val)) for _, val in row.iteritems()])
                             for _, row in b_mean[ret['it_parameters']].iterrows()]

    base_name = 'mean_min_RTerrors_and_mean_corresp_opts_' + \
                short_concat_str(ret['it_parameters']) + '_for_part_' + ret['dataf_name_partition']
    tex_infos = {'title': 'Mean Values of Smallest Combined R \\& t Errors and Their Corresponding ' + \
                          ' Mean Parameters ' + ret['sub_title_it_pars'] + \
                          ' over all ' + replaceCSVLabels(keywords['x_axis_column'][0], True, True, True) + \
                          ' for Different ' + ret['sub_title_partitions'],
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
        f.write('# Mean values over minimum combined R & t errors (b_min) and corresponding mean parameters ' +
                ' over all ' + keywords['x_axis_column'][0] + ' for different ' +
                '-'.join(map(str, ret['partitions'])) + '\n')
        f.write('# Parameters: ' + '-'.join(ret['it_parameters']) + '\n')
        b_mean.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    section_name = 'Mean values of smallest combined R \\& t errors and their corresponding\\\\' + \
                   ' mean parameters ' + strToLower(ret['sub_title_it_pars']) + \
                   ' over all ' + replaceCSVLabels(keywords['x_axis_column'][0], True, False, True) + \
                   ' for different ' + strToLower(ret['sub_title_partitions'])
    caption = 'Mean values of smallest combined R \\& t errors and their ' + \
              ' corresponding mean parameters on top of each bar separated by a comma in the order ' + \
              strToLower(ret['sub_title_it_pars']) + ' over all ' + \
              replaceCSVLabels(keywords['x_axis_column'][0], True, False, True)
    _, use_limits, use_log, exp_value = get_limits_log_exp(b_mean, True, True, False, ['options_tex'] +
                                                           ret['it_parameters'])
    is_numeric = pd.to_numeric(b_mean.reset_index()[ret['partitions'][0]], errors='coerce').notnull().all()
    section_name = split_large_titles(section_name)
    exp_value = enl_space_title(exp_value, section_name, b_mean, ret['partitions'][0],
                                1, 'ybar')
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'ybar',
                                  'plots': ['b_min'],
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
                                         'mean_Rt_error': float(b_mmean['b_min'])}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    return ret['res']