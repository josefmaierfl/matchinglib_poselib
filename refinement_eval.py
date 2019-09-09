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
from usac_eval import ji_env

def get_best_comb_scenes_1(**keywords):
    from usac_eval import pars_calc_single_fig_partitions
    ret = pars_calc_single_fig_partitions(**keywords)
    # if len(ret['partitions']) != 2:
    #     raise ValueError('Only a number of 2 partitions is allowed in function get_best_comb_scenes_1')
    b_new_index = ret['b'].reset_index()
    if len(ret['it_parameters'])  > 1:
        it_pars_name = '-'.join(ret['it_parameters'])
    else:
        it_pars_name = ret['it_parameters'][0]
    b_mean_l = []
    for i, val in enumerate(ret['partitions']):
        drops = [val1 for i1, val1 in enumerate(ret['partitions']) if i1 != i]
        drops += [ret['grp_names'][-1]]
        b_mean_l.append(b_new_index.drop(drops, axis=1).groupby(ret['it_parameters'] + [val]).mean())
    b_mean_l.append(b_new_index.drop(ret['partitions'], axis=1).groupby(ret['it_parameters'] +
                                                                        ret['grp_names'][-1]).mean())
    data_parts = ret['partitions'] + [ret['grp_names'][-1]]
    data_it_indices = []
    for i, df, dp in enumerate(zip(b_mean_l, data_parts)):
        tmp = df.unstack(level=-1)
        if len(ret['it_parameters']) > 1:
            data_it_indices.append('-'.join(a) for a in tmp.index)
            tmp.index = data_it_indices[-1]
            tmp.index.name = it_pars_name
        else:
            data_it_indices.append(str(a) for a in tmp.index)
        tmp = tmp.T.reset_index()



    b_mean = ret['b'].mean(axis=1)
    b_mean.rename('b_min', inplace=True)
    tmp1 = b_mean.reset_index()
    tmp2 = tmp1.loc[tmp1.groupby(ret['partitions'][:-1] + ret['it_parameters'])['b_min'].idxmin(axis=0)]
    tmp2 = tmp2.set_index(ret['it_parameters'])
    tmp2.index = ['-'.join(map(str, a)) for a in tmp2.index]
    it_pars_ov = '-'.join(ret['it_parameters'])
    tmp2.index.name = it_pars_ov
    tmp2 = tmp2.reset_index().set_index(ret['partitions'][:-1])
    tmp2.index = ['-'.join(map(str, a)) for a in tmp2.index]
    partitions_ov = '-'.join(ret['partitions'][:-1])
    from statistics_and_plot import replaceCSVLabels
    partitions_legend = ' -- '.join([replaceCSVLabels(i) for i in ret['partitions'][:-1]])
    tmp2.index.name = partitions_ov
    tmp2 = tmp2.reset_index().set_index([it_pars_ov, partitions_ov]).unstack(level=0)
    column_legend = [' -- '.join([i if 'b_min' in i else replaceCSVLabels(i) for i in map(str, a)]) for a in tmp2.columns]
    tmp2.columns = ['-'.join(map(str, a)) for a in tmp2.columns]

    b_mean_name = 'data_mean_min_RTerrors_and_corresp_' + ret['partitions'][-1] + '_for_opts_' + \
                  '-'.join(ret['it_parameters']) + '_and_props_' + ret['dataf_name_partition'] + '.csv'
    fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
    with open(fb_mean_name, 'a') as f:
        f.write('# Minimum combined R & t errors (b_min) and corresponding ' + ret['partitions'][-1] +
                ' over all ' + str(ret['grp_names'][-1]) + ' (mean) for options ' +
                '-'.join(ret['it_parameters']) + ' and separately for properties ' +
                '-'.join(map(str, ret['partitions'][:-1])) + '\n')
        f.write('# Row (column options) parameters: (b_min/' + ret['partitions'][-1] + ')-' +
                '-'.join(ret['it_parameters']) + '\n')
        tmp2.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    right_cols = [a for a in tmp2.columns if ret['partitions'][-1] in a]
    left_cols = [a for a in tmp2.columns if 'b_min' in a]
    right_legend = [column_legend[i] for i, a in enumerate(tmp2.columns) if ret['partitions'][-1] in a]
    left_legend = [column_legend[i].replace('b_min', 'R\\&t error') for i, a in enumerate(tmp2.columns) if 'b_min' in a]
    # Get data for tex file generation
    if len(tmp2.columns) > 10:
        fig_type = 'xbar'
    else:
        fig_type = 'ybar'

    tex_infos = {'title': 'Smallest Combined R \\& t Errors and Their ' + \
                          replaceCSVLabels(str(ret['partitions'][-1]), True, True) + \
                          ' for Parameters ' + ret['sub_title_it_pars'] + \
                          ' and Properties ' + ret['sub_title_partitions'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true, non-numeric entries can be provided for the x-axis
                 'nonnumeric_x': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': ret['gloss']
                 }

    section_name = 'Smallest combined R \\& t errors and their ' + \
                   replaceCSVLabels(str(ret['partitions'][-1]), True) + \
                   '\\\\for parameters ' + ret['sub_title_it_pars'] + \
                   '\\\\and properties ' + ret['sub_title_partitions']
    if fig_type == 'xbar':
        caption = 'Smallest combined R \\& t errors (bottom axis) and their ' +\
                  replaceCSVLabels(str(ret['partitions'][-1]), True) +\
                  ' (top axis) for parameters ' + ret['sub_title_it_pars'] +\
                  ' and properties ' + ret['sub_title_partitions'] + '.'
    else:
        caption = 'Smallest combined R \\& t errors (left axis) and their ' + \
                  replaceCSVLabels(str(ret['partitions'][-1]), True) + \
                  ' (right axis) for parameters ' + ret['sub_title_it_pars'] + \
                  ' and properties ' + ret['sub_title_partitions'] + '.'
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                  # Name of the whole section
                                  'name': section_name.replace('\\\\', ' '),
                                  # Title of the figure
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': fig_type,
                                  # Column name for charts based on the left y-axis
                                  'plots_l': left_cols,
                                  # Label of the left y-axis.
                                  'label_y_l': 'error',
                                  # Column name for charts based on the right y-axis
                                  'plots_r': right_cols,
                                  # Label of the right y-axis.
                                  'label_y_r': replaceCSVLabels(str(ret['partitions'][-1])),
                                  # Label of the x-axis.
                                  'label_x': partitions_legend,
                                  # Column name of the x-axis.
                                  'plot_x': partitions_ov,
                                  # Maximum and/or minimum y value/s on the left y-axis
                                  'limits_l': None,
                                  # Legend entries for the charts belonging to the left y-axis
                                  'legend_l': [a + ' (left axis)' if fig_type == 'ybar'
                                               else a + ' (bottom axis)' for a in left_legend],
                                  # Maximum and/or minimum y value/s on the right y-axis
                                  'limits_r': None,
                                  # Legend entries for the charts belonging to the right y-axis
                                  'legend_r': [a + ' (right axis)' if fig_type == 'ybar'
                                               else a + ' (top axis)' for a in right_legend],
                                  'legend_cols': 1,
                                  'use_marks': True,
                                  'caption': caption
                                  })

    for rc, lc, rl, ll in zip(right_cols, left_cols, right_legend, left_legend):
        par_str = [i for i in rl.split(' -- ') if ret['partitions'][-1] not in i][0]
        section_name = 'Smallest combined R \\& t errors and their ' + \
                       replaceCSVLabels(str(ret['partitions'][-1]), True) + \
                       '\\\\for parameters ' + par_str + \
                       '\\\\and properties ' + ret['sub_title_partitions']

        caption = 'Smallest combined R \\& t errors (left axis) and their ' + \
                  replaceCSVLabels(str(ret['partitions'][-1]), True) + \
                  ' (right axis) for parameters ' + par_str + \
                  ' and properties ' + ret['sub_title_partitions'] + '.'
        tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                      # Name of the whole section
                                      'name': section_name.replace('\\\\', ' '),
                                      # Title of the figure
                                      'title': section_name,
                                      'title_rows': section_name.count('\\\\'),
                                      'fig_type': 'ybar',
                                      # Column name for charts based on the left y-axis
                                      'plots_l': [lc],
                                      # Label of the left y-axis.
                                      'label_y_l': 'error',
                                      # Column name for charts based on the right y-axis
                                      'plots_r': [rc],
                                      # Label of the right y-axis.
                                      'label_y_r': replaceCSVLabels(str(ret['partitions'][-1])),
                                      # Label of the x-axis.
                                      'label_x': partitions_legend,
                                      # Column name of the x-axis.
                                      'plot_x': partitions_ov,
                                      # Maximum and/or minimum y value/s on the left y-axis
                                      'limits_l': None,
                                      # Legend entries for the charts belonging to the left y-axis
                                      'legend_l': [ll + ' (left axis)'],
                                      # Maximum and/or minimum y value/s on the right y-axis
                                      'limits_r': None,
                                      # Legend entries for the charts belonging to the right y-axis
                                      'legend_r': [rl + ' (right axis)'],
                                      'legend_cols': 1,
                                      'use_marks': True,
                                      'caption': caption
                                      })

    base_out_name = 'tex_min_mean_RTerrors_and_corresp_' + \
                    str(ret['partitions'][-1]) + '_for_opts_' + \
                    '-'.join(ret['it_parameters']) + '_and_props_' + \
                    ret['dataf_name_partition']
    template = ji_env.get_template('usac-testing_2D_plots_2y_axis.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   nonnumeric_x=tex_infos['nonnumeric_x'],
                                   sections=tex_infos['sections'],
                                   fill_bar=True,
                                   abbreviations=tex_infos['abbreviations'])
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    from statistics_and_plot import compile_tex
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

    tex_infos = {'title': 'Smallest Combined R \\& t Errors ' + \
                          ' for Parameters ' + ret['sub_title_it_pars'] + \
                          ' and Properties ' + ret['sub_title_partitions'],
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
    section_name = 'Smallest combined R \\& t errors ' + \
                   '\\\\for parameters ' + ret['sub_title_it_pars'] + \
                   '\\\\and properties ' + ret['sub_title_partitions']
    caption = 'Smallest combined R \\& t errors and their ' + \
              replaceCSVLabels(str(ret['partitions'][-1]), True) + \
              ' (on top of each bar) for parameters ' + ret['sub_title_it_pars'] + \
              ' and properties ' + ret['sub_title_partitions'] + '.'
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'ybar',
                                  'plots': left_cols,
                                  'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': partitions_legend,
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': partitions_ov,
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': right_cols,
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 45,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': left_legend,
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': True,
                                  'use_log_y_axis': False,
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    base_out_name = 'tex_min_mean_RTerrors_and_corresp_' + \
                    str(ret['partitions'][-1]) + '_as_meta_for_opts_' + \
                    '-'.join(ret['it_parameters']) + '_and_props_' + \
                    ret['dataf_name_partition']
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

    return ret['res']