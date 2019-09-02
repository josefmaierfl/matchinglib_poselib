"""
Calculates the best performin parameter values with given testing data as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np, math
# import modin.pandas as pd
import pandas as pd
#from jinja2 import Template as ji
import jinja2 as ji
import ruamel.yaml as yaml
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from scipy import stats
# import tempfile
# import shutil
#from copy import deepcopy
#import shutil
# import time

# warnings.simplefilter('ignore', category=UserWarning)

ji_env = ji.Environment(
    block_start_string='\BLOCK{',
    block_end_string='}',
    variable_start_string='\VAR{',
    variable_end_string='}',
    comment_start_string='\#{',
    comment_end_string='}',
    line_statement_prefix='%-',
    line_comment_prefix='%#',
    trim_blocks=True,
    autoescape=False,
    loader=ji.FileSystemLoader(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tex_templates')))


class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True

    def increase_indent(self, flow=False, sequence=None, indentless=False):
        return super(NoAliasDumper, self).increase_indent(flow, False)


def readYaml(file):
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)
    with open(file, 'r') as fi:
        data = fi.readlines()
    data = [line for line in data if line[0] is not '%']
    try:
        data = yaml.load_all("\n".join(data))
        data1 = {}
        for d in data:
            data1.update(d)
        data = data1
    except:
        print('Exception during reading yaml.')
        e = sys.exc_info()
        print(str(e))
        sys.stdout.flush()
        raise BaseException
    return data


def pars_calc_single_fig_partitions(**keywords):
    if len(keywords) < 3 or len(keywords) > 7:
        raise ValueError('Wrong number of arguments for function pars_calc_single_fig_partitions')
    if 'data' not in keywords:
        raise ValueError('Missing data argument of function pars_calc_single_fig_partitions')
    if 'partitions' not in keywords:
        raise ValueError('Missing partitions argument of function pars_calc_single_fig_partitions')
    data = keywords['data']
    ret = {}
    ret['partitions'] = keywords['partitions']
    if 'res_folder' not in keywords:
        raise ValueError('Missing res_folder argument of function pars_calc_single_fig_partitions')
    ret['res_folder'] = keywords['res_folder']
    ret['use_marks'] = False
    if 'use_marks' not in keywords:
        print('No information provided if marks should be used: Disabling marks')
    else:
        ret['use_marks'] = keywords['use_marks']
    ret['build_pdf'] = (False, True,)
    if 'build_pdf' in keywords:
        ret['build_pdf'] = keywords['build_pdf']
    if len(ret['build_pdf']) != 2:
        raise ValueError('Wrong number of arguments for build_pdf')
    ret['pdf_folder'] = None
    if ret['build_pdf'][0] or ret['build_pdf'][1]:
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
    nr_partitions = len(ret['partitions'])
    ret['it_parameters'] = ret['grp_names'][nr_partitions:-1]
    nr_it_parameters = len(ret['it_parameters'])
    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols, replaceCSVLabels
    ret['sub_title_it_pars'] = ''
    for i, val in enumerate(ret['it_parameters']):
        ret['sub_title_it_pars'] += replaceCSVLabels(val, True, True)
        if nr_it_parameters <= 2:
            if i < nr_it_parameters - 1:
                ret['sub_title_it_pars'] += ' and '
        else:
            if i < nr_it_parameters - 2:
                ret['sub_title_it_pars'] += ', '
            elif i < nr_it_parameters - 1:
                ret['sub_title_it_pars'] += ', and '
    ret['sub_title_partitions'] = ''
    for i, val in enumerate(ret['partitions']):
        ret['sub_title_partitions'] += replaceCSVLabels(val, True, True)
        if (nr_partitions <= 2):
            if i < nr_partitions - 1:
                ret['sub_title_partitions'] += ' and '
        else:
            if i < nr_partitions - 2:
                ret['sub_title_partitions'] += ', '
            elif i < nr_partitions - 1:
                ret['sub_title_partitions'] += ', and '

    ret['dataf_name_main'] = str(ret['grp_names'][-1]) + '_for_options_' + \
                             '-'.join(ret['it_parameters']) + \
                             '_and_properties_'
    ret['dataf_name_partition'] = '-'.join([a[:min(3, len(a))] for a in map(str, ret['partitions'])])
    ret['b'] = combineRt(data)
    ret['b_all_partitions'] = ret['b'].reset_index().set_index(ret['partitions'])
    tex_infos = {'title': 'Combined R \\& t Errors vs ' + replaceCSVLabels(str(ret['grp_names'][-1]), True, True) +
                          ' for Parameter Variations of ' + ret['sub_title_it_pars'] + ' separately for ' +
                          ret['sub_title_partitions'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': True,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    ret['b_single_partitions'] = []
    idx_old = None
    for p in ret['b_all_partitions'].index:
        if idx_old is not None and idx_old == p:
            continue
        idx_old = p
        tmp2 = ret['b_all_partitions'].loc[p]
        part_name = '_'.join([str(ni) + '-' + str(vi) for ni, vi in zip(tmp2.index.names, tmp2.index[0])])
        part_name_l = [replaceCSVLabels(str(ni)) + ' = ' +
                       tex_string_coding_style(str(vi)) for ni, vi in zip(tmp2.index.names, tmp2.index[0])]
        part_name_title = ''
        for i, val in enumerate(part_name_l):
            part_name_title += val
            if (len(part_name_l) <= 2):
                if i < len(part_name_l) - 1:
                    part_name_title += ' and '
            else:
                if i < len(part_name_l) - 2:
                    part_name_title += ', '
                elif i < len(part_name_l) - 1:
                    part_name_title += ', and '
        tmp2 = tmp2.reset_index().drop(ret['partitions'], axis=1)
        tmp2 = tmp2.set_index(ret['it_parameters']).T
        tmp2.columns = ['-'.join(map(str, a)) for a in tmp2.columns]
        tmp2.columns.name = '-'.join(ret['it_parameters'])
        dataf_name_main_property = ret['dataf_name_main'] + part_name.replace('.', 'd')
        dataf_name = dataf_name_main_property + '.csv'
        b_name = 'data_RTerrors_vs_' + dataf_name
        fb_name = os.path.join(ret['tdata_folder'], b_name)
        ret['b_single_partitions'].append({'data': tmp2,
                                           'part_name': part_name,
                                           'part_name_title': part_name_title,
                                           'dataf_name_main_property': dataf_name_main_property,
                                           'dataf_name': dataf_name})
        with open(fb_name, 'a') as f:
            f.write('# Combined R & t errors vs ' + str(ret['grp_names'][-1]) + ' for properties ' + part_name + '\n')
            f.write('# Column parameters: ' + '-'.join(ret['it_parameters']) + '\n')
            tmp2.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        stats_all = tmp2.stack().reset_index()
        stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
        # figure types: sharp plot, smooth, const plot, ybar, xbar
        use_limits = {'miny': None, 'maxy': None}
        if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
            use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
        if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
            use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
        reltex_name = os.path.join(ret['rel_data_path'], b_name)
        tex_infos['sections'].append({'file': reltex_name,
                                      'name': 'Combined R \\& t errors vs ' +
                                              replaceCSVLabels(str(ret['grp_names'][-1]), True) +
                                              ' for parameter variations of \\\\' + ret['sub_title_it_pars'] +
                                              ' based on properties \\\\' + part_name.replace('_', '\\_'),
                                      # If caption is None, the field name is used
                                      'caption': 'Combined R \\& t errors vs ' +
                                                 replaceCSVLabels(str(ret['grp_names'][-1]), True) +
                                                 ' for parameter variations of ' + ret['sub_title_it_pars'] +
                                                 ' based on properties ' + part_name.replace('_', '\\_'),
                                      'fig_type': 'smooth',
                                      'plots': list(tmp2.columns.values),
                                      'label_y': 'Combined R \\& t error',
                                      'plot_x': str(ret['grp_names'][-1]),
                                      'label_x': replaceCSVLabels(str(ret['grp_names'][-1])),
                                      'limits': use_limits,
                                      'legend': [tex_string_coding_style(a) for a in list(tmp2.columns.values)],
                                      'legend_cols': None,
                                      'use_marks': ret['use_marks'],
                                      'use_log_y_axis': False,
                                      })
        tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_RTerrors_vs_' + ret['dataf_name_main'] + ret['dataf_name_partition']
    texf_name = base_out_name + '.tex'
    if ret['build_pdf'][0]:
        pdf_name = base_out_name + '.pdf'
        ret['res'] = abs(compile_tex(rendered_tex,
                                     ret['tex_folder'],
                                     texf_name,
                                     tex_infos['make_index'],
                                     os.path.join(ret['pdf_folder'], pdf_name),
                                     tex_infos['figs_externalize']))
    else:
        ret['res'] = abs(compile_tex(rendered_tex, ret['tex_folder'], texf_name))
    if ret['res'] != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)
    return ret


def pars_calc_single_fig(**keywords):
    if len(keywords) < 3 or len(keywords) > 7:
        raise ValueError('Wrong number of arguments for function pars_calc_single_fig')
    if 'data' not in keywords:
        raise ValueError('Missing data argument of function pars_calc_single_fig')
    data = keywords['data']
    ret = {}
    if 'res_folder' not in keywords:
        raise ValueError('Missing res_folder argument of function pars_calc_single_fig')
    ret['res_folder'] = keywords['res_folder']
    ret['use_marks'] = False
    if 'use_marks' not in keywords:
        print('No information provided if marks should be used: Disabling marks')
    else:
        ret['use_marks'] = keywords['use_marks']
    ret['build_pdf'] = (False, True,)
    if 'build_pdf' in keywords:
        ret['build_pdf'] = keywords['build_pdf']
    if len(ret['build_pdf']) != 2:
        raise ValueError('Wrong number of arguments for build_pdf')
    ret['pdf_folder'] = None
    if ret['build_pdf'][0] or ret['build_pdf'][1]:
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
    ret['dataf_name_main'] = str(ret['grp_names'][-1]) + '_for_options_' + '-'.join(ret['grp_names'][0:-1])
    ret['dataf_name'] = ret['dataf_name_main'] + '.csv'
    ret['b'] = combineRt(data)
    ret['b'] = ret['b'].T
    ret['b'].columns = ['-'.join(map(str, a)) for a in ret['b'].columns]
    ret['b'].columns.name = '-'.join(ret['grp_names'][0:-1])
    b_name = 'data_RTerrors_vs_' + ret['dataf_name']
    fb_name = os.path.join(ret['tdata_folder'], b_name)
    with open(fb_name, 'a') as f:
        f.write('# Combined R & t errors vs ' + str(ret['grp_names'][-1]) + '\n')
        f.write('# Parameters: ' + '-'.join(ret['grp_names'][0:-1]) + '\n')
        ret['b'].to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
    ret['sub_title'] = ''
    nr_it_parameters = len(ret['grp_names'][0:-1])
    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols, replaceCSVLabels
    for i, val in enumerate(ret['grp_names'][0:-1]):
        ret['sub_title'] += replaceCSVLabels(val, True, True)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                ret['sub_title'] += ' and '
        else:
            if i < nr_it_parameters - 2:
                ret['sub_title'] += ', '
            elif i < nr_it_parameters - 1:
                ret['sub_title'] += ', and '
    tex_infos = {'title': 'Combined R \\& t Errors vs ' + replaceCSVLabels(str(ret['grp_names'][-1]), True, True) +
                          ' for Parameter Variations of ' + ret['sub_title'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    stats_all = ret['b'].stack().reset_index()
    stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
    # figure types: sharp plot, smooth, const plot, ybar, xbar
    use_limits = {'miny': None, 'maxy': None}
    if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
        use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
    if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
        use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
    reltex_name = os.path.join(ret['rel_data_path'], b_name)
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': 'Combined R \\& t errors vs ' +
                                          replaceCSVLabels(str(ret['grp_names'][-1]), True) +
                                          ' for parameter variations of \\\\' + ret['sub_title'],
                                  # If caption is None, the field name is used
                                  'caption': 'Combined R \\& t errors vs ' +
                                             replaceCSVLabels(str(ret['grp_names'][-1]), True) +
                                             ' for parameter variations of ' + ret['sub_title'],
                                  'fig_type': 'smooth',
                                  'plots': list(ret['b'].columns.values),
                                  'label_y': 'Combined R \\& t error',
                                  'plot_x': str(ret['grp_names'][-1]),
                                  'label_x': replaceCSVLabels(str(ret['grp_names'][-1])),
                                  'limits': use_limits,
                                  'legend': [tex_string_coding_style(a) for a in list(ret['b'].columns.values)],
                                  'legend_cols': None,
                                  'use_marks': ret['use_marks'],
                                  'use_log_y_axis': False,
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_RTerrors_vs_' + ret['dataf_name_main']
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


def pars_calc_multiple_fig(**keywords):
    if len(keywords) < 4 or len(keywords) > 7:
        raise ValueError('Wrong number of arguments for function pars_calc_multiple_fig')
    if 'data' not in keywords:
        raise ValueError('Missing data argument of function pars_calc_multiple_fig')
    data = keywords['data']
    ret = {}
    if 'res_folder' not in keywords:
        raise ValueError('Missing res_folder argument of function pars_calc_multiple_fig')
    ret['res_folder'] = keywords['res_folder']
    ret['fig_type'] = 'surface'
    if 'fig_type' not in keywords:
        print('No information provided about the figure type: Using \'surface\'')
    else:
        ret['fig_type'] = keywords['fig_type']
    ret['use_marks'] = False
    if 'use_marks' not in keywords:
        print('No information provided if marks should be used: Disabling marks')
    else:
        ret['use_marks'] = keywords['use_marks']
    ret['build_pdf'] = (False, True,)
    if 'build_pdf' in keywords:
        ret['build_pdf'] = keywords['build_pdf']
    if len(ret['build_pdf']) != 2:
        raise ValueError('Wrong number of arguments for build_pdf')
    ret['pdf_folder'] = None
    if ret['build_pdf'][0] or ret['build_pdf'][1]:
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
    ret['dataf_name_main'] = ret['grp_names'][-2] + '_and_' + ret['grp_names'][-1] + \
                             '_for_options_' + '-'.join(ret['grp_names'][0:-2])
    ret['dataf_name'] = ret['dataf_name_main'] + '.csv'
    ret['b'] = combineRt(data)
    ret['b'] = ret['b'].unstack()
    ret['b'] = ret['b'].T
    ret['b'].columns = ['-'.join(map(str, a)) for a in ret['b'].columns]
    ret['b'].columns.name = '-'.join(ret['grp_names'][0:-2])
    ret['b'] = ret['b'].reset_index()
    nr_equal_ss = int(ret['b'].groupby(ret['b'].columns.values[0]).size().array[0])
    b_name = 'data_RTerrors_vs_' + ret['dataf_name']
    fb_name = os.path.join(ret['tdata_folder'], b_name)
    with open(fb_name, 'a') as f:
        f.write('# Combined R & t errors vs ' + ret['grp_names'][-2] + ' and ' + ret['grp_names'][-1] + '\n')
        f.write('# Parameters: ' + '-'.join(ret['grp_names'][0:-2]) + '\n')
        ret['b'].to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    ret['sub_title'] = ''
    nr_it_parameters = len(ret['grp_names'][0:-2])
    from statistics_and_plot import tex_string_coding_style, compile_tex, replaceCSVLabels
    for i, val in enumerate(ret['grp_names'][0:-2]):
        ret['sub_title'] += replaceCSVLabels(val, True, True)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                ret['sub_title'] += ' and '
        else:
            if i < nr_it_parameters - 2:
                ret['sub_title'] += ', '
            elif i < nr_it_parameters - 1:
                ret['sub_title'] += ', and '
    tex_infos = {'title': 'Combined R \\& t Errors vs ' + replaceCSVLabels(ret['grp_names'][-2], True, True) +
                          ' and ' + replaceCSVLabels(ret['grp_names'][-1], True, True) +
                          ' for Parameter Variations of ' + ret['sub_title'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': False,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': True}
    reltex_name = os.path.join(ret['rel_data_path'], b_name)
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': 'Combined R \\& t errors vs ' +
                                          replaceCSVLabels(str(ret['grp_names'][-2]), True, True) +
                                          ' and ' + replaceCSVLabels(str(ret['grp_names'][-1]), True, True) +
                                          ' for parameter variations of ' + ret['sub_title'],
                                  'fig_type': ret['fig_type'],
                                  'plots_z': list(ret['b'].columns.values)[2:],
                                  'diff_z_labels': False,
                                  'label_z': 'Combined R \\& t error',
                                  'plot_x': str(ret['b'].columns.values[1]),
                                  'label_x': str(ret['b'].columns.values[1]),
                                  'plot_y': str(ret['b'].columns.values[0]),
                                  'label_y': str(ret['b'].columns.values[0]),
                                  'legend': [tex_string_coding_style(a) for a in list(ret['b'].columns.values)[2:]],
                                  'use_marks': ret['use_marks'],
                                  'mesh_cols': nr_equal_ss,
                                  'use_log_z_axis': False
                                  })
    template = ji_env.get_template('usac-testing_3D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   use_fixed_caption=False,
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_RTerrors_vs_' + ret['dataf_name_main']
    texf_name = base_out_name + '.tex'
    if ret['build_pdf'][0]:
        pdf_name = base_out_name + '.pdf'
        ret['res'] = abs(compile_tex(rendered_tex,
                                     ret['tex_folder'],
                                     texf_name,
                                     True,
                                     os.path.join(ret['pdf_folder'], pdf_name),
                                     tex_infos['figs_externalize']))
    else:
        ret['res'] = abs(compile_tex(rendered_tex, ret['tex_folder'], texf_name, True))
    if ret['res'] != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)
    return ret


def combineRt(data):
    #Get R and t mean and standard deviation values
    stat_R = data['R_diffAll'].unstack()
    stat_t = data['t_angDiff_deg'].unstack()
    stat_R_mean = stat_R['mean']
    stat_t_mean = stat_t['mean']
    stat_R_std = stat_R['std']
    stat_t_std = stat_t['std']
    # comb_stat_r = stat_R_mean.abs() + 2 * stat_R_std
    # comb_stat_t = stat_t_mean.abs() + 2 * stat_t_std
    comb_stat_r = stat_R_mean.abs() + stat_R_std
    comb_stat_t = stat_t_mean.abs() + stat_t_std
    ma = comb_stat_r.select_dtypes(include=[np.number]).dropna().values.max()
    mi = comb_stat_r.select_dtypes(include=[np.number]).dropna().values.min()
    r_r = ma - mi
    ma = comb_stat_t.select_dtypes(include=[np.number]).dropna().values.max()
    mi = comb_stat_t.select_dtypes(include=[np.number]).dropna().values.min()
    r_t = ma - mi
    comb_stat_r = comb_stat_r / r_r
    comb_stat_t = comb_stat_t / r_t
    b = (comb_stat_r + comb_stat_t) / 2
    return b


#def get_best_comb_and_th_1(data, res_folder, build_pdf=(False, True, )):
def get_best_comb_and_th_1(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    ret = pars_calc_single_fig(**keywords)

    #Output best and worst b values for every combination
    if len(ret['b'].columns) > 10:
        fig_type = 'xbar'
    else:
        fig_type = 'ybar'
    b_best_idx = ret['b'].idxmin(axis=0)
    # Insert a tex line break for long options
    b_cols_tex = insert_opt_lbreak(ret['b'].columns)
    b_best_l = [[val, ret['b'].loc[val].iloc[i], ret['b'].columns[i], b_cols_tex[i]] for i, val in enumerate(b_best_idx)]
    b_best = pd.DataFrame.from_records(data=b_best_l, columns=[ret['grp_names'][-1], 'b_best', 'options', 'options_tex'])
    #b_best.set_index('options', inplace=True)
    b_best_name = 'data_best_RTerrors_and_' + ret['dataf_name']
    fb_best_name = os.path.join(ret['tdata_folder'], b_best_name)
    with open(fb_best_name, 'a') as f:
        f.write('# Best combined R & t errors and their ' + str(ret['grp_names'][-1]) + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(ret['grp_names'][0:-1]) + '\n')
        b_best.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    b_worst_idx = ret['b'].idxmax(axis=0)
    b_worst_l = [[val, ret['b'].loc[val].iloc[i], ret['b'].columns[i], b_cols_tex[i]] for i, val in enumerate(b_worst_idx)]
    b_worst = pd.DataFrame.from_records(data=b_worst_l, columns=[ret['grp_names'][-1], 'b_worst', 'options', 'options_tex'])
    b_worst_name = 'data_worst_RTerrors_and_' + ret['dataf_name']
    fb_worst_name = os.path.join(ret['tdata_folder'], b_worst_name)
    with open(fb_worst_name, 'a') as f:
        f.write('# Best combined R & t errors and their ' + str(ret['grp_names'][-1]) + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(ret['grp_names'][0:-1]) + '\n')
        b_worst.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    #Get data for tex file generation
    from statistics_and_plot import replaceCSVLabels
    tex_infos = {'title': 'Best and Worst Combined R \\& t Errors and Their ' +
                          replaceCSVLabels(str(ret['grp_names'][-1]), False, True) +
                          ' for Parameter Variations of ' + ret['sub_title'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    section_name = 'Smallest combined R \\& t errors and their ' + replaceCSVLabels(str(ret['grp_names'][-1]))
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_best_name),
                                  'name': section_name,
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': fig_type,
                                  'plots': ['b_best'],
                                  'label_y': 'error',#Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'Options',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'options_tex',
                                  #Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': [str(ret['grp_names'][-1])],
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 0,
                                  'limits': None,
                                  #If None, no legend is used, otherwise use a list
                                  'legend': None,
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': True,
                                  'use_log_y_axis': False,
                                  'large_meta_space_needed': False,
                                  'caption': 'Smallest combined R \\& t errors (error bars) and their ' +
                                             replaceCSVLabels(str(ret['grp_names'][-1])) +
                                             ' which appears on top of each bar.'
                                  })
    section_name = 'Worst combined R \\& t errors and their ' + replaceCSVLabels(str(ret['grp_names'][-1]))
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_worst_name),
                                  'name': section_name,
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': fig_type,
                                  'plots': ['b_worst'],
                                  'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'Options',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'options_tex',
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': [str(ret['grp_names'][-1])],
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
                                  'large_meta_space_needed': False,
                                  'caption': 'Biggest combined R \\& t errors (error bars) and their ' +
                                             replaceCSVLabels(str(ret['grp_names'][-1])) +
                                             ' which appears on top of each bar.'
                                  })
    ret['res'] = compile_2D_bar_chart('tex_best-worst_RT-errors_and_' + ret['grp_names'][-1], tex_infos, ret)

    b_best2_idx = b_best['b_best'].idxmin()
    alg_comb_best = str(b_best['options'].loc[b_best2_idx])
    th_best = float(b_best[ret['grp_names'][-1]].loc[b_best2_idx])
    th_best_mean = round(float(b_best[ret['grp_names'][-1]].mean()), 3)
    b_best_val = float(b_best['b_best'].loc[b_best2_idx])

    main_parameter_name = keywords['res_par_name']#'USAC_opt_refine_ops_th'
    # Check if file and parameters exist
    ppar_file, ret['res'] = check_par_file_exists(main_parameter_name, ret['res_folder'], ret['res'])

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = alg_comb_best.split('-')
        if len(ret['grp_names'][0:-1]) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(ret['grp_names'][0:-1]):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithms': alg_w,
                                         'th_best': th_best,
                                         'th_best_mean': th_best_mean,
                                         'b_best_val': b_best_val}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    return ret['res']


def get_best_comb_inlrat_1(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    ret = pars_calc_single_fig(**keywords)
    b_mean = ret['b'].mean(axis=0)
    b_mean_best = b_mean.idxmin()
    alg_best = str(b_mean_best)
    b_best = float(b_mean.loc[b_mean_best])

    b_mean = b_mean.reset_index()
    b_mean.columns = ['options', 'b_mean']
    # Insert a tex line break for long options
    b_mean['options_tex'] = insert_opt_lbreak(ret['b'].columns)
    b_mean_name = 'data_mean_RTerrors_over_all_' + ret['dataf_name']
    fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
    with open(fb_mean_name, 'a') as f:
        f.write('# Mean combined R & t errors over all ' + str(ret['grp_names'][-1]) + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(ret['grp_names'][0:-1]) + '\n')
        b_mean.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    # Get data for tex file generation
    if len(ret['b'].columns) > 10:
        fig_type = 'xbar'
    else:
        fig_type = 'ybar'
    from statistics_and_plot import replaceCSVLabels
    tex_infos = {'title': 'Mean Combined R \\& t Errors over all ' +
                          replaceCSVLabels(str(ret['grp_names'][-1]), True, True) +
                          ' for Parameter Variations of ' + ret['sub_title'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    section_name = 'Mean combined R \\& t errors over all ' + replaceCSVLabels(str(ret['grp_names'][-1]), True)
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], b_mean_name),
                                  'name': section_name,
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': fig_type,
                                  'plots': ['b_mean'],
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
                                  'large_meta_space_needed': False,
                                  'caption': 'Mean combined R \\& t errors (error bars) over all ' +
                                             replaceCSVLabels(str(ret['grp_names'][-1]), True) + '.'
                                  })
    ret['res'] = compile_2D_bar_chart('tex_mean_RT-errors_' + ret['grp_names'][-1], tex_infos, ret)

    main_parameter_name = keywords['res_par_name']#'USAC_opt_refine_ops_inlrat'
    # Check if file and parameters exist
    ppar_file, ret['res'] = check_par_file_exists(main_parameter_name, ret['res_folder'], ret['res'])

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = alg_best.split('-')
        if len(ret['grp_names'][0:-1]) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(ret['grp_names'][0:-1]):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithms': alg_w,
                                         'b_best_val': b_best}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    return ret['res']


def check_par_file_exists(main_parameter_name, res_folder, res):
    par_file_main = 'resulting_best_parameters'
    par_file = par_file_main + '.yaml'
    res_folder_parent = os.path.abspath(os.path.join(res_folder, os.pardir))  # Get parent directory
    ppar_file = os.path.join(res_folder_parent, par_file)
    # Check if file and parameters exist
    cnt = 1
    while True:
        if os.path.exists(ppar_file):
            try:
                ydata = readYaml(ppar_file)
                try:
                    stored_pars = ydata[main_parameter_name]
                    warnings.warn('Best USAC refine options already found in ' + ppar_file, UserWarning)
                    par_file = par_file_main + '_' + str(int(cnt)) + '.yaml'
                    ppar_file = os.path.join(res_folder_parent, par_file)
                    cnt += 1
                    if cnt == 2:
                        res += 1
                except:
                    break
            except BaseException:
                warnings.warn('Unable to read parameter file', UserWarning)
                return -1
        else:
            break
    return ppar_file, res


def insert_opt_lbreak(columns):
    # Insert a tex line break for long options
    from statistics_and_plot import tex_string_coding_style
    b_cols_tex = []
    for el in columns:
        if len(el) > 20:
            if '-' in el:
                cl = int(len(el) / 2.5)
                idx = [m.start() for m in re.finditer('-', el)]
                found = False
                for i in idx:
                    if i > cl:
                        found = True
                        b_cols_tex.append(tex_string_coding_style(el[0:(i + 1)]) +
                                          '\\\\' + tex_string_coding_style(el[(i + 1):]))
                        break
                if not found:
                    b_cols_tex.append(tex_string_coding_style(el[0:(idx[-1] + 1)]) +
                                      '\\\\' + tex_string_coding_style(el[(idx[-1] + 1):]))
            else:
                b_cols_tex.append(tex_string_coding_style(el))
        else:
            b_cols_tex.append(tex_string_coding_style(el))
    return b_cols_tex


def compile_2D_bar_chart(filen_pre, tex_infos, ret):
    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = filen_pre + '_options_' + '-'.join(map(str, ret['grp_names'][0:-1]))
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
    return ret['res']

def compile_2D_2y_axis(filen_pre, tex_infos, ret):
    template = ji_env.get_template('usac-testing_2D_plots_2y_axis.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   nonnumeric_x=tex_infos['nonnumeric_x'],
                                   sections=tex_infos['sections'],
                                   fill_bar=True)
    base_out_name = filen_pre + '_options_' + '-'.join(map(str, ret['grp_names'][0:-1]))
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
    return ret['res']


def get_best_comb_and_th_for_inlrat_1(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    ret = pars_calc_multiple_fig(**keywords)
    tmp = ret['b'].groupby(ret['grp_names'][-1])
    grp_keys = tmp.groups.keys()
    data_l = []
    for grp in grp_keys:
        b_min_i = tmp.get_group(grp).iloc[:, 2:].idxmin(axis=0)
        rows = []
        for idx, val in enumerate(b_min_i):
            rows.append([grp,
                         tmp.get_group(grp).iloc[:, 1].loc[val],
                         tmp.get_group(grp).loc[val, b_min_i.index.values[idx]],
                         b_min_i.index.values[idx]])
        data_l += rows
    data = pd.DataFrame.from_records(data=data_l, columns=[ret['grp_names'][-1],
                                                           ret['grp_names'][-2],
                                                           'b_min',
                                                           ret['b'].columns.name])
    from statistics_and_plot import replaceCSVLabels, tex_string_coding_style
    tex_infos = {'title': 'Smallest Combined R \\& t Errors and Their Corresponding ' +
                          replaceCSVLabels(str(ret['grp_names'][-2]), False, True) + ' for every ' +
                          replaceCSVLabels(str(ret['grp_names'][-1]), False, True) +
                          ' and Parameter Variations of ' + ret['sub_title'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': False,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true, non-numeric entries can be provided for the x-axis
                 'nonnumeric_x': False
                 }
    data1 = data.set_index(ret['grp_names'][-1]).groupby(ret['b'].columns.name)
    grp_keys = data1.groups.keys()
    dataf_name_main = 'data_' + ret['grp_names'][-1] + '_vs_b_min_and_corresponding_' + \
                      ret['grp_names'][-2] + '_for_option_'
    for grp in grp_keys:
        data_a = data1.get_group(grp).drop(ret['b'].columns.name, axis=1)
        dataf_name = dataf_name_main + str(grp) + '.csv'
        datapf_name = os.path.join(ret['tdata_folder'], dataf_name)
        with open(datapf_name, 'a') as f:
            f.write('# Smallest combined R & t errors and their ' + str(ret['grp_names'][-2])
                    + ' for every ' + str(ret['grp_names'][-1]) + '\n')
            f.write('# Used parameters: ' + str(grp) + '\n')
            data_a.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        section_name = 'Smallest combined R \\& t errors and their ' +\
                       replaceCSVLabels(str(ret['grp_names'][-2])) +\
                       '\\\\vs ' + replaceCSVLabels(str(ret['grp_names'][-1])) +\
                       ' for parameters ' + tex_string_coding_style(str(grp))
        tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], dataf_name),
                                      # Name of the whole section
                                      'name': section_name.replace('\\\\', ' '),
                                      # Title of the figure
                                      'title': section_name,
                                      'title_rows': section_name.count('\\\\'),
                                      'fig_type': 'smooth',
                                      # Column name for charts based on the left y-axis
                                      'plots_l': ['b_min'],
                                      # Label of the left y-axis.
                                      'label_y_l': 'error',
                                      # Column name for charts based on the right y-axis
                                      'plots_r': [ret['grp_names'][-2]],
                                      # Label of the right y-axis.
                                      'label_y_r': replaceCSVLabels(str(ret['grp_names'][-2])),
                                      # Label of the x-axis.
                                      'label_x': replaceCSVLabels(str(ret['grp_names'][-1])),
                                      # Column name of the x-axis.
                                      'plot_x': ret['grp_names'][-1],
                                      # Maximum and/or minimum y value/s on the left y-axis
                                      'limits_l': None,
                                      # Legend entries for the charts belonging to the left y-axis
                                      'legend_l': ['min. error (left axis)'],
                                      # Maximum and/or minimum y value/s on the right y-axis
                                      'limits_r': None,
                                      # Legend entries for the charts belonging to the right y-axis
                                      'legend_r': [replaceCSVLabels(str(ret['grp_names'][-2])) + ' (right axis)'],
                                      'legend_cols': 1,
                                      'use_marks': True,
                                      'caption': 'Smallest combined R \\& t errors (left axis) and their ' +
                                                 replaceCSVLabels(str(ret['grp_names'][-2])) +
                                                 ' (right axis) vs ' + replaceCSVLabels(str(ret['grp_names'][-1])) +
                                                 ' for parameters ' + tex_string_coding_style(str(grp)) + '.'
                                      })
    ret['res'] = compile_2D_2y_axis('tex_min_RT-errors_and_corresponding_' + ret['grp_names'][-2] +
                                    '_vs_' + ret['grp_names'][-1] + '_for_', tex_infos, ret)

    data_min = data.loc[data.groupby(ret['grp_names'][-1])['b_min'].idxmin()]
    col_n = str(ret['b'].columns.name) + '--' + str(ret['grp_names'][-2])
    data_min[col_n] = data_min[[ret['b'].columns.name,
                                ret['grp_names'][-2]]].apply(lambda x: ', '.join(map(str, x)).replace('_', '\\_'),
                                                             axis=1)
    data_min1 = data_min.drop([ret['b'].columns.name, ret['grp_names'][-2]], axis=1)
    dataf_name_main = 'data_' + ret['grp_names'][-1] + '_vs_b_min_and_corresponding_' + \
                      ret['grp_names'][-2] + '_and_used_option'
    dataf_name = dataf_name_main + '.csv'
    datapf_name = os.path.join(ret['tdata_folder'], dataf_name)
    with open(datapf_name, 'a') as f:
        f.write('# Smallest combined R & t errors and their corresponding ' + str(ret['grp_names'][-2])
                + ' and parameter set for every ' + str(ret['grp_names'][-1]) + '\n')
        f.write('# Used parameters: ' + str(ret['b'].columns.name) + '\n')
        data_min1.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    tex_infos = {'title': 'Smallest Combined R \\& t Errors and Their Corresponding ' +
                          replaceCSVLabels(str(ret['grp_names'][-2]), False, True) +
                          ' and Parameter Set of ' + ret['sub_title'] +
                          ' for every ' +
                          replaceCSVLabels(str(ret['grp_names'][-1]), False, True),
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], dataf_name),
                                  'name': 'Smallest Combined R \\& t Errors',
                                  'title': 'Smallest Combined R \\& t Errors',
                                  'title_rows': 0,
                                  'fig_type': 'ybar',
                                  'plots': ['b_min'],
                                  'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': replaceCSVLabels(str(ret['grp_names'][-1])),
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': ret['grp_names'][-1],
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': [col_n],
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 45,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': None,
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': False,
                                  'use_log_y_axis': False,
                                  'large_meta_space_needed': False,
                                  'caption': 'Smallest combined R \\& t errors and their ' +
                                             'corresponding parameter set and ' +
                                             replaceCSVLabels(str(ret['grp_names'][-2]), False, True) +
                                             ' (on top of bar separated by a comma)' +
                                             ' for every ' +
                                             replaceCSVLabels(str(ret['grp_names'][-1]), False, True) + '.'
                                  })
    ret['res'] = compile_2D_bar_chart('tex_' + ret['grp_names'][-1] + '_vs_b_min_and_corresponding_' + \
                      ret['grp_names'][-2] + '_and_used_option_of', tex_infos, ret)

    data_min_c = data_min[ret['b'].columns.name].value_counts()
    if data_min_c.count() > 1:
        if data_min_c.iloc[0] == data_min_c.iloc[1]:
            data_min_c = data_min_c.loc[data_min_c == data_min_c.iloc[0]]
            data_min2 = data_min.loc[data_min[ret['b'].columns.name].isin(data_min_c.index.values)]
            data_min2 = data_min2.loc[data_min2['b_min'].idxmin()]
            th_mean = float(data_min2[ret['grp_names'][-2]])
            alg = str(data_min2[ret['b'].columns.name])
            b_min = float(data_min2['b_min'])
        else:
            data_min2 = data_min.loc[data_min[ret['b'].columns.name] == data_min_c.index.values[0]]
            if len(data_min2.shape) == 1:
                th_mean = float(data_min2[ret['grp_names'][-2]])
                alg = str(data_min2[ret['b'].columns.name])
                b_min = float(data_min2['b_min'])
            else:
                th_mean = float(data_min2[ret['grp_names'][-2]].mean())
                alg = str(data_min2[ret['b'].columns.name].iloc[0])
                b_min = float(data_min2['b_min'].mean())
    else:
        th_mean = float(data_min[ret['grp_names'][-2]].mean())
        alg = str(data_min[ret['b'].columns.name].iloc[0])
        b_min = float(data_min['b_min'].mean())

    main_parameter_name = keywords['res_par_name']#'USAC_opt_refine_ops_inlrat_th'
    # Check if file and parameters exist
    ppar_file, ret['res'] = check_par_file_exists(main_parameter_name, ret['res_folder'], ret['res'])

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = alg.split('-')
        if len(ret['grp_names'][0:-2]) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(ret['grp_names'][0:-2]):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithms': alg_w,
                                         'th': th_mean,
                                         'b_min': b_min}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    return ret['res']


def get_best_comb_th_scenes_1(**keywords):
    ret = pars_calc_single_fig_partitions(**keywords)
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
                 'nonnumeric_x': True
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
                                   fill_bar=True)
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
                 'fill_bar': True
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
                                   sections=tex_infos['sections'])
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


def filter_nr_kps(**vars):
    return vars['data'].loc[vars['data']['nrTP'] == '100to1000']


def calc_Time_Model(**vars):
    # it_parameters: algorithms
    # xy_axis_columns: nrCorrs_GT, (inlRat_GT)
    # eval_columns: robEstimationAndRef_us
    # data_separators: inlRatMin, th
    if 'partitions' in vars:
        for key in vars['partitions']:
            if key not in vars['data_separators']:
                raise ValueError('All partition names must be included in the data separators.')
        if ('x_axis_column' in vars and len(vars['data_separators']) != (len(vars['partitions']) + 1)) or \
           ('xy_axis_columns' in vars and len(vars['data_separators']) != (len(vars['partitions']) + 2)):
            raise ValueError('Wrong number of data separators.')
    elif 'x_axis_column' in vars and len(vars['data_separators']) != 1:
        raise ValueError('Only one data separator is allowed.')
    elif 'xy_axis_columns' in vars and len(vars['data_separators']) != 2:
        raise ValueError('Only two data separators are allowed.')
    if 'x_axis_column' in vars:
        x_axis_column = vars['x_axis_column']
    elif 'xy_axis_columns' in vars:
        x_axis_column = vars['xy_axis_columns']
    else:
        raise ValueError('Missing x-axis column names')
    needed_cols = vars['eval_columns'] + vars['it_parameters'] + x_axis_column + vars['data_separators']
    df = vars['data'][needed_cols]
    # Calculate TP
    # df['actNrTP'] = (df[x_axis_column[0]] * df[x_axis_column[1]]).round()
    grpd_cols = vars['data_separators'] + vars['it_parameters']
    df_grp = df.groupby(grpd_cols)
    # Check if we can use a linear model or a second degree model (t = t_fixed + t_lin * nrCorrs_GT + t_2nd * nrCorrs_GT
    grp_keys = df_grp.groups.keys()
    std_dev = 3 # Number of standard deviations for filtering
    model_type = []
    # model = HuberRegressor()
    model = LinearRegression(n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for grp in grp_keys:
            tmp = df_grp.get_group(grp)
            # Filter out spikes (split data in chunks first)
            if tmp.shape[0] > 60:
                ma = tmp[vars['eval_columns'][0]].dropna().values.max()
                mi = tmp[vars['eval_columns'][0]].dropna().values.min()
                r_r = ma - mi
                r5 = r_r / 5
                a = mi
                parts = []
                for b in np.arange(mi + r5, ma + r5 / 2, r5):
                    if round(b - ma) == 0:
                        b += 1
                    part = tmp.loc[(tmp[vars['eval_columns'][0]] >= a) &
                                   (tmp[vars['eval_columns'][0]] < b)]
                    part = part.loc[np.abs(stats.zscore(part[vars['eval_columns'][0]])) < float(std_dev)]
                    parts.append(part)
                    a = b
                tmp1 = pd.concat(parts, ignore_index=False, sort=False, copy=False)
            if tmp.shape[0] < 4:
                warnings.warn('Too less measurements for calculating temporal model', UserWarning)
                model_type.append({'score': 0,
                                   'par_neg': 0,
                                   'parameters': [0, 0],
                                   'valid': False,
                                   'type': 0,
                                   'data': tmp,
                                   'grp': grp})
                continue
            else:
                tmp1 = tmp

            # values converts it into a numpy array, -1 means that calculate the dimension of rows
            X = pd.DataFrame(tmp1[x_axis_column[0]]).values.reshape(-1, 1)
            Y = pd.DataFrame(tmp1[vars['eval_columns'][0]]).values.reshape(-1, 1)
            scores = []
            par_negs = []
            if X.shape[0] > 150:
                kfold = KFold(n_splits=3, shuffle=True)
                for train, test in kfold.split(X, Y):
                    model.fit(X[train], Y[train])
                    score = model.score(X[test], Y[test])
                    scores.append(score)
                    # For LinearRegression:
                    par_neg = (0 if model.intercept_[0] >= 0 else 50 * abs(model.intercept_[0])) + \
                              (0 if model.coef_[0, 0] >= 0 else 10 * abs(model.coef_[0, 0]))
                    # For HuberRegressor:
                    # par_neg = (0 if model.intercept_ >= 0 else 50 * abs(model.intercept_)) + \
                    #           (0 if model.coef_[0] >= 0 else 10 * abs(model.coef_[0]))
                    par_negs.append(par_neg)
                mean_score = sum(scores) / len(scores)
                mean_par_neg = round(sum(par_negs) / len(par_negs))
            else:
                model.fit(X, Y)
                score = model.score(X, Y)
                par_neg = (0 if model.intercept_[0] >= 0 else 50 * abs(model.intercept_[0])) + \
                          (0 if model.coef_[0, 0] >= 0 else 10 * abs(model.coef_[0, 0]))
                mean_score = score
                mean_par_neg = round(par_neg)
            if mean_score > 0.67 and mean_par_neg == 0:
                # The linear model will fit the data
                if X.shape[0] > 150:
                    model.fit(X, Y)
                    score = model.score(X, Y)
                    # For LinearRegression:
                    par_neg = (0 if model.intercept_[0] >= 0 else 50 * abs(model.intercept_[0])) + \
                              (0 if model.coef_[0, 0] >= 0 else 10 * abs(model.coef_[0, 0]))
                parameters = model.intercept_.tolist() + model.coef_[0, :].tolist()
                # For HuberRegressor:
                # par_neg = (0 if model.intercept_ >= 0 else 50 * abs(model.intercept_)) + \
                #           (0 if model.coef_[0] >= 0 else 10 * abs(model.coef_[0]))
                # parameters = [model.intercept_] + model.coef_.tolist()
                model_type.append({'score': score,
                                   'par_neg': par_neg,
                                   'parameters': parameters,
                                   'valid': True,
                                   'type': 0,
                                   'data': tmp1,
                                   'grp': grp})
            else:
                polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
                if X.shape[0] > 150:
                    scores = []
                    par_negs = []
                    for train, test in kfold.split(X, Y):
                        x_poly = polynomial_features.fit_transform(X[train])
                        model.fit(x_poly, Y[train])
                        x_poly_test = polynomial_features.fit_transform(X[test], Y[test])
                        score = model.score(x_poly_test, Y[test])
                        scores.append(score)
                        # For LinearRegression:
                        par_neg = (0 if model.intercept_[0] >= 0 else 50 * abs(model.intercept_[0])) + \
                                  (0 if model.coef_[0, 0] >= 0 else 10 * abs(model.coef_[0, 0])) + \
                                  (0 if model.coef_[0, 1] >= 0 else 2 * abs(model.coef_[0, 1]))
                        # For HuberRegressor:
                        # par_neg = (0 if model.intercept_ >= 0 else 50 * abs(model.intercept_)) + \
                        #           (0 if model.coef_[0] >= 0 else 10 * abs(model.coef_[0])) + \
                        #           (0 if model.coef_[1] >= 0 else 2 * abs(model.coef_[1]))
                        par_negs.append(par_neg)
                    mean_score1 = sum(scores) / len(scores)
                    mean_par_neg1 = round(sum(par_negs) / len(par_negs))
                else:
                    x_poly = polynomial_features.fit_transform(X)
                    model.fit(x_poly, Y)
                    score = model.score(x_poly, Y)
                    par_neg = (0 if model.intercept_[0] >= 0 else 50 * abs(model.intercept_[0])) + \
                              (0 if model.coef_[0, 0] >= 0 else 10 * abs(model.coef_[0, 0])) + \
                              (0 if model.coef_[0, 1] >= 0 else 2 * abs(model.coef_[0, 1]))
                    mean_score1 = score
                    mean_par_neg1 = round(par_neg)
                if mean_score1 > 1.2 * mean_score or (mean_par_neg1 < mean_par_neg and mean_score1 > 0.95 * mean_score):
                    if X.shape[0] > 150:
                        x_poly = polynomial_features.fit_transform(X)
                        model.fit(x_poly, Y)
                        score = model.score(x_poly, Y)
                        # For LinearRegression:
                        par_neg = (0 if model.intercept_[0] >= 0 else 50 * abs(model.intercept_[0])) + \
                                  (0 if model.coef_[0, 0] >= 0 else 10 * abs(model.coef_[0, 0])) + \
                                  (0 if model.coef_[0, 1] >= 0 else 2 * abs(model.coef_[0, 1]))
                    parameters = model.intercept_.tolist() + model.coef_[0, :].tolist()
                    # For HuberRegressor:
                    # par_neg = (0 if model.intercept_ >= 0 else 50 * abs(model.intercept_)) + \
                    #           (0 if model.coef_[0] >= 0 else 10 * abs(model.coef_[0])) + \
                    #           (0 if model.coef_[1] >= 0 else 2 * abs(model.coef_[1]))
                    # parameters = [model.intercept_] + model.coef_.tolist()
                    model_type.append({'score': score,
                                       'par_neg': par_neg,
                                       'parameters': parameters,
                                       'valid': True,
                                       'type': 1,
                                       'data': tmp1,
                                       'grp': grp})
                else:
                    model.fit(X, Y)
                    score = model.score(X, Y)
                    # For LinearRegression:
                    par_neg = (0 if model.intercept_[0] >= 0 else 50 * abs(model.intercept_[0])) + \
                              (0 if model.coef_[0, 0] >= 0 else 10 * abs(model.coef_[0, 0]))
                    parameters = model.intercept_.tolist() + model.coef_[0, :].tolist()
                    # For HuberRegressor:
                    # par_neg = (0 if model.intercept_ >= 0 else 50 * abs(model.intercept_)) + \
                    #           (0 if model.coef_[0] >= 0 else 10 * abs(model.coef_[0]))
                    # parameters = [model.intercept_] + model.coef_.tolist()
                    model_type.append({'score': score,
                                       'par_neg': par_neg,
                                       'parameters': parameters,
                                       'valid': True,
                                       'type': 0,
                                       'data': tmp1,
                                       'grp': grp})

        mod_sel_valid = [a['type'] for a in model_type if a['valid']]
        mod_sel = round(sum(mod_sel_valid)/len(mod_sel_valid), 4)
        if mod_sel > 0 and mod_sel < 1:
            scores1 = [a['score'] for a in model_type if a['type'] == 0 and a['valid']]
            par_negs1 = [a['par_neg'] for a in model_type if a['type'] == 0 and a['valid']]
            if len(scores1) > 0:
                scores11 = sum(scores1) / len(scores1)
                par_negs11 = sum(par_negs1) / len(par_negs1)
            else:
                scores11 = 0
                par_negs11 = 0
            scores2 = [a['score'] for a in model_type if a['type'] == 1 and a['valid']]
            par_negs2 = [a['par_neg'] for a in model_type if a['type'] == 1 and a['valid']]
            if len(scores2) > 0:
                scores21 = sum(scores2) / len(scores2)
                par_negs21 = sum(par_negs2) / len(par_negs2)
            else:
                scores21 = 0
                par_negs21 = 0
            l_rat2 = len(scores2) / len(mod_sel_valid)
            if l_rat2 > 0.5 and (scores21 > scores11 or par_negs11 > par_negs21):
                polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
                for i, val in enumerate(model_type):
                    if val['type'] == 0:
                        if val['valid']:
                            X = pd.DataFrame(val['data'][x_axis_column[0]]).values.reshape(-1, 1)
                            Y = pd.DataFrame(val['data'][vars['eval_columns'][0]]).values.reshape(-1, 1)
                            x_poly = polynomial_features.fit_transform(X)
                            model.fit(x_poly, Y)
                            model_type[i]['score'] = model.score(x_poly, Y)
                            # For LinearRegression:
                            model_type[i]['parameters'] = model.intercept_.tolist() + model.coef_[0, :].tolist()
                            # For HuberRegressor:
                            # model_type[i]['parameters'] = [model.intercept_] + model.coef_.tolist()
                            model_type[i]['type'] = 1
                        else:
                            model_type[i]['type'] = 1
                            model_type[i]['parameters'] = [0, 0, 0]
            else:
                for i, val in enumerate(model_type):
                    if val['type'] == 1:
                        X = pd.DataFrame(val['data'][x_axis_column[0]]).values.reshape(-1, 1)
                        Y = pd.DataFrame(val['data'][vars['eval_columns'][0]]).values.reshape(-1, 1)
                        model.fit(X, Y)
                        model_type[i]['score'] = model.score(X, Y)
                        # For LinearRegression:
                        model_type[i]['parameters'] = model.intercept_.tolist() + model.coef_[0, :].tolist()
                        # For HuberRegressor:
                        # model_type[i]['parameters'] = [model.intercept_] + model.coef_.tolist()
                        model_type[i]['type'] = 0
    if len(mod_sel_valid) != len(model_type):
        #Get valid parameters
        pars_valid = [a['parameters'] for a in model_type if a['valid']]
        mean_vals = [sum([a[i] for a in pars_valid]) / len(pars_valid) for i in range(len(pars_valid[0]))]
        for i, val in enumerate(model_type):
            if not val['valid']:
                model_type[i]['parameters'] = mean_vals

    data_new = {'score': [], 'fixed_time': [], 'linear_time': []}
    if model_type[0]['type'] == 1:
        data_new['squared_time'] = []
    for i in grpd_cols:
        data_new[i] = []
    for val in model_type:
        data_new['score'].append(val['score'])
        data_new['fixed_time'].append(val['parameters'][0])
        data_new['linear_time'].append(val['parameters'][1])
        if val['type'] == 1:
            data_new['squared_time'].append(val['parameters'][2])
        for i, name in enumerate(grpd_cols):
            data_new[name].append(val['grp'][i])
    data_new = pd.DataFrame(data_new)
    eval_columns = ['score', 'fixed_time', 'linear_time']
    eval_cols_lname = ['score $R^{2}$', 'fixed time $t_{f}$', 'time per keypoint $t_{n}$']
    eval_cols_log_scaling = [False, True, True]
    units = [('score', ''), ('fixed_time', '/$\\mu s$'), ('linear_time', '/$\\mu s$')]
    if model_type[0]['type'] == 1:
        eval_columns += ['squared_time']
        eval_cols_lname += ['quadratic time coefficient $t_{n^{2}}$']
        eval_cols_log_scaling += [True]
        units += [('squared_time', '')]
    ret = {'data': data_new,
           'it_parameters': vars['it_parameters'],
           'eval_columns': eval_columns,
           'eval_cols_lname': eval_cols_lname,
           'eval_cols_log_scaling': eval_cols_log_scaling,
           'units': units + vars['units'],
           'eval_init_input': vars['eval_columns']}
    if 'x_axis_column' in vars:
        if 'partitions' in vars:
            if len(vars['data_separators']) != (len(vars['partitions']) + 1):
                raise ValueError('Wrong number of data separators.')
            for key in vars['data_separators']:
                if key not in vars['partitions']:
                    ret['x_axis_column'] = [key]
                    break
            if 'x_axis_column' not in ret:
                raise ValueError('One element in the data separators should not be included in partitions as '
                                 'it is used for the x axis.')
        elif len(vars['data_separators']) != 1:
            raise ValueError('If no partitions for a 2D environment are used, only 1 data separator is allowed.')
        else:
            ret['x_axis_column'] = vars['data_separators']
    else:
        if 'partitions' in vars:
            if len(vars['data_separators']) != (len(vars['partitions']) + 2):
                raise ValueError('Wrong number of data separators.')
            ret['xy_axis_columns'] = []
            for key in vars['data_separators']:
                if key not in vars['partitions']:
                    ret['xy_axis_columns'].append(key)
            if len(ret['xy_axis_columns']) != 2:
                raise ValueError('Two elements in the data separators should not be included in partitions as '
                                 'they are used for the x- and y-axis.')
        elif len(vars['data_separators']) != 2:
            raise ValueError('If no partitions for a 3D environment are used, only 2 data separators are allowed.')
        else:
            ret['xy_axis_columns'] = vars['data_separators']
    if 'partitions' in vars:
        ret['partitions'] = vars['partitions']
    return ret


def estimate_alg_time_fixed_kp(**vars):
    if 'res_par_name' not in vars:
        raise ValueError('Missing parameter res_par_name')
    tmp, col_name = get_time_fixed_kp(**vars)
    tmp1 = tmp.loc[tmp.groupby(vars['t_data_separators'])[col_name].idxmin(axis=0)]
    tmp1.set_index(vars['it_parameters'], inplace=True)
    index_new = ['-'.join(a) for a in tmp1.index]
    tmp1.index = index_new
    index_name = '-'.join(vars['it_parameters'])
    tmp1.index.name = index_name
    tmp1['pars_tex'] = insert_opt_lbreak(index_new)

    vars = prepare_io(**vars)
    # if 'res_folder' not in vars:
    #     raise ValueError('Missing res_folder argument of function estimate_alg_time_fixed_kp')
    # res_folder = vars['res_folder']
    # use_marks = False
    # if 'use_marks' not in vars:
    #     print('No information provided if marks should be used: Disabling marks')
    # else:
    #     use_marks = vars['use_marks']
    # build_pdf = (False, True,)
    # if 'build_pdf' in vars:
    #     build_pdf = vars['build_pdf']
    # if len(build_pdf) != 2:
    #     raise ValueError('Wrong number of arguments for build_pdf')
    # pdf_folder = None
    # if build_pdf[0] or build_pdf[1]:
    #     pdf_folder = os.path.join(res_folder, 'pdf')
    #     try:
    #         os.mkdir(pdf_folder)
    #     except FileExistsError:
    #         # print('Folder', ret['pdf_folder'], 'for storing pdf files already exists')
    #         pass
    # tex_folder = os.path.join(res_folder, 'tex')
    # try:
    #     os.mkdir(tex_folder)
    # except FileExistsError:
    #     # print('Folder', ret['tex_folder'], 'for storing tex files already exists')
    #     pass
    # tdata_folder = os.path.join(tex_folder, 'data')
    # try:
    #     os.mkdir(tdata_folder)
    # except FileExistsError:
    #     # print('Folder', ret['tdata_folder'], 'for storing data files already exists')
    #     pass
    # rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    # nr_it_parameters = len(vars['it_parameters'])
    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols, replaceCSVLabels, strToLower
    # sub_title_it_pars = ''
    # for i, val in enumerate(vars['it_parameters']):
    #     sub_title_it_pars += replaceCSVLabels(val, True, True)
    #     if nr_it_parameters <= 2:
    #         if i < nr_it_parameters - 1:
    #             sub_title_it_pars += ' and '
    #     else:
    #         if i < nr_it_parameters - 2:
    #             sub_title_it_pars += ', '
    #         elif i < nr_it_parameters - 1:
    #             sub_title_it_pars += ', and '

    tmp.set_index(vars['it_parameters'], inplace=True)
    tmp = tmp.T
    par_cols = ['-'.join(map(str, a)) for a in tmp.columns]
    tmp.columns = par_cols
    it_pars_cols_name = '-'.join(map(str, vars['it_parameters']))
    tmp.columns.name = it_pars_cols_name
    tmp = tmp.T.reset_index().set_index([vars['xy_axis_columns'][0], it_pars_cols_name]).unstack(level=-1)
    tmp.columns = [h for g in tmp.columns for h in g if h != col_name]

    t_main_name = 'mean_time_over_all_' + str(vars['xy_axis_columns'][1]) + '_for_' + \
                  str(int(vars['nr_target_kps'])) + 'kpts_vs_' + str(vars['xy_axis_columns'][0]) + '_for_opts_' + \
                  '-'.join(map(str, vars['it_parameters']))
    t_mean_name = 'data_' + t_main_name + '.csv'
    ft_mean_name = os.path.join(vars['tdata_folder'], t_mean_name)
    with open(ft_mean_name, 'a') as f:
        f.write('# Mean execution times over all ' + str(vars['xy_axis_columns'][1]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    title = 'Mean Execution Times for Parameter Variations of ' + vars['sub_title_it_pars'] + ' Over All ' + \
            replaceCSVLabels(str(vars['xy_axis_columns'][1]), True, True) + \
            ' Extrapolated for ' + str(int(vars['nr_target_kps'])) + ' Keypoints'
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': False,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    stats_all = tmp.stack().reset_index()
    stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
    # figure types: sharp plot, smooth, const plot, ybar, xbar
    use_limits = {'miny': None, 'maxy': None}
    if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
        use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
    if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
        use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
    reltex_name = os.path.join(vars['rel_data_path'], t_mean_name)
    fig_name = 'Mean execution times for parameter ariations of\\\\' + strToLower(vars['sub_title_it_pars']) + ' over all ' + \
               replaceCSVLabels(str(vars['xy_axis_columns'][1]), True, False) + \
               '\\\\extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': fig_name,
                                  # If caption is None, the field name is used
                                  'caption': fig_name.replace('\\\\', ' '),
                                  'fig_type': 'smooth',
                                  'plots': list(tmp.columns.values),
                                  'label_y': 'mean execution times/$\\mu s$',
                                  'plot_x': str(vars['xy_axis_columns'][0]),
                                  'label_x': replaceCSVLabels(str(vars['xy_axis_columns'][0])),
                                  'limits': use_limits,
                                  'legend': [tex_string_coding_style(a) for a in list(tmp.columns.values)],
                                  'legend_cols': None,
                                  'use_marks': vars['use_marks'],
                                  'use_log_y_axis': True
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    if vars['build_pdf'][0]:
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

    t_main_name = 'min_mean_time_over_all_' + str(vars['xy_axis_columns'][1]) + '_for_' + \
                  str(int(vars['nr_target_kps'])) + 'kpts_vs_' + str(vars['xy_axis_columns'][0]) + '_for_opts_' + \
                  '-'.join(map(str, vars['it_parameters']))
    t_min_name = 'data_' + t_main_name + '.csv'
    ft_min_name = os.path.join(vars['tdata_folder'], t_min_name)
    with open(ft_min_name, 'a') as f:
        f.write('# Minimum execution times over parameter variations of ' + '-'.join(vars['it_parameters']) +
                ' for mean execution times over all ' + str(vars['xy_axis_columns'][1]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        tmp1.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    title = 'Minimum Execution Times Over Parameter Variations of ' + vars['sub_title_it_pars'] + \
            ' for Mean Execution Times Over All ' + \
            replaceCSVLabels(str(vars['xy_axis_columns'][1]), True, True) + \
            ' Extrapolated for ' + str(int(vars['nr_target_kps'])) + ' Keypoints'
    # Get data for tex file generation
    if len(tmp1.columns) > 10:
        fig_type = 'xbar'
    else:
        fig_type = 'ybar'
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    section_name = 'Minimum execution times over parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   ' for mean execution times over all ' + \
                   replaceCSVLabels(str(vars['xy_axis_columns'][1]), True) + \
                   '\\\\extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Minimum execution times over parameter variations of ' + strToLower(vars['sub_title_it_pars']) + \
              ' (corresponding parameter on top of bar) for mean execution times over all ' + \
              replaceCSVLabels(str(vars['xy_axis_columns'][1]), True) + \
              'extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints.'
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_min_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': fig_type,
                                  'plots': [col_name],
                                  # Label of the value axis. For xbar it labels the x-axis
                                  'label_y': 'Minimum time/$\\mu s$',
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': replaceCSVLabels(str(vars['xy_axis_columns'][0])),
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': str(vars['xy_axis_columns'][0]),
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': ['pars_tex'],
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 90,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': None,
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': False,
                                  'use_log_y_axis': True,
                                  'large_meta_space_needed': True,
                                  'caption': caption
                                  })
    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if vars['build_pdf'][1]:
        res1 = compile_tex(rendered_tex,
                           vars['tex_folder'],
                           texf_name,
                           tex_infos['make_index'],
                           os.path.join(vars['pdf_folder'], pdf_name),
                           tex_infos['figs_externalize'])
    else:
        res1 = compile_tex(rendered_tex, vars['tex_folder'], texf_name)
    if res1 != 0:
        res += abs(res1)
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    tmp1.reset_index(inplace=True)
    data_min_c = tmp1[index_name].value_counts()
    if data_min_c.count() > 1:
        if data_min_c.iloc[0] == data_min_c.iloc[1]:
            data_min_c = data_min_c.loc[data_min_c == data_min_c.iloc[0]]
            data_min2 = tmp1.loc[tmp1[index_name].isin(data_min_c.index.values)]
            data_min2 = data_min2.loc[data_min2[vars['xy_axis_columns'][0]].idxmax()]
            alg = str(data_min2[index_name])
        else:
            data_min2 = tmp1.loc[tmp1[index_name] == data_min_c.index.values[0]]
            if len(data_min2.shape) == 1:
                alg = str(data_min2[index_name])
            else:
                alg = str(data_min2[index_name].iloc[0])
    else:
        alg = str(tmp1[index_name].iloc[0])

    main_parameter_name = vars['res_par_name']#'USAC_opt_refine_min_time'
    # Check if file and parameters exist
    ppar_file, res = check_par_file_exists(main_parameter_name, vars['res_folder'], res)

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = alg.split('-')
        if len(vars['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(vars['it_parameters']):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: alg_w},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    return res

def get_time_fixed_kp(**vars):
    if 't_data_separators' not in vars:
        raise ValueError('No data separators specified for calculating time budget')
    drop_cols = []
    if 'partitions' in vars:
        if 'x_axis_column' in vars:
            for sep in vars['t_data_separators']:
                if sep not in vars['partitions'] and sep not in vars['x_axis_column']:
                    raise ValueError('Data separator ' + str(sep) + ' not found in partitions or x_axis_column')
            drop_cols = list(set(vars['partitions'] + vars['x_axis_column']).difference(vars['t_data_separators']))
        elif 'xy_axis_columns' in vars:
            for sep in vars['t_data_separators']:
                if sep not in vars['partitions'] and sep not in vars['xy_axis_columns']:
                    raise ValueError('Data separator ' + str(sep) + ' not found in partitions or xy_axis_columns')
            drop_cols = list(set(vars['partitions'] + vars['xy_axis_columns']).difference(vars['t_data_separators']))
        else:
            raise ValueError('Either x_axis_column or xy_axis_columns must be provided')
    elif 'x_axis_column' in vars:
        for sep in vars['t_data_separators']:
            if sep not in vars['x_axis_column']:
                raise ValueError('Data separator ' + str(sep) + ' not found in x_axis_column')
    elif 'xy_axis_columns' in vars:
        for sep in vars['t_data_separators']:
            if sep not in vars['xy_axis_columns']:
                raise ValueError('Data separator ' + str(sep) + ' not found in xy_axis_columns')
        drop_cols = list(set(vars['xy_axis_columns']).difference(vars['t_data_separators']))
    else:
        raise ValueError('Either x_axis_column or xy_axis_columns must be provided')
    individual_grps = vars['it_parameters'] + vars['t_data_separators']
    df = vars['data'].groupby(individual_grps).mean()
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
    if len(vars['eval_columns']) > 3:
        tmp = df[vars['eval_columns'][1]] + vars['nr_target_kps'] * df[vars['eval_columns'][2]] +\
              (vars['nr_target_kps'] * vars['nr_target_kps']) * df[vars['eval_columns'][3]]
    else:
        tmp = df[vars['eval_columns'][1]] + vars['nr_target_kps'] * df[vars['eval_columns'][2]]
    col_name = 't_' + str(int(vars['nr_target_kps'])) + 'kpts'
    tmp.rename(col_name, inplace=True)
    tmp = tmp.reset_index()
    return tmp, col_name

def prepare_io(**keywords):
    if 'res_folder' not in keywords:
        raise ValueError('Missing res_folder argument of function estimate_alg_time_fixed_kp')
    if 'use_marks' not in keywords:
        print('No information provided if marks should be used: Disabling marks')
        keywords['use_marks'] = False
    if 'build_pdf' not in keywords:
        keywords['build_pdf'] = (False, True,)
    if len(keywords['build_pdf']) != 2:
        raise ValueError('Wrong number of arguments for build_pdf')
    keywords['pdf_folder'] = None
    if keywords['build_pdf'][0] or keywords['build_pdf'][1]:
        keywords['pdf_folder'] = os.path.join(keywords['res_folder'], 'pdf')
        try:
            os.mkdir(keywords['pdf_folder'])
        except FileExistsError:
            # print('Folder', ret['pdf_folder'], 'for storing pdf files already exists')
            pass
    keywords['tex_folder'] = os.path.join(keywords['res_folder'], 'tex')
    try:
        os.mkdir(keywords['tex_folder'])
    except FileExistsError:
        # print('Folder', ret['tex_folder'], 'for storing tex files already exists')
        pass
    keywords['tdata_folder'] = os.path.join(keywords['tex_folder'], 'data')
    try:
        os.mkdir(keywords['tdata_folder'])
    except FileExistsError:
        # print('Folder', ret['tdata_folder'], 'for storing data files already exists')
        pass
    keywords['rel_data_path'] = os.path.relpath(keywords['tdata_folder'], keywords['tex_folder'])
    nr_it_parameters = len(keywords['it_parameters'])
    from statistics_and_plot import replaceCSVLabels
    keywords['sub_title_it_pars'] = ''
    for i, val in enumerate(keywords['it_parameters']):
        keywords['sub_title_it_pars'] += replaceCSVLabels(val, True, True)
        if nr_it_parameters <= 2:
            if i < nr_it_parameters - 1:
                keywords['sub_title_it_pars'] += ' and '
        else:
            if i < nr_it_parameters - 2:
                keywords['sub_title_it_pars'] += ', '
            elif i < nr_it_parameters - 1:
                keywords['sub_title_it_pars'] += ', and '
    return keywords

def estimate_alg_time_fixed_kp_for_props(**vars):
    if 'res_par_name' not in vars:
        raise ValueError('Missing parameter res_par_name')
    tmp, col_name = get_time_fixed_kp(**vars)
    tmp[vars['t_data_separators']] = tmp[vars['t_data_separators']].round(3)
    tmp[col_name] = tmp[col_name].round(0)
    tmp1min = tmp.loc[tmp.groupby(vars['it_parameters'] + [vars['t_data_separators'][1]])[col_name].idxmin(axis=0)]
    tmp1max = tmp.loc[tmp.groupby(vars['it_parameters'] + [vars['t_data_separators'][1]])[col_name].idxmax(axis=0)]
    tmp2min = tmp1min.loc[tmp1min.groupby(vars['it_parameters'])[col_name].idxmin(axis=0)]
    tmp2max = tmp1max.loc[tmp1max.groupby(vars['it_parameters'])[col_name].idxmax(axis=0)]

    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols, replaceCSVLabels, strToLower
    tmp1min.set_index(vars['it_parameters'], inplace=True)
    index_new1 = ['-'.join(a) for a in tmp1min.index]
    tmp1min.index = index_new1
    index_name = '-'.join(vars['it_parameters'])
    tmp1min.index.name = index_name
    # pars_tex1 = insert_opt_lbreak(index_new1)
    pars_tex1 = [tex_string_coding_style(a) for a in list(dict.fromkeys(index_new1))]
    tmp1min = tmp1min.reset_index().set_index([vars['t_data_separators'][1], index_name]).unstack(level=-1)
    comb_cols1 = ['-'.join(a) for a in tmp1min.columns]
    tmp1min.columns = comb_cols1
    val_axis_cols1 = [a for a in comb_cols1 if col_name in a]
    meta_cols1 = [a for a in comb_cols1 if vars['t_data_separators'][0] in a]

    tmp1max.set_index(vars['it_parameters'], inplace=True)
    index_new2 = ['-'.join(a) for a in tmp1max.index]
    tmp1max.index = index_new2
    tmp1max.index.name = index_name
    # pars_tex2 = insert_opt_lbreak(index_new2)
    pars_tex2 = [tex_string_coding_style(a) for a in list(dict.fromkeys(index_new2))]
    tmp1max = tmp1max.reset_index().set_index([vars['t_data_separators'][1], index_name]).unstack(level=-1)
    comb_cols2 = ['-'.join(a) for a in tmp1max.columns]
    tmp1max.columns = comb_cols2
    val_axis_cols2 = [a for a in comb_cols2 if col_name in a]
    meta_cols2 = [a for a in comb_cols2 if vars['t_data_separators'][0] in a]

    vars = prepare_io(**vars)
    t_main_name = 'time_over_all_' + str(vars['t_data_separators'][0]) + '_vs_' + str(vars['t_data_separators'][1]) + \
                  '_for_' + str(int(vars['nr_target_kps'])) + 'kpts_for_opts_' + \
                  '-'.join(map(str, vars['it_parameters']))
    t_min_name = 'data_min_' + t_main_name + '.csv'
    ft_min_name = os.path.join(vars['tdata_folder'], t_min_name)
    with open(ft_min_name, 'a') as f:
        f.write('# Minimum execution times over all ' + str(vars['t_data_separators'][0]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp1min.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    t_max_name = 'data_max_' + t_main_name + '.csv'
    ft_max_name = os.path.join(vars['tdata_folder'], t_max_name)
    with open(ft_max_name, 'a') as f:
        f.write('# Maximum execution times over all ' + str(vars['t_data_separators'][0]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp1max.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    title = 'Minimum and Maximum Execution Times vs ' + \
            replaceCSVLabels(str(vars['t_data_separators'][1]), True, True) + \
            ' for Parameter Variations of ' + vars['sub_title_it_pars'] + \
            ' Over All ' + replaceCSVLabels(str(vars['t_data_separators'][0]), True, True) + \
            ' Extrapolated for ' + str(int(vars['nr_target_kps'])) + ' Keypoints'

    # Get data for tex file generation
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    section_name = 'Minimum execution times vs ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   '\\\\over all ' + replaceCSVLabels(str(vars['t_data_separators'][0]), True) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Minimum execution times vs ' + replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' (corresponding ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' on top of each bar) extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_min_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'xbar',
                                  'plots': val_axis_cols1,
                                  # Label of the value axis. For xbar it labels the x-axis
                                  'label_y': 'Minimum time/$\\mu s$',
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': replaceCSVLabels(str(vars['t_data_separators'][1])),
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': str(vars['t_data_separators'][1]),
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': meta_cols1,
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 0,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': pars_tex1,
                                  'legend_cols': None,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': False,
                                  'use_log_y_axis': False,
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    section_name = 'Maximum execution times vs ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   '\\\\over all ' + replaceCSVLabels(str(vars['t_data_separators'][0]), True) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Maximum execution times vs ' + replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' (corresponding ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' on top of each bar) extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_max_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'xbar',
                                  'plots': val_axis_cols2,
                                  # Label of the value axis. For xbar it labels the x-axis
                                  'label_y': 'Maximum time/$\\mu s$',
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': replaceCSVLabels(str(vars['t_data_separators'][1])),
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': str(vars['t_data_separators'][1]),
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': meta_cols2,
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 0,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': pars_tex2,
                                  'legend_cols': None,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': False,
                                  'use_log_y_axis': False,
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_min_max_' + t_main_name
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if vars['build_pdf'][1]:
        res = compile_tex(rendered_tex,
                          vars['tex_folder'],
                          texf_name,
                          tex_infos['make_index'],
                          os.path.join(vars['pdf_folder'], pdf_name),
                          tex_infos['figs_externalize'])
    else:
        res = compile_tex(rendered_tex, vars['tex_folder'], texf_name)
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    tmp2min.set_index(vars['it_parameters'], inplace=True)
    index_new1 = ['-'.join(a) for a in tmp2min.index]
    tmp2min.index = index_new1
    tmp2min.index.name = index_name
    tmp2min['pars_tex'] = insert_opt_lbreak(index_new1)
    meta_col1 = str(vars['t_data_separators'][0]) + '-' + str(vars['t_data_separators'][1])
    tmp2min[meta_col1] = tmp2min.loc[:, vars['t_data_separators'][0]].apply(lambda x: str(x) + ' - ') + \
                         tmp2min.loc[:, vars['t_data_separators'][1]].apply(lambda x: str(x))
    tmp2min.drop(vars['t_data_separators'], axis=1, inplace=True)

    tmp2max.set_index(vars['it_parameters'], inplace=True)
    index_new2 = ['-'.join(a) for a in tmp2max.index]
    tmp2max.index = index_new2
    tmp2max.index.name = index_name
    tmp2max['pars_tex'] = insert_opt_lbreak(index_new2)
    meta_col2 = str(vars['t_data_separators'][0]) + '-' + str(vars['t_data_separators'][1])
    tmp2max[meta_col2] = tmp2max.loc[:, vars['t_data_separators'][0]].apply(lambda x: str(x) + ' - ') + \
                         tmp2max.loc[:, vars['t_data_separators'][1]].apply(lambda x: str(x))
    tmp2max.drop(vars['t_data_separators'], axis=1, inplace=True)

    t_main_name = 'time_over_all_' + str(vars['t_data_separators'][0]) + '_and_' + str(vars['t_data_separators'][1]) + \
                  '_for_' + str(int(vars['nr_target_kps'])) + 'kpts_for_opts_' + \
                  '-'.join(map(str, vars['it_parameters']))
    t_min_name = 'data_min_' + t_main_name + '.csv'
    ft_min_name = os.path.join(vars['tdata_folder'], t_min_name)
    with open(ft_min_name, 'a') as f:
        f.write('# Minimum execution times over all ' + str(vars['t_data_separators'][0]) + ' and ' +
                str(vars['t_data_separators'][1]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp2min.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    t_max_name = 'data_max_' + t_main_name + '.csv'
    ft_max_name = os.path.join(vars['tdata_folder'], t_max_name)
    with open(ft_max_name, 'a') as f:
        f.write('# Maximum execution times over all ' + str(vars['t_data_separators'][0]) + ' and ' +
                str(vars['t_data_separators'][1]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp2max.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    title = 'Minimum and Maximum Execution Times over all ' + \
            replaceCSVLabels(str(vars['t_data_separators'][0]), True, True) + ' and ' + \
            replaceCSVLabels(str(vars['t_data_separators'][1]), True, True) + \
            ' for Parameter Variations of ' + vars['sub_title_it_pars'] + \
            ' Extrapolated for ' + str(int(vars['nr_target_kps'])) + ' Keypoints'
    # Get data for tex file generation
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    section_name = 'Minimum execution times over all ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' and ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Minimum execution times over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' and ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1]), True) + ' (corresponding '  + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' -- ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1])) + ' values on top of each bar)' + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_min_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'xbar',
                                  'plots': [col_name],
                                  # Label of the value axis. For xbar it labels the x-axis
                                  'label_y': 'Minimum time/$\\mu s$',
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'Parameter combination',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'pars_tex',
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': [meta_col1],
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
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    section_name = 'Maximum execution times over all ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' and ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Maximum execution times over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' and ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1]), True) + ' (corresponding ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' -- ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1])) + ' values on top of each bar)' + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_max_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'xbar',
                                  'plots': [col_name],
                                  # Label of the value axis. For xbar it labels the x-axis
                                  'label_y': 'Maximum time/$\\mu s$',
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'Parameter combination',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'pars_tex',
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': [meta_col2],
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
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_min_max_' + t_main_name
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if vars['build_pdf'][1]:
        res1 = compile_tex(rendered_tex,
                           vars['tex_folder'],
                           texf_name,
                           tex_infos['make_index'],
                           os.path.join(vars['pdf_folder'], pdf_name),
                           tex_infos['figs_externalize'])
    else:
        res1 = compile_tex(rendered_tex, vars['tex_folder'], texf_name)
    if res1 != 0:
        res += abs(res1)
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    min_t = tmp2min.loc[[tmp2min[col_name].idxmin()]].reset_index()

    main_parameter_name = vars['res_par_name']#'USAC_opt_search_min_time_inlrat_th'
    # Check if file and parameters exist
    ppar_file, res = check_par_file_exists(main_parameter_name, vars['res_folder'], res)

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = str(min_t[index_name].values[0]).split('-')
        if len(vars['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(vars['it_parameters']):
            alg_w[val] = alg_comb_bestl[i]
        par1 = str(min_t[meta_col2].values[0]).split(' - ')
        par_name = meta_col2.split('-')
        yaml.dump({main_parameter_name: {'Algorithm': alg_w,
                                         par_name[0]: par1[0],
                                         par_name[1]: par1[1],
                                         'Time_us': float(min_t[col_name].values[0])}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    return res


def estimate_alg_time_fixed_kp_for_3_props(**vars):
    if 'res_par_name' not in vars:
        raise ValueError('Missing parameter res_par_name')
    tmp, col_name = get_time_fixed_kp(**vars)
    tmp[vars['t_data_separators']] = tmp[vars['t_data_separators']].round(3)
    tmp[col_name] = tmp[col_name].round(0)
    first_grp2 = [a for a in vars['t_data_separators'] if a != vars['accum_step_props'][0]]
    first_grp = vars['it_parameters'] + first_grp2
    tmp1mean = tmp.groupby(first_grp).mean().drop(vars['accum_step_props'][0], axis=1)
    second_grp2 = [a for a in vars['t_data_separators'] if a != vars['accum_step_props'][1]]
    second_grp = vars['it_parameters'] + second_grp2
    tmp2mean = tmp.groupby(second_grp).mean().drop(vars['accum_step_props'][1], axis=1)
    minmax_grp = vars['it_parameters'] + [vars['eval_minmax_for']]
    tmp1mean_min = tmp1mean.loc[tmp1mean.groupby(minmax_grp)[col_name].idxmin(axis=0)]
    tmp1mean_max = tmp1mean.loc[tmp1mean.groupby(minmax_grp)[col_name].idxmax(axis=0)]
    tmp2mean_min = tmp2mean.loc[tmp2mean.groupby(minmax_grp)[col_name].idxmin(axis=0)]
    tmp2mean_max = tmp2mean.loc[tmp2mean.groupby(minmax_grp)[col_name].idxmax(axis=0)]
    tmp12_min = tmp1mean_min.loc[tmp1mean_min[col_name].idxmin(axis=0)]
    tmp12_max = tmp1mean_max.loc[tmp1mean_max[col_name].idxmax(axis=0)]
    tmp22_min = tmp2mean_min.loc[tmp2mean_min[col_name].idxmin(axis=0)]
    tmp22_max = tmp2mean_max.loc[tmp2mean_max[col_name].idxmax(axis=0)]

    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols, replaceCSVLabels, strToLower
    tmp1mean.set_index(vars['it_parameters'], inplace=True)
    index_new1 = ['-'.join(a) for a in tmp1mean.index]
    tmp1mean.index = index_new1
    index_name = '-'.join(vars['it_parameters'])
    tmp1mean.index.name = index_name
    tmp1mean = tmp1mean.reset_index().set_index(first_grp2 + index_name).unstack(level=-1)
    index_new11 = ['-'.join(a) for a in tmp1mean.columns]
    legend1 =
    tmp1mean.columns =  index_new11
    tmp1mean.reset_index(inplace=True)

    tmp2mean.set_index(vars['it_parameters'], inplace=True)
    index_new2 = ['-'.join(a) for a in tmp2mean.index]
    tmp2mean.index = index_new2
    tmp2mean.index.name = index_name
    tmp2mean = tmp2mean.reset_index().set_index(second_grp2 + index_name).unstack(level=-1)
    index_new21 = ['-'.join(a) for a in tmp2mean.columns]
    tmp2mean.columns = index_new21
    tmp2mean.reset_index(inplace=True)



    # pars_tex1 = insert_opt_lbreak(index_new1)
    pars_tex1 = [tex_string_coding_style(a) for a in list(dict.fromkeys(index_new1))]
    tmp1min = tmp1min.reset_index().set_index([vars['t_data_separators'][1], index_name]).unstack(level=-1)
    comb_cols1 = ['-'.join(a) for a in tmp1min.columns]
    tmp1min.columns = comb_cols1
    val_axis_cols1 = [a for a in comb_cols1 if col_name in a]
    meta_cols1 = [a for a in comb_cols1 if vars['t_data_separators'][0] in a]

    tmp1max.set_index(vars['it_parameters'], inplace=True)
    index_new2 = ['-'.join(a) for a in tmp1max.index]
    tmp1max.index = index_new2
    tmp1max.index.name = index_name
    # pars_tex2 = insert_opt_lbreak(index_new2)
    pars_tex2 = [tex_string_coding_style(a) for a in list(dict.fromkeys(index_new2))]
    tmp1max = tmp1max.reset_index().set_index([vars['t_data_separators'][1], index_name]).unstack(level=-1)
    comb_cols2 = ['-'.join(a) for a in tmp1max.columns]
    tmp1max.columns = comb_cols2
    val_axis_cols2 = [a for a in comb_cols2 if col_name in a]
    meta_cols2 = [a for a in comb_cols2 if vars['t_data_separators'][0] in a]

    vars = prepare_io(**vars)
    t_main_name = 'time_over_all_' + str(vars['t_data_separators'][0]) + '_vs_' + str(vars['t_data_separators'][1]) + \
                  '_for_' + str(int(vars['nr_target_kps'])) + 'kpts_for_opts_' + \
                  '-'.join(map(str, vars['it_parameters']))
    t_min_name = 'data_min_' + t_main_name + '.csv'
    ft_min_name = os.path.join(vars['tdata_folder'], t_min_name)
    with open(ft_min_name, 'a') as f:
        f.write('# Minimum execution times over all ' + str(vars['t_data_separators'][0]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp1min.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    t_max_name = 'data_max_' + t_main_name + '.csv'
    ft_max_name = os.path.join(vars['tdata_folder'], t_max_name)
    with open(ft_max_name, 'a') as f:
        f.write('# Maximum execution times over all ' + str(vars['t_data_separators'][0]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp1max.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    title = 'Minimum and Maximum Execution Times vs ' + \
            replaceCSVLabels(str(vars['t_data_separators'][1]), True, True) + \
            ' for Parameter Variations of ' + vars['sub_title_it_pars'] + \
            ' Over All ' + replaceCSVLabels(str(vars['t_data_separators'][0]), True, True) + \
            ' Extrapolated for ' + str(int(vars['nr_target_kps'])) + ' Keypoints'

    # Get data for tex file generation
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    section_name = 'Minimum execution times vs ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   '\\\\over all ' + replaceCSVLabels(str(vars['t_data_separators'][0]), True) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Minimum execution times vs ' + replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' (corresponding ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' on top of each bar) extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_min_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'xbar',
                                  'plots': val_axis_cols1,
                                  # Label of the value axis. For xbar it labels the x-axis
                                  'label_y': 'Minimum time/$\\mu s$',
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': replaceCSVLabels(str(vars['t_data_separators'][1])),
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': str(vars['t_data_separators'][1]),
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': meta_cols1,
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 0,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': pars_tex1,
                                  'legend_cols': None,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': False,
                                  'use_log_y_axis': False,
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    section_name = 'Maximum execution times vs ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   '\\\\over all ' + replaceCSVLabels(str(vars['t_data_separators'][0]), True) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Maximum execution times vs ' + replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' (corresponding ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' on top of each bar) extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_max_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'xbar',
                                  'plots': val_axis_cols2,
                                  # Label of the value axis. For xbar it labels the x-axis
                                  'label_y': 'Maximum time/$\\mu s$',
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': replaceCSVLabels(str(vars['t_data_separators'][1])),
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': str(vars['t_data_separators'][1]),
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': meta_cols2,
                                  # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                  'rotate_meta': 0,
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': pars_tex2,
                                  'legend_cols': None,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': False,
                                  'use_log_y_axis': False,
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_min_max_' + t_main_name
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if vars['build_pdf'][1]:
        res = compile_tex(rendered_tex,
                          vars['tex_folder'],
                          texf_name,
                          tex_infos['make_index'],
                          os.path.join(vars['pdf_folder'], pdf_name),
                          tex_infos['figs_externalize'])
    else:
        res = compile_tex(rendered_tex, vars['tex_folder'], texf_name)
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    tmp2min.set_index(vars['it_parameters'], inplace=True)
    index_new1 = ['-'.join(a) for a in tmp2min.index]
    tmp2min.index = index_new1
    tmp2min.index.name = index_name
    tmp2min['pars_tex'] = insert_opt_lbreak(index_new1)
    meta_col1 = str(vars['t_data_separators'][0]) + '-' + str(vars['t_data_separators'][1])
    tmp2min[meta_col1] = tmp2min.loc[:, vars['t_data_separators'][0]].apply(lambda x: str(x) + ' - ') + \
                         tmp2min.loc[:, vars['t_data_separators'][1]].apply(lambda x: str(x))
    tmp2min.drop(vars['t_data_separators'], axis=1, inplace=True)

    tmp2max.set_index(vars['it_parameters'], inplace=True)
    index_new2 = ['-'.join(a) for a in tmp2max.index]
    tmp2max.index = index_new2
    tmp2max.index.name = index_name
    tmp2max['pars_tex'] = insert_opt_lbreak(index_new2)
    meta_col2 = str(vars['t_data_separators'][0]) + '-' + str(vars['t_data_separators'][1])
    tmp2max[meta_col2] = tmp2max.loc[:, vars['t_data_separators'][0]].apply(lambda x: str(x) + ' - ') + \
                         tmp2max.loc[:, vars['t_data_separators'][1]].apply(lambda x: str(x))
    tmp2max.drop(vars['t_data_separators'], axis=1, inplace=True)

    t_main_name = 'time_over_all_' + str(vars['t_data_separators'][0]) + '_and_' + str(vars['t_data_separators'][1]) + \
                  '_for_' + str(int(vars['nr_target_kps'])) + 'kpts_for_opts_' + \
                  '-'.join(map(str, vars['it_parameters']))
    t_min_name = 'data_min_' + t_main_name + '.csv'
    ft_min_name = os.path.join(vars['tdata_folder'], t_min_name)
    with open(ft_min_name, 'a') as f:
        f.write('# Minimum execution times over all ' + str(vars['t_data_separators'][0]) + ' and ' +
                str(vars['t_data_separators'][1]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp2min.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    t_max_name = 'data_max_' + t_main_name + '.csv'
    ft_max_name = os.path.join(vars['tdata_folder'], t_max_name)
    with open(ft_max_name, 'a') as f:
        f.write('# Maximum execution times over all ' + str(vars['t_data_separators'][0]) + ' and ' +
                str(vars['t_data_separators'][1]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp2max.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    title = 'Minimum and Maximum Execution Times over all ' + \
            replaceCSVLabels(str(vars['t_data_separators'][0]), True, True) + ' and ' + \
            replaceCSVLabels(str(vars['t_data_separators'][1]), True, True) + \
            ' for Parameter Variations of ' + vars['sub_title_it_pars'] + \
            ' Extrapolated for ' + str(int(vars['nr_target_kps'])) + ' Keypoints'
    # Get data for tex file generation
    tex_infos = {'title': title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    section_name = 'Minimum execution times over all ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' and ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Minimum execution times over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' and ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1]), True) + ' (corresponding '  + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' -- ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1])) + ' values on top of each bar)' + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_min_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'xbar',
                                  'plots': [col_name],
                                  # Label of the value axis. For xbar it labels the x-axis
                                  'label_y': 'Minimum time/$\\mu s$',
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'Parameter combination',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'pars_tex',
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': [meta_col1],
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
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    section_name = 'Maximum execution times over all ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' and ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Maximum execution times over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' and ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1]), True) + ' (corresponding ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' -- ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1])) + ' values on top of each bar)' + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_max_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': 'xbar',
                                  'plots': [col_name],
                                  # Label of the value axis. For xbar it labels the x-axis
                                  'label_y': 'Maximum time/$\\mu s$',
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'Parameter combination',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'pars_tex',
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': [meta_col2],
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
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_min_max_' + t_main_name
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if vars['build_pdf'][1]:
        res1 = compile_tex(rendered_tex,
                           vars['tex_folder'],
                           texf_name,
                           tex_infos['make_index'],
                           os.path.join(vars['pdf_folder'], pdf_name),
                           tex_infos['figs_externalize'])
    else:
        res1 = compile_tex(rendered_tex, vars['tex_folder'], texf_name)
    if res1 != 0:
        res += abs(res1)
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    min_t = tmp2min.loc[[tmp2min[col_name].idxmin()]].reset_index()

    main_parameter_name = vars['res_par_name']#'USAC_opt_search_min_time_inlrat_th'
    # Check if file and parameters exist
    ppar_file, res = check_par_file_exists(main_parameter_name, vars['res_folder'], res)

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = str(min_t[index_name].values[0]).split('-')
        if len(vars['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(vars['it_parameters']):
            alg_w[val] = alg_comb_bestl[i]
        par1 = str(min_t[meta_col2].values[0]).split(' - ')
        par_name = meta_col2.split('-')
        yaml.dump({main_parameter_name: {'Algorithm': alg_w,
                                         par_name[0]: par1[0],
                                         par_name[1]: par1[1],
                                         'Time_us': float(min_t[col_name].values[0])}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    return res


def get_inlrat_diff(**vars):
    if len(vars['eval_columns']) != 2:
        raise ValueError('For calculating the difference of inlier ratios, eval_columns must hold 2 entries')
    needed_columns = vars['eval_columns'] + vars['it_parameters'] + \
                     vars['x_axis_column'] + vars['partitions']
    data = vars['data'].loc[:, needed_columns]
    eval_columns = ['inlRat_diff']
    data['inlRat_diff'] = data[vars['eval_columns'][0]] - data[vars['eval_columns'][1]]
    data.drop(vars['eval_columns'], axis=1, inplace=True)
    ret = {'data': data,
           'eval_columns': eval_columns,
           'it_parameters': vars['it_parameters'],
           'x_axis_column': vars['x_axis_column'],
           'partitions': vars['partitions']}
    return ret


def get_min_inlrat_diff(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    if len(keywords) < 4 or len(keywords) > 6:
        raise ValueError('Wrong number of arguments for function pars_calc_single_fig_partitions')
    if 'data' not in keywords:
        raise ValueError('Missing data argument of function pars_calc_single_fig_partitions')
    if 'partitions' not in keywords:
        raise ValueError('Missing partitions argument of function pars_calc_single_fig_partitions')
    data = keywords['data']
    partitions = keywords['partitions']
    keywords = prepare_io(**keywords)
    # if 'res_folder' not in keywords:
    #     raise ValueError('Missing res_folder argument of function pars_calc_single_fig_partitions')
    # res_folder = keywords['res_folder']
    # use_marks = False
    # if 'use_marks' not in keywords:
    #     print('No information provided if marks should be used: Disabling marks')
    # else:
    #     use_marks = keywords['use_marks']
    # build_pdf = (False, True,)
    # if 'build_pdf' in keywords:
    #     build_pdf = keywords['build_pdf']
    # if len(build_pdf) != 2:
    #     raise ValueError('Wrong number of arguments for build_pdf')
    # pdf_folder = None
    # if build_pdf[0] or build_pdf[1]:
    #     pdf_folder = os.path.join(res_folder, 'pdf')
    #     try:
    #         os.mkdir(pdf_folder)
    #     except FileExistsError:
    #         # print('Folder', pdf_folder, 'for storing pdf files already exists')
    #         pass
    # tex_folder = os.path.join(res_folder, 'tex')
    # try:
    #     os.mkdir(tex_folder)
    # except FileExistsError:
    #     # print('Folder', tex_folder, 'for storing tex files already exists')
    #     pass
    # tdata_folder = os.path.join(tex_folder, 'data')
    # try:
    #     os.mkdir(tdata_folder)
    # except FileExistsError:
    #     # print('Folder', tdata_folder, 'for storing data files already exists')
    #     pass
    # rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    grp_names = data.index.names
    nr_partitions = len(partitions)
    it_parameters = grp_names[nr_partitions:-1]
    nr_it_parameters = len(it_parameters)
    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols, replaceCSVLabels, strToLower
    # sub_title_it_pars = ''
    # for i, val in enumerate(it_parameters):
    #     sub_title_it_pars += replaceCSVLabels(val, True, True)
    #     if nr_it_parameters <= 2:
    #         if i < nr_it_parameters - 1:
    #             sub_title_it_pars += ' and '
    #     else:
    #         if i < nr_it_parameters - 2:
    #             sub_title_it_pars += ', '
    #         elif i < nr_it_parameters - 1:
    #             sub_title_it_pars += ', and '

    dataf_name_main = str(grp_names[-1]) + '_for_options_' + '-'.join(it_parameters)
    hlp = [a for a in data.columns.values if 'mean' in a]
    if len(hlp) != 1 or len(hlp[0]) != 2:
        raise ValueError('Wrong DataFrame format for inlier ratio difference statistics')
    diff_mean = data[hlp[0]]
    diff_mean.name = hlp[0][0]
    diff_mean = diff_mean.abs().reset_index().drop(partitions, axis=1).groupby(it_parameters +
                                                                               [grp_names[-1]]).mean().unstack()
    diff_mean.index = ['-'.join(map(str,a)) for a in diff_mean.index]
    it_parameters_name = '-'.join(it_parameters)
    diff_mean.index.name = it_parameters_name
    min_mean_diff = diff_mean.stack().reset_index()
    diff_mean = diff_mean.T.reset_index()
    diff_mean.drop(diff_mean.columns[0], axis=1, inplace=True)
    diff_mean.set_index(grp_names[-1], inplace=True)

    b_name = 'data_mean_inlrat_diff_vs_' + dataf_name_main + '.csv'
    fb_name = os.path.join(keywords['tdata_folder'], b_name)
    with open(fb_name, 'a') as f:
        f.write('# Absolute mean inlier ratio differences vs ' + str(grp_names[-1]) + '\n')
        f.write('# Parameters: ' + it_parameters_name + '\n')
        diff_mean.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    tex_infos = {'title': 'Absolute Mean Inlier Ratio Differences vs ' +
                          replaceCSVLabels(str(grp_names[-1]), True, True) +
                          ' for Parameter Variations of ' + keywords['sub_title_it_pars'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    stats_all = diff_mean.stack().reset_index()
    stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
    # figure types: sharp plot, smooth, const plot, ybar, xbar
    use_limits = {'miny': None, 'maxy': None}
    if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
        use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
    if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
        use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
    reltex_name = os.path.join(keywords['rel_data_path'], b_name)
    fig_name = 'Absolute mean inlier ratio differences vs ' + replaceCSVLabels(str(grp_names[-1]), True) + \
               ' for parameter variations of \\\\' + keywords['sub_title_it_pars']
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': fig_name,
                                  # If caption is None, the field name is used
                                  'caption': fig_name.replace('\\\\', ' '),
                                  'fig_type': 'smooth',
                                  'plots': list(diff_mean.columns.values),
                                  'label_y': 'Absolute mean $\\Delta \\epsilon$',
                                  'plot_x': str(grp_names[-1]),
                                  'label_x': replaceCSVLabels(str(grp_names[-1])),
                                  'limits': use_limits,
                                  'legend': [tex_string_coding_style(a) for a in list(diff_mean.columns.values)],
                                  'legend_cols': None,
                                  'use_marks': keywords['use_marks'],
                                  'use_log_y_axis': False,
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_mean_inlrat_diff_vs_' + dataf_name_main
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

    min_mean_diff = min_mean_diff.loc[min_mean_diff.groupby(it_parameters_name).inlRat_diff.idxmin()]
    min_mean_diff['options_for_tex'] = [tex_string_coding_style(a)
                                        for i, a in min_mean_diff[it_parameters_name].iteritems()]
    b_name = 'data_min_mean_inlrat_diff_vs_' + dataf_name_main + '.csv'
    fb_name = os.path.join(keywords['tdata_folder'], b_name)
    with open(fb_name, 'a') as f:
        f.write('# Minimum absolute mean inlier ratio differences vs ' + str(grp_names[-1]) +
                ' for every ' + it_parameters_name + ' combination\n')
        min_mean_diff.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

    # Get data for tex file generation
    if min_mean_diff.shape[0] > 10:
        fig_type = 'xbar'
    else:
        fig_type = 'ybar'
    tex_infos = {'title': 'Minimum Absolute Mean Inlier Ratio Difference and its Corresponding ' +
                          replaceCSVLabels(str(grp_names[-1]), False, True) +
                          ' for Parameter Variations of ' + keywords['sub_title_it_pars'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True
                 }
    section_name = 'Minimum absolute mean inlier ratio difference\\\\and its corresponding ' + \
                   replaceCSVLabels(str(grp_names[-1])) + \
                   ' for parameter variations of\\\\' + strToLower(keywords['sub_title_it_pars'])
    caption = 'Minimum absolute mean inlier ratio difference and its corresponding ' + \
              replaceCSVLabels(str(grp_names[-1])) + \
              ' (on top of each bar) for parameter variations of ' + strToLower(keywords['sub_title_it_pars'])
    tex_infos['sections'].append({'file': os.path.join(keywords['rel_data_path'], b_name),
                                  'name': section_name.replace('\\\\', ' '),
                                  'title': section_name,
                                  'title_rows': section_name.count('\\\\'),
                                  'fig_type': fig_type,
                                  'plots': ['inlRat_diff'],
                                  # Label of the value axis. For xbar it labels the x-axis
                                  'label_y': 'Min. absolute mean inlier ratio difference',
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'USAC options',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'options_for_tex',
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': [str(grp_names[-1])],
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
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_min_mean_inlrat_diff_vs_' + dataf_name_main
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

    main_parameter_name = 'USAC_opt_search_min_inlrat_diff'
    # Check if file and parameters exist
    ppar_file, res = check_par_file_exists(main_parameter_name, keywords['res_folder'], res)

    min_diff = min_mean_diff.loc[[min_mean_diff['inlRat_diff'].idxmin()]]

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = str(min_diff[it_parameters_name].values[0]).split('-')
        if len(it_parameters) != len(alg_comb_bestl):
            raise ValueError('Nr of search algorithms does not match')
        alg_w = {}
        for i, val in enumerate(it_parameters):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithm': alg_w,
                                         str(grp_names[-1]): float(min_diff[grp_names[-1]].values[0]),
                                         'inlRatDiff': float(min_diff['inlRat_diff'].values[0])}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    return res
