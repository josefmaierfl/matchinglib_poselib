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


def pars_calc_single_fig(**keywords):
    if len(keywords) < 2 or len(keywords) > 4:
        raise ValueError('Wrong number of arguments for function get_best_comb_and_th_1')
    if 'data' not in keywords:
        raise ValueError('Missing data argument of function get_best_comb_and_th_1')
    data = keywords['data']
    res = {}
    if 'res_folder' not in keywords:
        raise ValueError('Missing res_folder argument of function get_best_comb_and_th_1')
    res['res_folder'] = keywords['res_folder']
    res['use_marks'] = False
    if 'use_marks' not in keywords:
        print('No information provided if marks should be used: Disabling marks')
    else:
        res['use_marks'] = keywords['use_marks']
    res['build_pdf'] = (False, True,)
    if 'build_pdf' in keywords:
        res['build_pdf'] = keywords['build_pdf']
    if len(res['build_pdf']) != 2:
        raise ValueError('Wrong number of arguments for build_pdf')
    res['pdf_folder'] = None
    if res['build_pdf'][0] or res['build_pdf'][1]:
        res['pdf_folder'] = os.path.join(res['res_folder'], 'pdf')
        try:
            os.mkdir(res['pdf_folder'])
        except FileExistsError:
            # print('Folder', res['pdf_folder'], 'for storing pdf files already exists')
            pass
    res['tex_folder'] = os.path.join(res['res_folder'], 'tex')
    try:
        os.mkdir(res['tex_folder'])
    except FileExistsError:
        # print('Folder', res['tex_folder'], 'for storing tex files already exists')
        pass
    res['tdata_folder'] = os.path.join(res['tex_folder'], 'data')
    try:
        os.mkdir(res['tdata_folder'])
    except FileExistsError:
        # print('Folder', res['tdata_folder'], 'for storing data files already exists')
        pass
    rel_data_path = os.path.relpath(res['tdata_folder'], res['tex_folder'])
    grp_names = data.index.names
    dataf_name_main = str(grp_names[-1]) + '_for_options_' + '-'.join(grp_names[0:-1])
    dataf_name = dataf_name_main + '.csv'
    b = combineRt(data)
    b = b.T
    b.columns = ['-'.join(map(str, a)) for a in b.columns]
    b.columns.name = '-'.join(grp_names[0:-1])
    b_name = 'data_RTerrors_vs_' + dataf_name
    fb_name = os.path.join(res['tdata_folder'], b_name)
    with open(fb_name, 'a') as f:
        f.write('# Combined R & t errors vs ' + str(grp_names[-1]) + '\n')
        f.write('# Parameters: ' + '-'.join(grp_names[0:-1]) + '\n')
        b.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
    sub_title = ''
    nr_it_parameters = len(grp_names[0:-1])
    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols
    for i, val in enumerate(grp_names[0:-1]):
        sub_title += tex_string_coding_style(val)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                sub_title += ' and '
        else:
            if i < nr_it_parameters - 2:
                sub_title += ', '
            elif i < nr_it_parameters - 1:
                sub_title += ', and '
    tex_infos = {'title': 'Combined R \\& t Errors vs ' + str(grp_names[-1]) +
                          ' for Parameter Variations of ' + sub_title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False}
    stats_all = b.stack().reset_index()
    stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
    # figure types: sharp plot, smooth, const plot, ybar, xbar
    use_limits = {'miny': None, 'maxy': None}
    if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
        use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
    if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
        use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
    reltex_name = os.path.join(rel_data_path, b_name)
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': 'Combined R \\& t errors vs ' + str(grp_names[-1]) +
                                          ' for parameter variations of ' + sub_title,
                                  'fig_type': 'smooth',
                                  'plots': list(b.columns.values),
                                  'axis_y': 'Combined R \\& t error',
                                  'plot_x': str(grp_names[-1]),
                                  'limits': use_limits,
                                  'legend': [tex_string_coding_style(a) for a in list(b.columns.values)],
                                  'legend_cols': None,
                                  'use_marks': res['use_marks']
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_RTerrors_vs_' + dataf_name_main
    texf_name = base_out_name + '.tex'
    if res['build_pdf'][1]:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex,
                              res['tex_folder'],
                              texf_name,
                              False,
                              os.path.join(res['pdf_folder'], pdf_name)))
    else:
        res = abs(compile_tex(rendered_tex, res['tex_folder'], texf_name, False))
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)


def combineRt(data):
    #Get R and t mean and standard deviation values
    stat_R = data['R_diffAll'].unstack()
    stat_t = data['t_angDiff_deg'].unstack()
    stat_R_mean = stat_R['mean']
    stat_t_mean = stat_t['mean']
    stat_R_std = stat_R['std']
    stat_t_std = stat_t['std']
    comb_stat_r = stat_R_mean.abs() + 2 * stat_R_std
    comb_stat_t = stat_t_mean.abs() + 2 * stat_t_std
    ma = comb_stat_r.select_dtypes(include=[np.number]).dropna().values.max()
    mi = comb_stat_r.select_dtypes(include=[np.number]).dropna().values.min()
    r_r = ma - mi
    ma = comb_stat_t.select_dtypes(include=[np.number]).dropna().values.max()
    mi = comb_stat_t.select_dtypes(include=[np.number]).dropna().values.min()
    r_t = ma - mi
    comb_stat_r = comb_stat_r / r_r
    comb_stat_t = comb_stat_t / r_t
    b = comb_stat_r + comb_stat_t
    return b


#def get_best_comb_and_th_1(data, res_folder, build_pdf=(False, True, )):
def get_best_comb_and_th_1(**keywords):
    if len(keywords) < 2 or len(keywords) > 4:
        raise ValueError('Wrong number of arguments for function get_best_comb_and_th_1')
    if 'data' not in keywords:
        raise ValueError('Missing data argument of function get_best_comb_and_th_1')
    data = keywords['data']
    if 'res_folder' not in keywords:
        raise ValueError('Missing res_folder argument of function get_best_comb_and_th_1')
    res_folder = keywords['res_folder']
    use_marks = False
    if 'use_marks' not in keywords:
        print('No information provided if marks should be used: Disabling marks')
    else:
        use_marks = keywords['use_marks']
    build_pdf = (False, True,)
    if 'build_pdf' in keywords:
        build_pdf = keywords['build_pdf']
    if len(build_pdf) != 2:
        raise ValueError('Wrong number of arguments for build_pdf')
    if build_pdf[0] or build_pdf[1]:
        pdf_folder = os.path.join(res_folder, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            # print('Folder', pdf_folder, 'for storing pdf files already exists')
            pass
    tex_folder = os.path.join(res_folder, 'tex')
    try:
        os.mkdir(tex_folder)
    except FileExistsError:
        # print('Folder', tex_folder, 'for storing tex files already exists')
        pass
    tdata_folder = os.path.join(tex_folder, 'data')
    try:
        os.mkdir(tdata_folder)
    except FileExistsError:
        # print('Folder', tdata_folder, 'for storing data files already exists')
        pass
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    grp_names = data.index.names
    dataf_name_main = str(grp_names[-1]) + '_for_options_' + '-'.join(grp_names[0:-1])
    dataf_name = dataf_name_main + '.csv'
    b = combineRt(data)
    b = b.T
    b.columns = ['-'.join(map(str, a)) for a in b.columns]
    b.columns.name = '-'.join(grp_names[0:-1])
    b_name = 'data_RTerrors_vs_' + dataf_name
    fb_name = os.path.join(tdata_folder, b_name)
    with open(fb_name, 'a') as f:
        f.write('# Combined R & t errors vs ' + str(grp_names[-1]) + '\n')
        f.write('# Parameters: ' + '-'.join(grp_names[0:-1]) + '\n')
        b.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
    sub_title = ''
    nr_it_parameters = len(grp_names[0:-1])
    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols
    for i, val in enumerate(grp_names[0:-1]):
        sub_title += tex_string_coding_style(val)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                sub_title += ' and '
        else:
            if i < nr_it_parameters - 2:
                sub_title += ', '
            elif i < nr_it_parameters - 1:
                sub_title += ', and '
    tex_infos = {'title': 'Combined R \\& t Errors vs ' + str(grp_names[-1]) +
                          ' for Parameter Variations of ' + sub_title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False}
    stats_all = b.stack().reset_index()
    stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
    # figure types: sharp plot, smooth, const plot, ybar, xbar
    use_limits = {'miny': None, 'maxy': None}
    if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
        use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
    if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
        use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
    reltex_name = os.path.join(rel_data_path, b_name)
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': 'Combined R \\& t errors vs ' + str(grp_names[-1]) +
                                          ' for parameter variations of ' + sub_title,
                                  'fig_type': 'smooth',
                                  'plots': list(b.columns.values),
                                  'axis_y': 'Combined R \\& t error',
                                  'plot_x': str(grp_names[-1]),
                                  'limits': use_limits,
                                  'legend': [tex_string_coding_style(a) for a in list(b.columns.values)],
                                  'legend_cols': None,
                                  'use_marks': use_marks
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_RTerrors_vs_' + dataf_name_main
    texf_name = base_out_name + '.tex'
    if build_pdf[1]:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex, tex_folder, texf_name, False, os.path.join(pdf_folder, pdf_name)))
    else:
        res = abs(compile_tex(rendered_tex, tex_folder, texf_name, False))
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    #Output best and worst b values for every combination
    if len(b.columns) > 10:
        fig_type = 'xbar'
    else:
        fig_type = 'ybar'
    b_best_idx = b.idxmin(axis=0)
    # Insert a tex line break for long options
    b_cols_tex = []
    for el in b.columns:
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
    b_best_l = [[val, b.loc[val].iloc[i], b.columns[i], b_cols_tex[i]] for i, val in enumerate(b_best_idx)]
    b_best = pd.DataFrame.from_records(data=b_best_l, columns=[grp_names[-1], 'b_best', 'options', 'options_tex'])
    #b_best.set_index('options', inplace=True)
    b_best_name = 'data_best_RTerrors_and_' + dataf_name
    fb_best_name = os.path.join(tdata_folder, b_best_name)
    with open(fb_best_name, 'a') as f:
        f.write('# Best combined R & t errors and their ' + str(grp_names[-1]) + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(grp_names[0:-1]) + '\n')
        b_best.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    b_worst_idx = b.idxmax(axis=0)
    b_worst_l = [[val, b.loc[val].iloc[i], b.columns[i], b_cols_tex[i]] for i, val in enumerate(b_worst_idx)]
    b_worst = pd.DataFrame.from_records(data=b_worst_l, columns=[grp_names[-1], 'b_worst', 'options', 'options_tex'])
    b_worst_name = 'data_worst_RTerrors_and_' + dataf_name
    fb_worst_name = os.path.join(tdata_folder, b_worst_name)
    with open(fb_worst_name, 'a') as f:
        f.write('# Best combined R & t errors and their ' + str(grp_names[-1]) + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(grp_names[0:-1]) + '\n')
        b_worst.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    #Get data for tex file generation
    tex_infos = {'title': 'Best and Worst Combined R \\& t Errors and Their ' + str(grp_names[-1]) +
                          ' for Parameter Variations of ' + sub_title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False
                 }
    tex_infos['sections'].append({'file': os.path.join(rel_data_path, b_best_name),
                                  'name': 'Smallest combined R \\& t errors and their ' + str(grp_names[-1]),
                                  'fig_type': fig_type,
                                  'plots': ['b_best'],
                                  'label_y': 'error',#Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'Options',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'options_tex',
                                  #Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': [str(grp_names[-1])],
                                  'limits': None,
                                  #If None, no legend is used, otherwise use a list
                                  'legend': None,
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': True,
                                  'caption': 'Smallest combined R \\& t errors (error bars) and their ' +
                                             str(grp_names[-1]) + ' which appears on top of each bar.'
                                  })
    tex_infos['sections'].append({'file': os.path.join(rel_data_path, b_worst_name),
                                  'name': 'Worst combined R \\& t errors and their ' + str(grp_names[-1]),
                                  'fig_type': fig_type,
                                  'plots': ['b_worst'],
                                  'label_y': 'error',  # Label of the value axis. For xbar it labels the x-axis
                                  # Label/column name of axis with bars. For xbar it labels the y-axis
                                  'label_x': 'Options',
                                  # Column name of axis with bars. For xbar it is the column for the y-axis
                                  'print_x': 'options_tex',
                                  # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                  'print_meta': True,
                                  'plot_meta': [str(grp_names[-1])],
                                  'limits': None,
                                  # If None, no legend is used, otherwise use a list
                                  'legend': None,
                                  'legend_cols': 1,
                                  'use_marks': False,
                                  # The x/y-axis values are given as strings if True
                                  'use_string_labels': True,
                                  'caption': 'Biggest combined R \\& t errors (error bars) and their ' +
                                             str(grp_names[-1]) + ' which appears on top of each bar.'
                                  })
    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_best-worst_RT-errors_options_' + '-'.join(map(str,grp_names[0:-1]))
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if build_pdf[0]:
        res1 = compile_tex(rendered_tex, tex_folder, texf_name, True, os.path.join(pdf_folder, pdf_name))
    else:
        res1 = compile_tex(rendered_tex, tex_folder, texf_name, True)
    if res1 != 0:
        res += abs(res1)
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)

    b_best2_idx = b_best['b_best'].idxmax(axis=0)
    alg_comb_best = str(b_best['options'].loc[b_best2_idx])
    th_best = float(b_best[grp_names[-1]].loc[b_best2_idx])
    th_best_mean = round(float(b_best[grp_names[-1]].mean()), 3)
    b_best_val = float(b_best['b_best'].loc[b_best2_idx])

    main_parameter_name = 'USAC_opt_refine_ops'
    par_file_main = 'resulting_best_parameters'
    par_file = par_file_main + '.yaml'
    res_folder_parent = os.path.abspath(os.path.join(res_folder, os.pardir))  # Get parent directory
    ppar_file = os.path.join(res_folder_parent, par_file)
    #Check if file and parameters exist
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

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = alg_comb_best.split('-')
        if len(grp_names[0:-1]) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(grp_names[0:-1]):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithms': alg_w,
                                         'th_best': th_best,
                                         'th_best_mean': th_best_mean,
                                         'b_best_val': b_best_val}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    return res


def get_best_comb_inlrat_1(**keywords):
    if len(keywords) < 2 or len(keywords) > 4:
        raise ValueError('Wrong number of arguments for function get_best_comb_and_th_1')
    if 'data' not in keywords:
        raise ValueError('Missing data argument of function get_best_comb_and_th_1')
    data = keywords['data']
    if 'res_folder' not in keywords:
        raise ValueError('Missing res_folder argument of function get_best_comb_and_th_1')
    res_folder = keywords['res_folder']
    use_marks = False
    if 'use_marks' not in keywords:
        print('No information provided if marks should be used: Disabling marks')
    else:
        use_marks = keywords['use_marks']
    build_pdf = (False, True,)
    if 'build_pdf' in keywords:
        build_pdf = keywords['build_pdf']
    if len(build_pdf) != 2:
        raise ValueError('Wrong number of arguments for build_pdf')
    if build_pdf[0] or build_pdf[1]:
        pdf_folder = os.path.join(res_folder, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            # print('Folder', pdf_folder, 'for storing pdf files already exists')
            pass
    tex_folder = os.path.join(res_folder, 'tex')
    try:
        os.mkdir(tex_folder)
    except FileExistsError:
        # print('Folder', tex_folder, 'for storing tex files already exists')
        pass
    tdata_folder = os.path.join(tex_folder, 'data')
    try:
        os.mkdir(tdata_folder)
    except FileExistsError:
        # print('Folder', tdata_folder, 'for storing data files already exists')
        pass
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    grp_names = data.index.names
    dataf_name_main = str(grp_names[-1]) + '_for_options_' + '-'.join(grp_names[0:-1])
    dataf_name = dataf_name_main + '.csv'
    b = combineRt(data)
    b = b.T
    b.columns = ['-'.join(map(str, a)) for a in b.columns]
    if len(b.columns) > 10:
        fig_type = 'xbar'
    else:
        fig_type = 'ybar'
    b.columns.name = '-'.join(grp_names[0:-1])
    b_name = 'data_RTerrors_vs_' + dataf_name
    fb_name = os.path.join(tdata_folder, b_name)
    with open(fb_name, 'a') as f:
        f.write('# Combined R & t errors vs ' + str(grp_names[-1]) + '\n')
        f.write('# Parameters: ' + '-'.join(grp_names[0:-1]) + '\n')
        b.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
    sub_title = ''
    nr_it_parameters = len(grp_names[0:-1])
    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols
    for i, val in enumerate(grp_names[0:-1]):
        sub_title += tex_string_coding_style(val)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                sub_title += ' and '
        else:
            if i < nr_it_parameters - 2:
                sub_title += ', '
            elif i < nr_it_parameters - 1:
                sub_title += ', and '
    tex_infos = {'title': 'Combined R \\& t Errors vs ' + str(grp_names[-1]) +
                          ' for Parameter Variations of ' + sub_title,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': False,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False}
    stats_all = b.stack().reset_index()
    stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
    # figure types: sharp plot, smooth, const plot, ybar, xbar
    use_limits = {'miny': None, 'maxy': None}
    if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
        use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
    if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
        use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
    reltex_name = os.path.join(rel_data_path, b_name)
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': 'Combined R \\& t errors vs ' + str(grp_names[-1]) +
                                          ' for parameter variations of ' + sub_title,
                                  'fig_type': 'smooth',
                                  'plots': list(b.columns.values),
                                  'axis_y': 'Combined R \\& t error',
                                  'plot_x': str(grp_names[-1]),
                                  'limits': use_limits,
                                  'legend': [tex_string_coding_style(a) for a in list(b.columns.values)],
                                  'legend_cols': None,
                                  'use_marks': use_marks
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    template = ji_env.get_template('usac-testing_2D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   sections=tex_infos['sections'])
    base_out_name = 'tex_RTerrors_vs_' + dataf_name_main
    texf_name = base_out_name + '.tex'
    if build_pdf[1]:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex, tex_folder, texf_name, False, os.path.join(pdf_folder, pdf_name)))
    else:
        res = abs(compile_tex(rendered_tex, tex_folder, texf_name, False))
    if res != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)