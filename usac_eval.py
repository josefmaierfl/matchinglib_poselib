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
    if len(keywords) < 3:
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
    nr_partitions = len(ret['partitions'])
    ret['it_parameters'] = keywords['it_parameters']#ret['grp_names'][nr_partitions:-1]
    ret['x_axis_column'] = keywords['x_axis_column']
    nr_it_parameters = len(ret['it_parameters'])
    from statistics_and_plot import tex_string_coding_style, \
        compile_tex, \
        calcNrLegendCols, \
        replaceCSVLabels, \
        add_to_glossary, \
        split_large_titles, \
        get_limits_log_exp, \
        enl_space_title, \
        short_concat_str, \
        check_if_series
    ret['sub_title_it_pars'] = ''
    for i, val in enumerate(ret['it_parameters']):
        ret['sub_title_it_pars'] += replaceCSVLabels(val, True, True, True)
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
        ret['sub_title_partitions'] += replaceCSVLabels(val, True, True, True)
        if (nr_partitions <= 2):
            if i < nr_partitions - 1:
                ret['sub_title_partitions'] += ' and '
        else:
            if i < nr_partitions - 2:
                ret['sub_title_partitions'] += ', '
            elif i < nr_partitions - 1:
                ret['sub_title_partitions'] += ', and '

    ret['dataf_name_main'] = str(ret['grp_names'][-1]) + '_for_options_' + \
                             short_concat_str(ret['it_parameters']) + \
                             '_and_properties_'
    ret['dataf_name_partition'] = '-'.join([a[:min(3, len(a))] for a in map(str, ret['partitions'])])
    if 'error_function' in keywords:
        ret['b'] = keywords['error_function'](data)
    else:
        ret['b'] = combineRt(data)
    ret['b_all_partitions'] = ret['b'].reset_index().set_index(ret['partitions'])
    if 'error_type_text' in keywords:
        title_text = keywords['error_type_text'] + ' vs ' + \
                     replaceCSVLabels(str(ret['grp_names'][-1]), True, True, True) +\
                     ' for Parameter Variations of ' + ret['sub_title_it_pars'] + ' separately for ' +\
                     ret['sub_title_partitions']
    else:
        title_text = 'Combined R \\& t Errors vs ' + \
                     replaceCSVLabels(str(ret['grp_names'][-1]), True, True, True) +\
                      ' for Parameter Variations of ' + ret['sub_title_it_pars'] + ' separately for ' +\
                      ret['sub_title_partitions']
    tex_infos = {'title': title_text,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': True,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations of a list of dicts
                 'abbreviations': None
                 }
    ret['b_single_partitions'] = []
    idx_old = None
    gloss_calced = False
    for p in ret['b_all_partitions'].index:
        if idx_old is not None and idx_old == p:
            continue
        idx_old = p
        tmp2 = ret['b_all_partitions'].loc[p]
        if check_if_series(tmp2):
            continue
        if not isinstance(tmp2.index[0], str) and len(tmp2.index[0]) > 1:
            idx_vals = tmp2.index[0]
        else:
            idx_vals = [tmp2.index[0]]
        part_name = '_'.join([str(ni) + '-' + str(vi) for ni, vi in zip(tmp2.index.names, idx_vals)])
        part_name_l = [replaceCSVLabels(str(ni), False, False, True) + ' = ' +
                       tex_string_coding_style(str(vi)) for ni, vi in zip(tmp2.index.names, idx_vals)]
        index_entries = [a for a in idx_vals]
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
        if not gloss_calced:
            from statistics_and_plot import glossary_from_list, add_to_glossary_eval
            if len(ret['it_parameters']) > 1:
                ret['gloss'] = glossary_from_list([str(b) for a in tmp2.columns for b in a])
            else:
                ret['gloss'] = glossary_from_list([str(a) for a in tmp2.columns])
            gloss_calced = True
            if 'R_diffAll' in data.columns and 'R_mostLikely_diffAll' not in data.columns:
                ret['gloss'] = add_to_glossary_eval('Rt_diff', ret['gloss'])
            elif 'R_diffAll' not in data.columns and 'R_mostLikely_diffAll' in data.columns:
                ret['gloss'] = add_to_glossary_eval('Rt_mostLikely_diff', ret['gloss'])
            elif 'K1_cxyfxfyNorm' in data.columns:
                ret['gloss'] = add_to_glossary_eval('K12_cxyfxfyNorm', ret['gloss'])
            else:
                raise ValueError('Combined Rt error column is missing.')
            ret['gloss'] = add_to_glossary_eval(keywords['eval_columns'] +
                                                keywords['partitions'] +
                                                keywords['x_axis_column'], ret['gloss'])
            tex_infos['abbreviations'] = ret['gloss']
        tex_infos['abbreviations'] = add_to_glossary(index_entries, tex_infos['abbreviations'])
        if len(ret['it_parameters']) > 1:
            tmp2.columns = ['-'.join(map(str, a)) for a in tmp2.columns]
            tmp2.columns.name = '-'.join(ret['it_parameters'])
        dataf_name_main_property = ret['dataf_name_main'] + part_name.replace('.', 'd')
        dataf_name = dataf_name_main_property + '.csv'
        if 'file_name_err_part' in keywords:
            b_name = 'data_' + keywords['file_name_err_part'] + '_vs_' + dataf_name
        else:
            b_name = 'data_RTerrors_vs_' + dataf_name
        b_name = 'data_RTerrors_vs_' + dataf_name
        fb_name = os.path.join(ret['tdata_folder'], b_name)
        ret['b_single_partitions'].append({'data': tmp2,
                                           'part_name': part_name,
                                           'part_name_title': part_name_title,
                                           'dataf_name_main_property': dataf_name_main_property,
                                           'dataf_name': dataf_name})
        with open(fb_name, 'a') as f:
            if 'error_type_text' in keywords:
                from statistics_and_plot import strToLower, capitalizeFirstChar
                f.write( '# ' + capitalizeFirstChar(strToLower(keywords['error_type_text'])) +
                         ' vs ' + str(ret['grp_names'][-1]) +
                         ' for properties ' + part_name + '\n')
            else:
                f.write('# Combined R & t errors vs ' + str(ret['grp_names'][-1]) + ' for properties ' + part_name + '\n')
            f.write('# Column parameters: ' + '-'.join(ret['it_parameters']) + '\n')
            tmp2.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        _, use_limits, use_log, exp_value = get_limits_log_exp(tmp2)
        is_numeric = pd.to_numeric(tmp2.reset_index()[ret['grp_names'][-1]], errors='coerce').notnull().all()
        reltex_name = os.path.join(ret['rel_data_path'], b_name)
        if 'error_type_text' in keywords:
            from statistics_and_plot import strToLower, capitalizeFirstChar
            sec_name = capitalizeFirstChar(strToLower(keywords['error_type_text'])) + ' vs ' +\
                       replaceCSVLabels(str(ret['grp_names'][-1]), True, False, True) +\
                       ' for parameter variations of \\\\' + ret['sub_title_it_pars'] +\
                       ' based on properties \\\\' + part_name.replace('_', '\\_')
            cap_name = capitalizeFirstChar(strToLower(keywords['error_type_text'])) + ' vs ' + \
                       replaceCSVLabels(str(ret['grp_names'][-1]), True, False, True) + \
                       ' for parameter variations of ' + ret['sub_title_it_pars'] + \
                       ' based on properties ' + part_name.replace('_', '\\_')
            label_y = strToLower(keywords['error_type_text'])
        else:
            sec_name = 'Combined R \\& t errors $e_{R\\vect{t}}$ vs ' + \
                       replaceCSVLabels(str(ret['grp_names'][-1]), True, False, True) + \
                       ' for parameter variations of \\\\' + ret['sub_title_it_pars'] + \
                       ' based on properties \\\\' + part_name.replace('_', '\\_')
            cap_name = 'Combined R \\& t errors $e_{R\\bm{t}}$ vs ' +\
                       replaceCSVLabels(str(ret['grp_names'][-1]), True, False, True) +\
                       ' for parameter variations of ' + ret['sub_title_it_pars'] +\
                       ' based on properties ' + part_name.replace('_', '\\_')
            label_y = 'Combined R \\& t error $e_{R\\bm{t}}$'
        nr_plots = len(list(tmp2.columns.values))
        exp_value_o = exp_value
        nr_plots_i = []
        if nr_plots <= 20:
            nr_plots_i = [nr_plots]
        else:
            pp = math.floor(nr_plots / 20)
            nr_plots_i = [20] * int(pp)
            rp = nr_plots - pp * 20
            if rp > 0:
                nr_plots_i += [nr_plots - pp * 20]
        pcnt = 0
        for i, it in enumerate(nr_plots_i):
            ps = list(tmp2.columns.values)[pcnt: pcnt + it]
            pcnt += it
            if nr_plots > 20:
                sec_name1 = sec_name + ' -- part ' + str(i + 1)
                cap_name1 = cap_name + ' -- part ' + str(i + 1)
            else:
                sec_name1 = sec_name
                cap_name1 = cap_name
            sec_name1 = split_large_titles(sec_name1)
            exp_value = enl_space_title(exp_value_o, sec_name1, tmp2, ret['grp_names'][-1],
                                        len(ps), 'smooth')
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': sec_name1,
                                          # If caption is None, the field name is used
                                          'caption': cap_name1,
                                          'fig_type': 'smooth',
                                          'plots': ps,
                                          'label_y': label_y,
                                          'plot_x': str(ret['grp_names'][-1]),
                                          'label_x': replaceCSVLabels(str(ret['grp_names'][-1])),
                                          'limits': use_limits,
                                          'legend': [tex_string_coding_style(a) for a in ps],
                                          'legend_cols': None,
                                          'use_marks': ret['use_marks'],
                                          'use_log_y_axis': use_log,
                                          'xaxis_txt_rows': 1,
                                          'enlarge_lbl_dist': None,
                                          'enlarge_title_space': exp_value,
                                          'use_string_labels': True if not is_numeric else False
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    if 'file_name_err_part' in keywords:
        base_out_name = 'tex_' + keywords['file_name_err_part'] + '_vs_' + ret['dataf_name_main'] + \
                        ret['dataf_name_partition']
    else:
        base_out_name = 'tex_RTerrors_vs_' + ret['dataf_name_main'] + ret['dataf_name_partition']
    pdfs_info = []
    max_figs_pdf = 50
    if tex_infos['ctrl_fig_size']:  # and not figs_externalize:
        max_figs_pdf = 30
    st_list = tex_infos['sections']
    if len(st_list) > max_figs_pdf:
        st_list2 = [{'figs': st_list[i:i + max_figs_pdf],
                     'pdf_nr': i1 + 1} for i1, i in enumerate(range(0, len(st_list), max_figs_pdf))]
    else:
        st_list2 = [{'figs': st_list, 'pdf_nr': 1}]
    for it in st_list2:
        if len(st_list2) == 1:
            title = tex_infos['title']
        else:
            title = tex_infos['title'] + ' -- Part ' + str(it['pdf_nr'])
        pdfs_info.append({'title': title,
                          'texf_name': base_out_name + '_' + str(it['pdf_nr']),
                          'figs_externalize': tex_infos['figs_externalize'],
                          'sections': it['figs'],
                          'make_index': tex_infos['make_index'],
                          'ctrl_fig_size': tex_infos['ctrl_fig_size'],
                          'fill_bar': tex_infos['fill_bar'],
                          'abbreviations': tex_infos['abbreviations']})

    template = ji_env.get_template('usac-testing_2D_plots.tex')
    pdf_l_info = {'rendered_tex': [], 'texf_name': [], 'pdf_name': [] if ret['build_pdf'][0] else None}
    for it in pdfs_info:
        rendered_tex = template.render(title=tex_infos['title'],
                                       make_index=tex_infos['make_index'],
                                       ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                       figs_externalize=tex_infos['figs_externalize'],
                                       fill_bar=tex_infos['fill_bar'],
                                       sections=tex_infos['sections'],
                                       abbreviations=tex_infos['abbreviations'])
        texf_name = it['texf_name'] + '.tex'
        if ret['build_pdf'][0]:
            pdf_name = it['texf_name'] + '.pdf'
            pdf_l_info['pdf_name'].append(os.path.join(ret['pdf_folder'], pdf_name))

        pdf_l_info['rendered_tex'].append(rendered_tex)
        pdf_l_info['texf_name'].append(texf_name)
    ret['res'] = abs(compile_tex(pdf_l_info['rendered_tex'], ret['tex_folder'], pdf_l_info['texf_name'],
                                 tex_infos['make_index'], pdf_l_info['pdf_name'], tex_infos['figs_externalize']))
    if ret['res'] != 0:
        warnings.warn('Error occurred during writing/compiling tex file', UserWarning)
    return ret


def pars_calc_single_fig(**keywords):
    if len(keywords) < 3:
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
    ret['dataf_name_main'] = str(ret['grp_names'][-1]) + '_for_options_' + '-'.join(keywords['it_parameters'])
    ret['dataf_name'] = ret['dataf_name_main'] + '.csv'
    ret['b'] = combineRt(data)
    ret['b'] = ret['b'].T
    from statistics_and_plot import glossary_from_list, add_to_glossary_eval, is_exp_used
    if len(keywords['it_parameters']) > 1:
        ret['gloss'] = glossary_from_list([str(b) for a in ret['b'].columns for b in a])
        ret['b'].columns = ['-'.join(map(str, a)) for a in ret['b'].columns]
        ret['b'].columns.name = '-'.join(keywords['it_parameters'])
    else:
        ret['gloss'] = glossary_from_list([str(a) for a in ret['b'].columns])
    ret['gloss'] = add_to_glossary_eval(keywords['eval_columns'] +
                                        keywords['x_axis_column'], ret['gloss'])
    if 'R_diffAll' in data.columns and 'R_mostLikely_diffAll' not in data.columns:
        ret['gloss'] = add_to_glossary_eval('Rt_diff', ret['gloss'])
    elif 'R_diffAll' not in data.columns and 'R_mostLikely_diffAll' in data.columns:
        ret['gloss'] = add_to_glossary_eval('Rt_mostLikely_diff', ret['gloss'])
    else:
        raise ValueError('Combined Rt error column is missing.')

    b_name = 'data_RTerrors_vs_' + ret['dataf_name']
    fb_name = os.path.join(ret['tdata_folder'], b_name)
    with open(fb_name, 'a') as f:
        f.write('# Combined R & t errors vs ' + str(ret['grp_names'][-1]) + '\n')
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
        enl_space_title
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
    tex_infos = {'title': 'Combined R \\& t Errors vs ' +
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
    section_name = 'Combined R \\& t errors $e_{R\\vect{t}}$ vs ' +\
                   replaceCSVLabels(str(ret['grp_names'][-1]), True, False, True) +\
                   ' for parameter variations of\\\\' + ret['sub_title']
    section_name = split_large_titles(section_name)
    exp_value = enl_space_title(exp_value, section_name, ret['b'], ret['grp_names'][-1],
                                len(list(ret['b'].columns.values)), 'smooth')
    reltex_name = os.path.join(ret['rel_data_path'], b_name)
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': section_name,
                                  # If caption is None, the field name is used
                                  'caption': 'Combined R \\& t errors $e_{R\\bm{t}}$ vs ' +
                                             replaceCSVLabels(str(ret['grp_names'][-1]), True) +
                                             ' for parameter variations of ' + ret['sub_title'],
                                  'fig_type': 'smooth',
                                  'plots': list(ret['b'].columns.values),
                                  'label_y': 'Combined R \\& t error $e_{R\\bm{t}}$',
                                  'plot_x': str(ret['grp_names'][-1]),
                                  'label_x': replaceCSVLabels(str(ret['grp_names'][-1])),
                                  'limits': use_limits,
                                  'legend': [tex_string_coding_style(a) for a in list(ret['b'].columns.values)],
                                  'legend_cols': None,
                                  'use_marks': ret['use_marks'],
                                  'use_log_y_axis': use_log,
                                  'xaxis_txt_rows': 1,
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
    if len(keywords) < 4:
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
    ret['dataf_name_main'] = ret['grp_names'][-2] + '_and_' + ret['grp_names'][-1] + \
                             '_for_options_' + '-'.join(keywords['it_parameters'])
    ret['dataf_name'] = ret['dataf_name_main'] + '.csv'
    ret['b'] = combineRt(data)
    ret['b'] = ret['b'].unstack()
    ret['b'] = ret['b'].T
    from statistics_and_plot import glossary_from_list, add_to_glossary_eval, get_3d_tex_info, get_usable_3D_cols
    if len(keywords['it_parameters']) > 1:
        ret['gloss'] = glossary_from_list([str(b) for a in ret['b'].columns for b in a])
        ret['b'].columns = ['-'.join(map(str, a)) for a in ret['b'].columns]
        ret['b'].columns.name = '-'.join(keywords['it_parameters'])
    else:
        ret['gloss'] = glossary_from_list([str(a) for a in ret['b'].columns])
    ret['gloss'] = add_to_glossary_eval(keywords['eval_columns'] +
                                        keywords['xy_axis_columns'], ret['gloss'])
    if 'R_diffAll' in data.columns and 'R_mostLikely_diffAll' not in data.columns:
        ret['gloss'] = add_to_glossary_eval('Rt_diff', ret['gloss'])
    elif 'R_diffAll' not in data.columns and 'R_mostLikely_diffAll' in data.columns:
        ret['gloss'] = add_to_glossary_eval('Rt_mostLikely_diff', ret['gloss'])
    else:
        raise ValueError('Combined Rt error column is missing.')
    ret['b'] = ret['b'].reset_index()
    # nr_equal_ss = int(ret['b'].groupby(ret['b'].columns.values[0]).size().array[0])
    env_3d_info = get_3d_tex_info(ret['b'], [ret['b'].columns.values[1], ret['b'].columns.values[0]])
    b_name = 'data_RTerrors_vs_' + ret['dataf_name']
    fb_name = os.path.join(ret['tdata_folder'], b_name)
    with open(fb_name, 'a') as f:
        f.write('# Combined R & t errors vs ' + ret['grp_names'][-2] + ' and ' + ret['grp_names'][-1] + '\n')
        f.write('# Parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
        ret['b'].to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    ret['sub_title'] = ''
    nr_it_parameters = len(keywords['it_parameters'])
    from statistics_and_plot import tex_string_coding_style, compile_tex, replaceCSVLabels
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
    plot_cols = get_usable_3D_cols(ret['b'], list(ret['b'].columns.values)[2:])
    if not plot_cols:
        ret['res'] = 1
        return ret
    if len(plot_cols) != len(list(ret['b'].columns.values)[2:]):
        drop_cols = []
        for it in list(ret['b'].columns.values)[2:]:
            if it not in plot_cols:
                drop_cols.append(it)
        if drop_cols:
            ret['b'].drop(drop_cols, axis=1, inplace=True)
    st_drops = list(dict.fromkeys(list(ret['b'].columns.values[0:2]) +
                                  [env_3d_info['colname_x'], env_3d_info['colname_y']]))
    stats_all = ret['b'].drop(st_drops, axis=1).stack().reset_index()
    stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
    use_limits = {'minz': None, 'maxz': None}
    if np.abs(stats_all['max'][0] - stats_all['min'][0]) < np.abs(stats_all['max'][0] / 200):
        if stats_all['min'][0] < 0:
            use_limits['minz'] = round(1.01 * stats_all['min'][0], 6)
        else:
            use_limits['minz'] = round(0.99 * stats_all['min'][0], 6)
        if stats_all['max'][0] < 0:
            use_limits['maxz'] = round(0.99 * stats_all['max'][0], 6)
        else:
            use_limits['maxz'] = round(1.01 * stats_all['max'][0], 6)
    tex_infos = {'title': 'Combined R \\& t Errors vs ' + replaceCSVLabels(ret['grp_names'][-2], True, True, True) +
                          ' and ' + replaceCSVLabels(ret['grp_names'][-1], True, True, True) +
                          ' for Parameter Variations of ' + ret['sub_title'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': False,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': ret['gloss']
                 }
    reltex_name = os.path.join(ret['rel_data_path'], b_name)
    tex_infos['sections'].append({'file': reltex_name,
                                  'name': 'Combined R \\& t errors $e_{R\\vect{t}}$ vs ' +
                                          replaceCSVLabels(str(ret['grp_names'][-2]), True, False, True) +
                                          ' and ' + replaceCSVLabels(str(ret['grp_names'][-1]), True, False, True) +
                                          ' for parameter variations of ' + ret['sub_title'],
                                  'fig_type': ret['fig_type'],
                                  'plots_z': list(ret['b'].columns.values)[2:],
                                  'diff_z_labels': False,
                                  'label_z': 'Combined R \\& t error $e_{R\\bm{t}}$',
                                  'plot_x': str(ret['b'].columns.values[1]),
                                  'label_x': replaceCSVLabels(str(ret['b'].columns.values[1])),
                                  'plot_y': str(ret['b'].columns.values[0]),
                                  'label_y': replaceCSVLabels(str(ret['b'].columns.values[0])),
                                  'legend': [tex_string_coding_style(a) for a in list(ret['b'].columns.values)[2:]],
                                  'use_marks': ret['use_marks'],
                                  'mesh_cols': env_3d_info['nr_equal_ss'],
                                  'use_log_z_axis': False,
                                  'limits': use_limits,
                                  'use_string_labels_x': env_3d_info['is_stringx'],
                                  'use_string_labels_y': env_3d_info['is_stringy'],
                                  'iterate_x': env_3d_info['colname_x'],
                                  'iterate_y': env_3d_info['colname_y'],
                                  'tick_dist': env_3d_info['tick_dist']
                                  })
    template = ji_env.get_template('usac-testing_3D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   use_fixed_caption=False,
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
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
    if 'R_diffAll' in data.columns and 'R_mostLikely_diffAll' not in data.columns:
        stat_R = data['R_diffAll'].unstack()
        stat_t = data['t_angDiff_deg'].unstack()
    elif 'R_mostLikely_diffAll' in data.columns and 'R_diffAll' not in data.columns:
        stat_R = data['R_mostLikely_diffAll'].unstack()
        stat_t = data['t_mostLikely_angDiff_deg'].unstack()
    else:
        raise ValueError('Unable to calculate combined Rt error as the necessary column is missing.')
    stat_R_mean = stat_R['mean']
    stat_t_mean = stat_t['mean']
    stat_R_std = stat_R['std']
    stat_t_std = stat_t['std']
    comb_stat_r = stat_R_mean.abs() + stat_R_std
    comb_stat_t = stat_t_mean.abs() + stat_t_std
    # ma = comb_stat_r.select_dtypes(include=[np.number]).dropna().values.max()
    # mi = comb_stat_r.select_dtypes(include=[np.number]).dropna().values.min()
    # r_r = ma - mi
    # ma = comb_stat_t.select_dtypes(include=[np.number]).dropna().values.max()
    # mi = comb_stat_t.select_dtypes(include=[np.number]).dropna().values.min()
    # r_t = ma - mi
    # comb_stat_r = comb_stat_r / r_r
    # comb_stat_t = comb_stat_t / r_t
    # b = (comb_stat_r + comb_stat_t) / 2

    tmp = comb_stat_r + comb_stat_t
    ma = tmp.select_dtypes(include=[np.number]).dropna().values.max()
    mi = tmp.select_dtypes(include=[np.number]).dropna().values.min()
    r_rt = ma - mi
    if np.isclose(r_rt, 0, atol=1e-06):
        b = tmp - mi
    else:
        b = (tmp - mi) / r_rt
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
    max_txt_rows_best = 1
    for idx, val in b_best['options_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows_best:
            max_txt_rows_best = txt_rows
    b_best_name = 'data_best_RTerrors_and_' + ret['dataf_name']
    fb_best_name = os.path.join(ret['tdata_folder'], b_best_name)
    with open(fb_best_name, 'a') as f:
        f.write('# Best combined R & t errors and their ' + str(ret['grp_names'][-1]) + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
        b_best.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    b_worst_idx = ret['b'].idxmax(axis=0)
    b_worst_l = [[val, ret['b'].loc[val].iloc[i], ret['b'].columns[i], b_cols_tex[i]] for i, val in enumerate(b_worst_idx)]
    b_worst = pd.DataFrame.from_records(data=b_worst_l, columns=[ret['grp_names'][-1], 'b_worst', 'options', 'options_tex'])
    max_txt_rows_worst = 1
    for idx, val in b_worst['options_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows_worst:
            max_txt_rows_worst = txt_rows
    b_worst_name = 'data_worst_RTerrors_and_' + ret['dataf_name']
    fb_worst_name = os.path.join(ret['tdata_folder'], b_worst_name)
    with open(fb_worst_name, 'a') as f:
        f.write('# Best combined R & t errors and their ' + str(ret['grp_names'][-1]) + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
        b_worst.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    #Get data for tex file generation
    from statistics_and_plot import replaceCSVLabels, add_to_glossary
    ret['gloss'] = add_to_glossary(b_best[ret['grp_names'][-1]].tolist(), ret['gloss'])
    ret['gloss'] = add_to_glossary(b_worst[ret['grp_names'][-1]].tolist(), ret['gloss'])
    tex_infos = {'title': 'Best and Worst Combined R \\& t Errors and Their ' +
                          replaceCSVLabels(str(ret['grp_names'][-1]), False, True, True) +
                          ' for Parameter Variations of ' + ret['sub_title'],
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
    section_name = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ and their ' + \
                   replaceCSVLabels(str(ret['grp_names'][-1]), False, False, True)
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
                                  'xaxis_txt_rows': max_txt_rows_best,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': False,
                                  'large_meta_space_needed': False,
                                  'caption': 'Smallest combined R \\& t errors $e_{R\\bm{t}}$ (error bars) and their ' +
                                             replaceCSVLabels(str(ret['grp_names'][-1])) +
                                             ' which appears on top of each bar.'
                                  })
    section_name = 'Worst combined R \\& t errors $e_{R\\vect{t}}$ and their ' + \
                   replaceCSVLabels(str(ret['grp_names'][-1]), False, False, True)
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
                                  'xaxis_txt_rows': max_txt_rows_worst,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': False,
                                  'large_meta_space_needed': False,
                                  'caption': 'Biggest combined R \\& t errors  $e_{R\\bm{t}}$ (error bars) and their ' +
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
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
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
    max_txt_rows = 1
    for idx, val in b_mean['options_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows:
            max_txt_rows = txt_rows
    b_mean_name = 'data_mean_RTerrors_over_all_' + ret['dataf_name']
    fb_mean_name = os.path.join(ret['tdata_folder'], b_mean_name)
    with open(fb_mean_name, 'a') as f:
        f.write('# Mean combined R & t errors over all ' + str(ret['grp_names'][-1]) + '\n')
        f.write('# Row (column options) parameters: ' + '-'.join(keywords['it_parameters']) + '\n')
        b_mean.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
    # Get data for tex file generation
    if len(ret['b'].columns) > 10:
        fig_type = 'xbar'
    else:
        fig_type = 'ybar'
    from statistics_and_plot import replaceCSVLabels
    tex_infos = {'title': 'Mean Combined R \\& t Errors over all ' +
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
    section_name = 'Mean combined R \\& t errors $e_{R\\vect{t}}$ over all ' + \
                   replaceCSVLabels(str(ret['grp_names'][-1]), True, False, True)
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
                                  'xaxis_txt_rows': max_txt_rows,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': False,
                                  'large_meta_space_needed': False,
                                  'caption': 'Mean combined R \\& t errors $e_{R\\bm{t}}$ (error bars) over all ' +
                                             replaceCSVLabels(str(ret['grp_names'][-1]), True) + '.'
                                  })
    ret['res'] = compile_2D_bar_chart('tex_mean_RT-errors_' + ret['grp_names'][-1], tex_infos, ret)

    main_parameter_name = keywords['res_par_name']#'USAC_opt_refine_ops_inlrat'
    # Check if file and parameters exist
    ppar_file, ret['res'] = check_par_file_exists(main_parameter_name, ret['res_folder'], ret['res'])

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = alg_best.split('-')
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
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
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    base_out_name = filen_pre + '_options_' + '-'.join(map(str, ret['it_parameters']))
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
                                   fill_bar=True,
                                   abbreviations=tex_infos['abbreviations'])
    base_out_name = filen_pre + '_options_' + '-'.join(map(str, ret['it_parameters']))
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
    from statistics_and_plot import replaceCSVLabels, tex_string_coding_style, add_to_glossary
    tex_infos = {'title': 'Smallest Combined R \\& t Errors and Their Corresponding ' +
                          replaceCSVLabels(str(ret['grp_names'][-2]), False, True, True) + ' for every ' +
                          replaceCSVLabels(str(ret['grp_names'][-1]), False, True, True) +
                          ' and Parameter Variations of ' + ret['sub_title'],
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': False,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': False,
                 # If true, non-numeric entries can be provided for the x-axis
                 'nonnumeric_x': False,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': ret['gloss']
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

        section_name = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ and their ' +\
                       replaceCSVLabels(str(ret['grp_names'][-2]), False, False, True) +\
                       '\\\\vs ' + replaceCSVLabels(str(ret['grp_names'][-1]), False, False, True) +\
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
                                      'xaxis_txt_rows': 1,
                                      'caption': 'Smallest combined R \\& t errors $e_{R\\bm{t}}$ (left axis) '
                                                 'and their ' +
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
    ret['gloss'] = add_to_glossary(data_min[ret['b'].columns.name].tolist(), ret['gloss'])
    ret['gloss'] = add_to_glossary(data_min[ret['grp_names'][-2]].tolist(), ret['gloss'])
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
                          replaceCSVLabels(str(ret['grp_names'][-2]), False, True, True) +
                          ' and Parameter Set of ' + ret['sub_title'] +
                          ' for every ' +
                          replaceCSVLabels(str(ret['grp_names'][-1]), False, True, True),
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
    tex_infos['sections'].append({'file': os.path.join(ret['rel_data_path'], dataf_name),
                                  'name': 'Smallest Combined R \\& t Errors $e_{R\\vect{t}}$',
                                  'title': 'Smallest Combined R \\& t Errors $e_{R\\bm{t}}$',
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
                                  'xaxis_txt_rows': 1,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': False,
                                  'large_meta_space_needed': False,
                                  'caption': 'Smallest combined R \\& t errors $e_{R\\bm{t}}$ and their ' +
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
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
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
    if len(ret['it_parameters']) > 1:
        tmp2.index = ['-'.join(map(str, a)) for a in tmp2.index]
        it_pars_ov = '-'.join(ret['it_parameters'])
        tmp2.index.name = it_pars_ov
    else:
        it_pars_ov = ret['it_parameters'][0]
    tmp2 = tmp2.reset_index().set_index(ret['partitions'][:-1])
    tmp2.index = ['-'.join(map(str, a)) for a in tmp2.index]
    partitions_ov = '-'.join(ret['partitions'][:-1])
    from statistics_and_plot import replaceCSVLabels, split_large_titles
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
                          replaceCSVLabels(str(ret['partitions'][-1]), True, True, True) + \
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

    section_name = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ and their ' + \
                   replaceCSVLabels(str(ret['partitions'][-1]), True, False, True) + \
                   '\\\\for parameters ' + ret['sub_title_it_pars'] + \
                   '\\\\and properties ' + ret['sub_title_partitions']
    if fig_type == 'xbar':
        caption = 'Smallest combined R \\& t errors $e_{R\\bm{t}}$ (bottom axis) and their ' +\
                  replaceCSVLabels(str(ret['partitions'][-1]), True) +\
                  ' (top axis) for parameters ' + ret['sub_title_it_pars'] +\
                  ' and properties ' + ret['sub_title_partitions'] + '.'
    else:
        caption = 'Smallest combined R \\& t errors $e_{R\\bm{t}}$ (left axis) and their ' + \
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
                                  'xaxis_txt_rows': 1,
                                  'caption': caption
                                  })

    for rc, lc, rl, ll in zip(right_cols, left_cols, right_legend, left_legend):
        par_str = [i for i in rl.split(' -- ') if ret['partitions'][-1] not in i][0]
        section_name = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ and their ' + \
                       replaceCSVLabels(str(ret['partitions'][-1]), True, False, True) + \
                       '\\\\for parameters ' + par_str + \
                       '\\\\and properties ' + ret['sub_title_partitions']

        caption = 'Smallest combined R \\& t errors $e_{R\\bm{t}}$ (left axis) and their ' + \
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
                                      'xaxis_txt_rows': 1,
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
    section_name = 'Smallest combined R \\& t errors $e_{R\\vect{t}}$ ' + \
                   '\\\\for parameters ' + ret['sub_title_it_pars'] + \
                   '\\\\and properties ' + ret['sub_title_partitions']
    caption = 'Smallest combined R \\& t errors $e_{R\\bm{t}}$ and their ' + \
              replaceCSVLabels(str(ret['partitions'][-1]), True) + \
              ' (on top of each bar) for parameters ' + ret['sub_title_it_pars'] + \
              ' and properties ' + ret['sub_title_partitions'] + '.'
    section_name = split_large_titles(section_name)
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
                                  'xaxis_txt_rows': 1,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': False,
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


def filter_nr_kps(**vars):
    return vars['data'].loc[vars['data']['nrTP'] == '100to1000']

def filter_nr_kps_stat(**vars):
    return vars['data'].loc[vars['data']['nrTP'] == '500']


def calc_Time_Model(**vars):
    # it_parameters: algorithms
    # xy_axis_columns: nrCorrs_GT, (inlRat_GT)
    # eval_columns: robEstimationAndRef_us
    # data_separators: inlRatMin, th
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
    if accum_all:
        needed_cols = vars['eval_columns'] + vars['it_parameters'] + x_axis_column
    else:
        needed_cols = vars['eval_columns'] + vars['it_parameters'] + x_axis_column + vars['data_separators']
    df = vars['data'][needed_cols]
    # Calculate TP
    # df['actNrTP'] = (df[x_axis_column[0]] * df[x_axis_column[1]]).round()
    if accum_all:
        grpd_cols = vars['it_parameters']
    else:
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
    eval_cols_log_scaling = [False]#, True, True]
    eval_cols_log_scaling.append(True if np.abs(np.log10(np.abs(data_new['fixed_time'].min())) -
                                                np.log10(np.abs(data_new['fixed_time'].max()))) > 1 else False)
    eval_cols_log_scaling.append(True if np.abs(np.log10(np.abs(data_new['linear_time'].min())) -
                                                np.log10(np.abs(data_new['linear_time'].max()))) > 1 else False)
    units = [('score', ''), ('fixed_time', '/$\\mu s$'), ('linear_time', '/$\\mu s$')]
    if model_type[0]['type'] == 1:
        eval_columns += ['squared_time']
        eval_cols_lname += ['quadratic time coefficient $t_{n^{2}}$']
        #eval_cols_log_scaling += [True]
        eval_cols_log_scaling.append(True if np.abs(np.log10(np.abs(data_new['squared_time'].min())) -
                                                    np.log10(np.abs(data_new['squared_time'].max()))) > 1 else False)
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
        elif accum_all:
            ret['x_axis_column'] = None
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
    from statistics_and_plot import tex_string_coding_style, \
        compile_tex, \
        calcNrLegendCols, \
        replaceCSVLabels, \
        strToLower, \
        split_large_titles, \
        get_limits_log_exp, \
        use_log_axis, \
        enl_space_title
    tmp, col_name = get_time_fixed_kp(**vars)
    tmp1 = tmp.loc[tmp.groupby(vars['t_data_separators'])[col_name].idxmin(axis=0)]
    tmp1.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new = ['-'.join(a) for a in tmp1.index]
        meta_data = [str(b) for a in tmp1.index for b in a]
        tmp1.index = index_new
        index_name = '-'.join(vars['it_parameters'])
        tmp1.index.name = index_name
    else:
        index_new = [a for a in tmp1.index]
        meta_data = [str(a) for a in tmp1.index]
        index_name = vars['it_parameters'][0]
    tmp1['pars_tex'] = insert_opt_lbreak(index_new)
    min_val = tmp1[col_name].min()
    max_val = tmp1[col_name].max()
    from statistics_and_plot import is_exp_used, check_legend_enlarge
    use_log1 = use_log_axis(min_val, max_val)
    exp_value1 = is_exp_used(min_val, max_val, use_log1)

    vars = prepare_io(**vars)

    tmp.set_index(vars['it_parameters'], inplace=True)
    tmp = tmp.T
    from statistics_and_plot import glossary_from_list, add_to_glossary, add_to_glossary_eval
    if len(vars['it_parameters']) > 1:
        gloss = glossary_from_list([str(b) for a in tmp.columns for b in a])
        par_cols = ['-'.join(map(str, a)) for a in tmp.columns]
        tmp.columns = par_cols
        it_pars_cols_name = '-'.join(map(str, vars['it_parameters']))
        tmp.columns.name = it_pars_cols_name
    else:
        gloss = glossary_from_list([str(a) for a in tmp.columns])
        it_pars_cols_name = vars['it_parameters'][0]
    gloss = add_to_glossary(meta_data, gloss)
    gloss = add_to_glossary_eval(vars['t_data_separators'] +
                                 vars['xy_axis_columns'], gloss)
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
            replaceCSVLabels(str(vars['xy_axis_columns'][1]), True, True, True) + \
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
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': gloss
                 }
    _, use_limits, use_log, exp_value = get_limits_log_exp(tmp)
    is_numeric = pd.to_numeric(tmp.reset_index()[vars['xy_axis_columns'][0]], errors='coerce').notnull().all()
    reltex_name = os.path.join(vars['rel_data_path'], t_mean_name)
    fig_name = 'Mean execution times for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
               ' over all ' + replaceCSVLabels(str(vars['xy_axis_columns'][1]), True, False, True) + \
               '\\\\extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    fig_name = split_large_titles(fig_name)
    exp_value = enl_space_title(exp_value, fig_name, tmp, vars['xy_axis_columns'][0],
                                len(list(tmp.columns.values)), 'smooth')
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
                                  'use_log_y_axis': use_log,
                                  'xaxis_txt_rows': 1,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': exp_value,
                                  'use_string_labels': True if not is_numeric else False,
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
            replaceCSVLabels(str(vars['xy_axis_columns'][1]), True, True, True) + \
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
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': gloss
                 }
    section_name = 'Minimum execution times over parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   ' for mean execution times over all ' + \
                   replaceCSVLabels(str(vars['xy_axis_columns'][1]), True, False, True) + \
                   '\\\\extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Minimum execution times over parameter variations of ' + strToLower(vars['sub_title_it_pars']) + \
              ' (corresponding parameter on top of bar) for mean execution times over all ' + \
              replaceCSVLabels(str(vars['xy_axis_columns'][1]), True) + \
              'extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints.'
    section_name = split_large_titles(section_name)
    enlarge_lbl_dist = check_legend_enlarge(tmp1, vars['xy_axis_columns'][0], 1, fig_type)
    exp_value1 = enl_space_title(exp_value1, section_name, tmp1, vars['xy_axis_columns'][0],
                                 1, fig_type)
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
                                  'use_log_y_axis': use_log1,
                                  'xaxis_txt_rows': 1,
                                  'enlarge_lbl_dist': enlarge_lbl_dist,
                                  'enlarge_title_space': exp_value1,
                                  'large_meta_space_needed': True,
                                  'caption': caption
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
    drop_cols = []
    no_seperators = False
    if 'partitions' in vars:
        if 't_data_separators' not in vars:
            raise ValueError('No data separators specified for calculating time budget')
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
        if 't_data_separators' not in vars:
            raise ValueError('No data separators specified for calculating time budget')
        for sep in vars['t_data_separators']:
            if sep not in vars['x_axis_column']:
                raise ValueError('Data separator ' + str(sep) + ' not found in x_axis_column')
    elif 'xy_axis_columns' in vars:
        if 't_data_separators' not in vars:
            raise ValueError('No data separators specified for calculating time budget')
        for sep in vars['t_data_separators']:
            if sep not in vars['xy_axis_columns']:
                raise ValueError('Data separator ' + str(sep) + ' not found in xy_axis_columns')
        drop_cols = list(set(vars['xy_axis_columns']).difference(vars['t_data_separators']))
    elif 't_data_separators' not in vars:
        no_seperators = True
    else:
        raise ValueError('Either x_axis_column or xy_axis_columns and t_data_separators or '
                         'neither of those must be provided')
    if no_seperators:
        individual_grps = vars['it_parameters']
    else:
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
    if len(keywords['build_pdf']) < 2:
        raise ValueError('Wrong number of arguments for build_pdf')
    keywords['pdf_folder'] = None
    if any(keywords['build_pdf']):
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
        keywords['sub_title_it_pars'] += replaceCSVLabels(val, True, True, True)
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
    from statistics_and_plot import glossary_from_list, add_to_glossary, add_to_glossary_eval, split_large_titles
    from statistics_and_plot import check_legend_enlarge
    tmp1min.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new1 = ['-'.join(a) for a in tmp1min.index]
        gloss = glossary_from_list([str(b) for a in tmp1min.index for b in a])
        tmp1min.index = index_new1
        index_name = '-'.join(vars['it_parameters'])
        tmp1min.index.name = index_name
    else:
        index_new1 = [a for a in tmp1min.index]
        gloss = glossary_from_list([str(a) for a in tmp1min.index])
        index_name = vars['it_parameters'][0]
    # pars_tex1 = insert_opt_lbreak(index_new1)
    pars_tex1 = [tex_string_coding_style(a) for a in list(dict.fromkeys(index_new1))]
    tmp1min = tmp1min.reset_index().set_index([vars['t_data_separators'][1], index_name]).unstack(level=-1)
    comb_cols1 = ['-'.join(a) for a in tmp1min.columns]
    tmp1min.columns = comb_cols1
    val_axis_cols1 = [a for a in comb_cols1 if col_name in a]
    meta_cols1 = [a for a in comb_cols1 if vars['t_data_separators'][0] in a]
    gloss = add_to_glossary(tmp1min[meta_cols1].stack().tolist(), gloss)
    gloss = add_to_glossary_eval(vars['t_data_separators'], gloss)

    tmp1max.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new2 = ['-'.join(a) for a in tmp1max.index]
        tmp1max.index = index_new2
        tmp1max.index.name = index_name
    else:
        index_new2 = [a for a in tmp1max.index]
    # pars_tex2 = insert_opt_lbreak(index_new2)
    pars_tex2 = [tex_string_coding_style(a) for a in list(dict.fromkeys(index_new2))]
    tmp1max = tmp1max.reset_index().set_index([vars['t_data_separators'][1], index_name]).unstack(level=-1)
    comb_cols2 = ['-'.join(a) for a in tmp1max.columns]
    tmp1max.columns = comb_cols2
    val_axis_cols2 = [a for a in comb_cols2 if col_name in a]
    meta_cols2 = [a for a in comb_cols2 if vars['t_data_separators'][0] in a]
    gloss = add_to_glossary(tmp1min[meta_cols1].stack().tolist(), gloss)

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

    enlarge_lbl_dist = check_legend_enlarge(tmp1min, vars['t_data_separators'][1], len(val_axis_cols1), 'xbar')
    title = 'Minimum and Maximum Execution Times vs ' + \
            replaceCSVLabels(str(vars['t_data_separators'][1]), True, True, True) + \
            ' for Parameter Variations of ' + vars['sub_title_it_pars'] + \
            ' Over All ' + replaceCSVLabels(str(vars['t_data_separators'][0]), True, True, True) + \
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
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': gloss
                 }
    section_name = 'Minimum execution times vs ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True, False, True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   '\\\\over all ' + replaceCSVLabels(str(vars['t_data_separators'][0]), True, False, True) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    section_name = split_large_titles(section_name)
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
                                  'xaxis_txt_rows': 1,
                                  'enlarge_lbl_dist': enlarge_lbl_dist,
                                  'enlarge_title_space': False,
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])
    section_name = 'Maximum execution times vs ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True, False, True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   '\\\\over all ' + replaceCSVLabels(str(vars['t_data_separators'][0]), True, False, True) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Maximum execution times vs ' + replaceCSVLabels(str(vars['t_data_separators'][1]), True) + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' (corresponding ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' on top of each bar) extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    section_name = split_large_titles(section_name)
    enlarge_lbl_dist = check_legend_enlarge(tmp1max, vars['t_data_separators'][1], len(val_axis_cols2), 'xbar')
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
                                  'xaxis_txt_rows': 1,
                                  'enlarge_lbl_dist': enlarge_lbl_dist,
                                  'enlarge_title_space': False,
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
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
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
    if len(vars['it_parameters']) > 1:
        index_new1 = ['-'.join(a) for a in tmp2min.index]
        tmp2min.index = index_new1
        tmp2min.index.name = index_name
    else:
        index_new1 = [a for a in tmp2min.index]
    tmp2min['pars_tex'] = insert_opt_lbreak(index_new1)
    max_txt_rows2min = 1
    for idx, val in tmp2min['pars_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows2min:
            max_txt_rows2min = txt_rows
    meta_col1 = str(vars['t_data_separators'][0]) + '-' + str(vars['t_data_separators'][1])
    tmp2min[meta_col1] = tmp2min.loc[:, vars['t_data_separators'][0]].apply(lambda x: str(x) + ' - ') + \
                         tmp2min.loc[:, vars['t_data_separators'][1]].apply(lambda x: str(x))
    gloss = add_to_glossary(tmp2min[vars['t_data_separators']].stack().tolist(), gloss)
    tmp2min.drop(vars['t_data_separators'], axis=1, inplace=True)

    tmp2max.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new2 = ['-'.join(a) for a in tmp2max.index]
        tmp2max.index = index_new2
        tmp2max.index.name = index_name
    else:
        index_new2 = [a for a in tmp2max.index]
    tmp2max['pars_tex'] = insert_opt_lbreak(index_new2)
    max_txt_rows2max = 1
    for idx, val in tmp2max['pars_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows2max:
            max_txt_rows2max = txt_rows
    meta_col2 = str(vars['t_data_separators'][0]) + '-' + str(vars['t_data_separators'][1])
    tmp2max[meta_col2] = tmp2max.loc[:, vars['t_data_separators'][0]].apply(lambda x: str(x) + ' - ') + \
                         tmp2max.loc[:, vars['t_data_separators'][1]].apply(lambda x: str(x))
    gloss = add_to_glossary(tmp2max[vars['t_data_separators']].stack().tolist(), gloss)
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
            replaceCSVLabels(str(vars['t_data_separators'][0]), True, True, True) + ' and ' + \
            replaceCSVLabels(str(vars['t_data_separators'][1]), True, True, True) + \
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
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': gloss
                 }
    section_name = 'Minimum execution times over all ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][0]), True, False, True) + ' and ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True, False, True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Minimum execution times over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' and ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1]), True) + ' (corresponding '  + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' -- ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1])) + ' values on top of each bar)' + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    section_name = split_large_titles(section_name)
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
                                  'xaxis_txt_rows': max_txt_rows2min,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': False,
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    section_name = 'Maximum execution times over all ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][0]), True, False, True) + ' and ' + \
                   replaceCSVLabels(str(vars['t_data_separators'][1]), True, False, True) + \
                   ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption = 'Maximum execution times over all ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0]), True) + ' and ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1]), True) + ' (corresponding ' + \
              replaceCSVLabels(str(vars['t_data_separators'][0])) + ' -- ' + \
              replaceCSVLabels(str(vars['t_data_separators'][1])) + ' values on top of each bar)' + \
              ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + ' extrapolated for ' + \
              str(int(vars['nr_target_kps'])) + ' keypoints'
    section_name = split_large_titles(section_name)
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
                                  'xaxis_txt_rows': max_txt_rows2max,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': False,
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
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
    # tmp1mean = tmp.groupby(first_grp).mean().drop(vars['accum_step_props'][0], axis=1)
    tmp1mean = tmp.groupby(first_grp)[col_name].mean().reset_index()
    second_grp2 = [a for a in vars['t_data_separators'] if a != vars['accum_step_props'][1]]
    second_grp = vars['it_parameters'] + second_grp2
    # tmp2mean = tmp.groupby(second_grp).mean().drop(vars['accum_step_props'][1], axis=1)
    tmp2mean = tmp.groupby(second_grp)[col_name].mean().reset_index()
    minmax_grp = vars['it_parameters'] + [vars['eval_minmax_for']]
    tmp1mean_min = tmp1mean.loc[tmp1mean.groupby(minmax_grp)[col_name].idxmin(axis=0)]
    tmp1mean_max = tmp1mean.loc[tmp1mean.groupby(minmax_grp)[col_name].idxmax(axis=0)]
    tmp2mean_min = tmp2mean.loc[tmp2mean.groupby(minmax_grp)[col_name].idxmin(axis=0)]
    tmp2mean_max = tmp2mean.loc[tmp2mean.groupby(minmax_grp)[col_name].idxmax(axis=0)]
    tmp12_min = tmp1mean_min.loc[tmp1mean_min.groupby(vars['it_parameters'])[col_name].idxmin(axis=0)]
    tmp12_max = tmp1mean_max.loc[tmp1mean_max.groupby(vars['it_parameters'])[col_name].idxmax(axis=0)]
    tmp22_min = tmp2mean_min.loc[tmp2mean_min.groupby(vars['it_parameters'])[col_name].idxmin(axis=0)]
    tmp22_max = tmp2mean_max.loc[tmp2mean_max.groupby(vars['it_parameters'])[col_name].idxmax(axis=0)]

    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols, replaceCSVLabels, strToLower
    from statistics_and_plot import add_to_glossary, \
        add_to_glossary_eval, \
        split_large_titles, \
        is_exp_used, \
        use_log_axis, \
        enl_space_title, \
        get_3d_tex_info
    tmp1mean.set_index(vars['it_parameters'], inplace=True)
    from statistics_and_plot import glossary_from_list, calc_limits, check_legend_enlarge
    if len(vars['it_parameters']) > 1:
        gloss = glossary_from_list([str(b) for a in tmp1mean.index for b in a])
        index_new1 = ['-'.join(a) for a in tmp1mean.index]
        tmp1mean.index = index_new1
        index_name = '-'.join(vars['it_parameters'])
        tmp1mean.index.name = index_name
    else:
        gloss = glossary_from_list([str(a) for a in tmp1mean.index])
        index_new1 = [a for a in tmp1mean.index]
        index_name = vars['it_parameters'][0]
    gloss = add_to_glossary_eval(vars['t_data_separators'], gloss)
    tmp1mean = tmp1mean.reset_index().set_index(first_grp2 + [index_name]).unstack(level=-1)
    index_new11 = ['-'.join(a) for a in tmp1mean.columns]
    legend1 = [tex_string_coding_style(b) for a in tmp1mean.columns for b in a if b in index_new1]
    tmp1mean.columns = index_new11
    tmp1mean.reset_index(inplace=True)
    all_vals = tmp1mean.drop(first_grp2, axis=1).stack().reset_index()
    min_val = all_vals.drop(all_vals.columns[0:-1], axis=1).min().abs()
    max_val = all_vals.drop(all_vals.columns[0:-1], axis=1).max().abs()
    use_log1 = use_log_axis(min_val[0], max_val[0])

    tmp2mean.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new2 = ['-'.join(a) for a in tmp2mean.index]
        tmp2mean.index = index_new2
        tmp2mean.index.name = index_name
    else:
        index_new2 = [a for a in tmp2mean.index]
    tmp2mean = tmp2mean.reset_index().set_index(second_grp2 + [index_name]).unstack(level=-1)
    index_new21 = ['-'.join(a) for a in tmp2mean.columns]
    legend2 = [tex_string_coding_style(b) for a in tmp2mean.columns for b in a if b in index_new2]
    tmp2mean.columns = index_new21
    tmp2mean.reset_index(inplace=True)
    all_vals = tmp2mean.drop(second_grp2, axis=1).stack().reset_index()
    min_val = all_vals.drop(all_vals.columns[0:-1], axis=1).min().abs()
    max_val = all_vals.drop(all_vals.columns[0:-1], axis=1).max().abs()
    use_log2 = use_log_axis(min_val[0], max_val[0])

    vars = prepare_io(**vars)

    t_main_name1 = 'mean_time_over_all_' + str(vars['accum_step_props'][0]) + '_vs_' + \
                  str(first_grp2[0]) + '_and_' + str(first_grp2[1]) + '_for_' + str(int(vars['nr_target_kps'])) + \
                  'kpts_for_opts_' + '-'.join(map(str, vars['it_parameters']))
    t_mean1_name = 'data_' + t_main_name1 + '.csv'
    ft_mean1_name = os.path.join(vars['tdata_folder'], t_mean1_name)
    with open(ft_mean1_name, 'a') as f:
        f.write('# Mean execution times over all ' + str(vars['accum_step_props'][0]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp1mean.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

    t_main_name2 = 'mean_time_over_all_' + str(vars['accum_step_props'][1]) + '_vs_' + \
                   str(second_grp2[0]) + '_and_' + str(second_grp2[1]) + '_for_' + str(int(vars['nr_target_kps'])) + \
                   'kpts_for_opts_' + '-'.join(map(str, vars['it_parameters']))
    t_mean2_name = 'data_' + t_main_name2 + '.csv'
    ft_mean2_name = os.path.join(vars['tdata_folder'], t_mean2_name)
    with open(ft_mean2_name, 'a') as f:
        f.write('# Mean execution times over all ' + str(vars['accum_step_props'][1]) + ' extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp2mean.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

    title = 'Mean Execution Times vs ' + \
            replaceCSVLabels(str([a for a in vars['t_data_separators'] if a not in vars['accum_step_props']][0]),
                             True, True, True) + \
            ' for Parameter Variations of ' + vars['sub_title_it_pars'] + \
            ' Seperately Over All ' + replaceCSVLabels(str(vars['accum_step_props'][0]), True, True, True) + \
            ' and ' + replaceCSVLabels(str(vars['accum_step_props'][1]), True, True, True) + \
            ' Extrapolated for ' + str(int(vars['nr_target_kps'])) + ' Keypoints'
    tex_infos = {'title': title,
                 'sections': [],
                 'use_fixed_caption': True,
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': True,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': True,
                 'figs_externalize': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': gloss}

    section_name = 'Mean execution times over all ' + \
                   replaceCSVLabels(str(vars['accum_step_props'][0]), True, False, True) + \
                   ' vs ' + replaceCSVLabels(str(first_grp2[0]), True, False, True) + ' and ' + \
                   replaceCSVLabels(str(first_grp2[1]), True, False, True) + \
                   ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    # nr_equal_ss1 = int(tmp1mean.groupby(first_grp2[0]).size().array[0])
    env_3d_info1 = get_3d_tex_info(tmp1mean, first_grp2)
    stats_all = tmp1mean[index_new11].stack().reset_index()
    stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
    use_limits = {'minz': None, 'maxz': None}
    if np.abs(stats_all['max'][0] - stats_all['min'][0]) < np.abs(stats_all['max'][0] / 200):
        if stats_all['min'][0] < 0:
            use_limits['minz'] = round(1.01 * stats_all['min'][0], 6)
        else:
            use_limits['minz'] = round(0.99 * stats_all['min'][0], 6)
        if stats_all['max'][0] < 0:
            use_limits['maxz'] = round(0.99 * stats_all['max'][0], 6)
        else:
            use_limits['maxz'] = round(1.01 * stats_all['max'][0], 6)
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_mean1_name),
                                  'name': section_name,
                                  'fig_type': vars['fig_type'][0],
                                  'plots_z': index_new11,
                                  'diff_z_labels': False,
                                  'label_z': 'Mean time/$\\mu s$',
                                  'plot_x': str(first_grp2[0]),
                                  'label_x': replaceCSVLabels(str(first_grp2[0])),
                                  'plot_y': str(first_grp2[1]),
                                  'label_y': replaceCSVLabels(str(first_grp2[1])),
                                  'legend': legend1,
                                  'use_marks': vars['use_marks'][0],
                                  'mesh_cols': env_3d_info1['nr_equal_ss'],
                                  'use_log_z_axis': use_log1,
                                  'limits': use_limits,
                                  'use_string_labels_x': env_3d_info1['is_stringx'],
                                  'use_string_labels_y': env_3d_info1['is_stringy'],
                                  'iterate_x': env_3d_info1['colname_x'],
                                  'iterate_y': env_3d_info1['colname_y'],
                                  'tick_dist': env_3d_info1['tick_dist']
                                  })

    section_name = 'Mean execution times over all ' + \
                   replaceCSVLabels(str(vars['accum_step_props'][1]), True, False, True) + \
                   ' vs ' + replaceCSVLabels(str(second_grp2[0]), True, False, True) + ' and ' + \
                   replaceCSVLabels(str(second_grp2[1]), True, False, True) + \
                   ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + \
                   ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    nr_equal_ss2 = int(tmp2mean.groupby(second_grp2[0]).size().array[0])
    env_3d_info2 = get_3d_tex_info(tmp2mean, second_grp2)
    stats_all = tmp2mean[index_new21].stack().reset_index()
    stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
    use_limits1 = {'minz': None, 'maxz': None}
    if np.abs(stats_all['max'][0] - stats_all['min'][0]) < np.abs(stats_all['max'][0] / 200):
        if stats_all['min'][0] < 0:
            use_limits1['minz'] = round(1.01 * stats_all['min'][0], 6)
        else:
            use_limits1['minz'] = round(0.99 * stats_all['min'][0], 6)
        if stats_all['max'][0] < 0:
            use_limits1['maxz'] = round(0.99 * stats_all['max'][0], 6)
        else:
            use_limits1['maxz'] = round(1.01 * stats_all['max'][0], 6)
    tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], t_mean2_name),
                                  'name': section_name,
                                  'fig_type': vars['fig_type'][0],
                                  'plots_z': index_new21,
                                  'diff_z_labels': False,
                                  'label_z': 'Mean time/$\\mu s$',
                                  'plot_x': str(second_grp2[0]),
                                  'label_x': replaceCSVLabels(str(second_grp2[0])),
                                  'plot_y': str(second_grp2[1]),
                                  'label_y': replaceCSVLabels(str(second_grp2[1])),
                                  'legend': legend2,
                                  'use_marks': vars['use_marks'][0],
                                  'mesh_cols': env_3d_info2['nr_equal_ss'],
                                  'use_log_z_axis': use_log2,
                                  'limits': use_limits1,
                                  'use_string_labels_x': env_3d_info2['is_stringx'],
                                  'use_string_labels_y': env_3d_info2['is_stringy'],
                                  'iterate_x': env_3d_info2['colname_x'],
                                  'iterate_y': env_3d_info2['colname_y'],
                                  'tick_dist': env_3d_info2['tick_dist']
                                  })

    template = ji_env.get_template('usac-testing_3D_plots.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   use_fixed_caption=tex_infos['use_fixed_caption'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    t_main_name = 'mean_time_sep_over_all_' + str(vars['accum_step_props'][0]) + '_and_' + \
                  str(vars['accum_step_props'][1]) + '_vs_' + \
                  str([a for a in vars['t_data_separators'] if a not in vars['accum_step_props']][0]) + \
                  '_for_' + str(int(vars['nr_target_kps'])) + \
                  'kpts_for_opts_' + '-'.join(map(str, vars['it_parameters']))
    base_out_name = 'tex_' + t_main_name
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if vars['build_pdf'][0]:
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

    index_y4 = []
    legend_y4 = []
    meta_col4 = []
    use_log4 = []
    exp_value4 = []
    enlarge_lbl_dist4 = []
    tmp1mean_min.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new12 = ['-'.join(a) for a in tmp1mean_min.index]
        tmp1mean_min.index = index_new12
        tmp1mean_min.index.name = index_name
    else:
        index_new12 = [a for a in tmp1mean_min.index]
    tmp1mean_min = tmp1mean_min.reset_index().set_index([vars['eval_minmax_for']] + [index_name]).unstack(level=-1)
    index_y4.append(['-'.join(a) for a in tmp1mean_min.columns if col_name in a])
    legend_y4.append([tex_string_coding_style(b) for a in tmp1mean_min.columns if col_name in a
                      for b in a if b in index_new12])
    tmp1mean_min.columns = ['-'.join(a) for a in tmp1mean_min.columns]
    meta_col4.append([a for a in tmp1mean_min.columns if col_name not in a])
    tmp1mean_min.reset_index(inplace=True)
    gloss = add_to_glossary(tmp1mean_min[meta_col4[-1]].stack().tolist(), gloss)
    time_on1 = [a for a in first_grp2 if a != vars['eval_minmax_for']][0]
    all_vals = tmp1mean_min.drop(meta_col4[-1] + [vars['eval_minmax_for']], axis=1).stack().reset_index()
    min_val = all_vals.drop(all_vals.columns[0:-1], axis=1).min().abs()
    max_val = all_vals.drop(all_vals.columns[0:-1], axis=1).max().abs()
    use_log4.append(use_log_axis(min_val[0], max_val[0]))
    exp_value4.append(is_exp_used(min_val[0], max_val[0], use_log4[-1]))
    enlarge_lbl_dist4.append(check_legend_enlarge(tmp1mean_min, vars['eval_minmax_for'],
                                                  len(index_y4[-1]), vars['fig_type'][1]))

    tmp1mean_max.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new13 = ['-'.join(a) for a in tmp1mean_max.index]
        tmp1mean_max.index = index_new13
        tmp1mean_max.index.name = index_name
    else:
        index_new13 = [a for a in tmp1mean_max.index]
    tmp1mean_max = tmp1mean_max.reset_index().set_index([vars['eval_minmax_for']] + [index_name]).unstack(level=-1)
    index_y4.append(['-'.join(a) for a in tmp1mean_max.columns if col_name in a])
    legend_y4.append([tex_string_coding_style(b) for a in tmp1mean_max.columns if col_name in a
                      for b in a if b in index_new13])
    tmp1mean_max.columns = ['-'.join(a) for a in tmp1mean_max.columns]
    meta_col4.append([a for a in tmp1mean_max.columns if col_name not in a])
    tmp1mean_max.reset_index(inplace=True)
    gloss = add_to_glossary(tmp1mean_max[meta_col4[-1]].stack().tolist(), gloss)
    all_vals = tmp1mean_max.drop(meta_col4[-1] + [vars['eval_minmax_for']], axis=1).stack().reset_index()
    min_val = all_vals.drop(all_vals.columns[0:-1], axis=1).min().abs()
    max_val = all_vals.drop(all_vals.columns[0:-1], axis=1).max().abs()
    use_log4.append(use_log_axis(min_val[0], max_val[0]))
    exp_value4.append(is_exp_used(min_val[0], max_val[0], use_log4[-1]))
    enlarge_lbl_dist4.append(check_legend_enlarge(tmp1mean_max, vars['eval_minmax_for'],
                                                  len(index_y4[-1]), vars['fig_type'][1]))

    tmp2mean_min.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new22 = ['-'.join(a) for a in tmp2mean_min.index]
        tmp2mean_min.index = index_new22
        tmp2mean_min.index.name = index_name
    else:
        index_new22 = [a for a in tmp2mean_min.index]
    tmp2mean_min = tmp2mean_min.reset_index().set_index([vars['eval_minmax_for']] + [index_name]).unstack(level=-1)
    index_y4.append(['-'.join(a) for a in tmp2mean_min.columns if col_name in a])
    legend_y4.append([tex_string_coding_style(b) for a in tmp2mean_min.columns if col_name in a
                      for b in a if b in index_new22])
    tmp2mean_min.columns = ['-'.join(a) for a in tmp2mean_min.columns]
    meta_col4.append([a for a in tmp2mean_min.columns if col_name not in a])
    tmp2mean_min.reset_index(inplace=True)
    gloss = add_to_glossary(tmp2mean_min[meta_col4[-1]].stack().tolist(), gloss)
    time_on2 = [a for a in second_grp2 if a != vars['eval_minmax_for']][0]
    all_vals = tmp2mean_min.drop(meta_col4[-1] + [vars['eval_minmax_for']], axis=1).stack().reset_index()
    min_val = all_vals.drop(all_vals.columns[0:-1], axis=1).min().abs()
    max_val = all_vals.drop(all_vals.columns[0:-1], axis=1).max().abs()
    use_log4.append(use_log_axis(min_val[0], max_val[0]))
    exp_value4.append(is_exp_used(min_val[0], max_val[0], use_log4[-1]))
    enlarge_lbl_dist4.append(check_legend_enlarge(tmp2mean_min, vars['eval_minmax_for'],
                                                  len(index_y4[-1]), vars['fig_type'][1]))

    tmp2mean_max.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new23 = ['-'.join(a) for a in tmp2mean_max.index]
        tmp2mean_max.index = index_new23
        tmp2mean_max.index.name = index_name
    else:
        index_new23 = [a for a in tmp2mean_max.index]
    tmp2mean_max = tmp2mean_max.reset_index().set_index([vars['eval_minmax_for']] + [index_name]).unstack(level=-1)
    index_y4.append(['-'.join(a) for a in tmp2mean_max.columns if col_name in a])
    legend_y4.append([tex_string_coding_style(b) for a in tmp2mean_max.columns if col_name in a
                      for b in a if b in index_new23])
    tmp2mean_max.columns = ['-'.join(a) for a in tmp2mean_max.columns]
    meta_col4.append([a for a in tmp2mean_max.columns if col_name not in a])
    tmp2mean_max.reset_index(inplace=True)
    gloss = add_to_glossary(tmp2mean_max[meta_col4[-1]].stack().tolist(), gloss)
    all_vals = tmp2mean_max.drop(meta_col4[-1] + [vars['eval_minmax_for']], axis=1).stack().reset_index()
    min_val = all_vals.drop(all_vals.columns[0:-1], axis=1).min().abs()
    max_val = all_vals.drop(all_vals.columns[0:-1], axis=1).max().abs()
    use_log4.append(use_log_axis(min_val[0], max_val[0]))
    exp_value4.append(is_exp_used(min_val[0], max_val[0], use_log4[-1]))
    enlarge_lbl_dist4.append(check_legend_enlarge(tmp2mean_max, vars['eval_minmax_for'],
                                                  len(index_y4[-1]), vars['fig_type'][1]))

    t_main_name1 = 'time_on_' + time_on1 +\
                   '_over_accumul_'+ str(vars['accum_step_props'][0]) + '_vs_' + \
                   str(vars['eval_minmax_for']) + '_for_' + str(int(vars['nr_target_kps'])) + \
                   'kpts_for_opts_' + '-'.join(map(str, vars['it_parameters']))
    fnames4 = []
    fnames4.append('data_min_' + t_main_name1 + '.csv')
    ft_min1_name = os.path.join(vars['tdata_folder'], fnames4[-1])
    with open(ft_min1_name, 'a') as f:
        f.write('# Minimum execution times over all ' + time_on1 + ' for accumulated execution times over ' +
                str(vars['accum_step_props'][0]) + ' values extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp1mean_min.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

    fnames4.append('data_max_' + t_main_name1 + '.csv')
    ft_max1_name = os.path.join(vars['tdata_folder'], fnames4[-1])
    with open(ft_max1_name, 'a') as f:
        f.write('# Maximum execution times over all ' + time_on1 + ' for accumulated execution times over ' +
                str(vars['accum_step_props'][0]) + ' values extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp1mean_max.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

    t_main_name2 = 'time_on_' + time_on2 + \
                   '_over_accumul_' + str(vars['accum_step_props'][1]) + '_vs_' + \
                   str(vars['eval_minmax_for']) + '_for_' + str(int(vars['nr_target_kps'])) + \
                   'kpts_for_opts_' + '-'.join(map(str, vars['it_parameters']))
    fnames4.append('data_min_' + t_main_name2 + '.csv')
    ft_min2_name = os.path.join(vars['tdata_folder'], fnames4[-1])
    with open(ft_min2_name, 'a') as f:
        f.write('# Minimum execution times over all ' + time_on2 + ' for accumulated execution times over ' +
                str(vars['accum_step_props'][1]) + ' values extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp2mean_min.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

    fnames4.append('data_max_' + t_main_name2 + '.csv')
    ft_max2_name = os.path.join(vars['tdata_folder'], fnames4[-1])
    with open(ft_max2_name, 'a') as f:
        f.write('# Maximum execution times over all ' + time_on2 + ' for accumulated execution times over ' +
                str(vars['accum_step_props'][1]) + ' values extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp2mean_max.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

    title = 'Minimum and Maximum Execution Times vs ' + \
            replaceCSVLabels(str(vars['eval_minmax_for']), True, True, True) + \
            ' for Parameter Variations of ' + vars['sub_title_it_pars'] + \
            ' Separately Over All ' + replaceCSVLabels(str(time_on1), True, True, True) + \
            ' and ' + replaceCSVLabels(str(time_on2), True, True, True) + \
            ' Extrapolated for ' + str(int(vars['nr_target_kps'])) + ' Keypoints'
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
    section_name = []
    caption = []
    section_name_main1 = 'execution times vs ' + \
                         replaceCSVLabels(str(vars['eval_minmax_for']), True, False, True) + \
                         ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                         '\\\\over all ' + replaceCSVLabels(str(time_on1), True, False, True) + \
                         ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    section_name.append(split_large_titles('Minimum ' + section_name_main1))
    section_name.append(split_large_titles('Maximum ' + section_name_main1))
    caption_main1 = 'execution times vs ' + \
            replaceCSVLabels(str(vars['eval_minmax_for']), True) + \
            ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + \
            ' over all ' + replaceCSVLabels(str(time_on1), True) + \
            ' (corresponding ' + replaceCSVLabels(str(time_on1)) + \
            ' on top of each bar) extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption.append('Minimum ' + caption_main1)
    caption.append('Maximum ' + caption_main1)

    section_name_main2 = 'execution times vs ' + \
                         replaceCSVLabels(str(vars['eval_minmax_for']), True, False, True) + \
                         ' for parameter variations of\\\\' + strToLower(vars['sub_title_it_pars']) + \
                         '\\\\over all ' + replaceCSVLabels(str(time_on2), True, False, True) + \
                         ' extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    section_name.append(split_large_titles('Minimum ' + section_name_main2))
    section_name.append(split_large_titles('Maximum ' + section_name_main2))
    caption_main2 = 'execution times vs ' + \
                    replaceCSVLabels(str(vars['eval_minmax_for']), True) + \
                    ' for parameter variations of ' + strToLower(vars['sub_title_it_pars']) + \
                    ' over all ' + replaceCSVLabels(str(time_on2), True) + \
                    ' (corresponding ' + replaceCSVLabels(str(time_on2)) + \
                    ' on top of each bar) extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    caption.append('Minimum ' + caption_main2)
    caption.append('Maximum ' + caption_main2)

    exp_value4[0] = enl_space_title(exp_value4[0], section_name[0], tmp1mean_min, vars['eval_minmax_for'],
                                    len(index_y4[0]), vars['fig_type'][1])
    exp_value4[1] = enl_space_title(exp_value4[1], section_name[1], tmp1mean_max, vars['eval_minmax_for'],
                                    len(index_y4[1]), vars['fig_type'][1])
    exp_value4[2] = enl_space_title(exp_value4[2], section_name[2], tmp2mean_min, vars['eval_minmax_for'],
                                    len(index_y4[2]), vars['fig_type'][1])
    exp_value4[3] = enl_space_title(exp_value4[3], section_name[3], tmp2mean_max, vars['eval_minmax_for'],
                                    len(index_y4[3]), vars['fig_type'][1])

    for i in range(0, 4):
        tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], fnames4[i]),
                                      'name': section_name[i].replace('\\\\', ' '),
                                      'title': section_name[i],
                                      'title_rows': section_name[i].count('\\\\'),
                                      'fig_type': vars['fig_type'][1],
                                      'plots': index_y4[i],
                                      # Label of the value axis. For xbar it labels the x-axis
                                      'label_y': 'time/$\\mu s$',
                                      # Label/column name of axis with bars. For xbar it labels the y-axis
                                      'label_x': replaceCSVLabels(str(vars['eval_minmax_for'])),
                                      # Column name of axis with bars. For xbar it is the column for the y-axis
                                      'print_x': str(vars['eval_minmax_for']),
                                      # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                      'print_meta': True,
                                      'plot_meta': meta_col4[i],
                                      # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                      'rotate_meta': 0,
                                      'limits': None,
                                      # If None, no legend is used, otherwise use a list
                                      'legend': legend_y4[i],
                                      'legend_cols': None,
                                      'use_marks': vars['use_marks'][1],
                                      # The x/y-axis values are given as strings if True
                                      'use_string_labels': False,
                                      'use_log_y_axis': use_log4[i],
                                      'xaxis_txt_rows': 1,
                                      'enlarge_lbl_dist': enlarge_lbl_dist4[i],
                                      'enlarge_title_space': exp_value4[i],
                                      'large_meta_space_needed': False,
                                      'caption': caption[i]
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
    t_main_name = 'time_on_' + time_on1 + '_and_' + time_on2 + \
                  '_over_sep_accumul_' + str(vars['accum_step_props'][0]) + '_and_' + \
                  str(vars['accum_step_props'][1]) + '_vs_' + str(vars['eval_minmax_for']) + '_for_' + \
                  str(int(vars['nr_target_kps'])) + 'kpts_for_opts_' + '-'.join(map(str, vars['it_parameters']))
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

    meta_col4 = []
    use_log4 = []
    exp_value4 = []
    tmp12_min.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new13 = ['-'.join(a) for a in tmp12_min.index]
        tmp12_min.index = index_new13
        tmp12_min.index.name = index_name
    else:
        index_new13 = [a for a in tmp12_min.index]
    tmp12_min['pars_tex'] = insert_opt_lbreak(index_new13)
    max_txt_rows = [1]
    for idx, val in tmp12_min['pars_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows[-1]:
            max_txt_rows[-1] = txt_rows
    meta_col4.append('-'.join(first_grp2))
    tmp12_min[meta_col4[-1]] = tmp12_min.loc[:, first_grp2[0]].apply(lambda x: str(x) + ' - ') + \
                               tmp12_min.loc[:, first_grp2[1]].apply(lambda x: str(x))
    gloss = add_to_glossary(tmp12_min[first_grp2].stack().tolist(), gloss)
    tmp12_min.drop(first_grp2, axis=1, inplace=True)
    meta_col4.append(meta_col4[-1])
    min_val = np.abs(tmp12_min[col_name].min())
    max_val = np.abs(tmp12_min[col_name].max())
    use_log4.append(use_log_axis(min_val, max_val))
    exp_value4.append(is_exp_used(min_val, max_val, use_log4[-1]))

    tmp12_max.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new14 = ['-'.join(a) for a in tmp12_max.index]
        tmp12_max.index = index_new14
        tmp12_max.index.name = index_name
    else:
        index_new14 = [a for a in tmp12_max.index]
    tmp12_max['pars_tex'] = insert_opt_lbreak(index_new14)
    max_txt_rows.append(1)
    for idx, val in tmp12_max['pars_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows[-1]:
            max_txt_rows[-1] = txt_rows
    tmp12_max[meta_col4[-1]] = tmp12_max.loc[:, first_grp2[0]].apply(lambda x: str(x) + ' - ') + \
                               tmp12_max.loc[:, first_grp2[1]].apply(lambda x: str(x))
    gloss = add_to_glossary(tmp12_max[first_grp2].stack().tolist(), gloss)
    tmp12_max.drop(first_grp2, axis=1, inplace=True)
    min_val = np.abs(tmp12_max[col_name].min())
    max_val = np.abs(tmp12_max[col_name].max())
    use_log4.append(use_log_axis(min_val, max_val))
    exp_value4.append(is_exp_used(min_val, max_val, use_log4[-1]))

    tmp22_min.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new23 = ['-'.join(a) for a in tmp22_min.index]
        tmp22_min.index = index_new23
        tmp22_min.index.name = index_name
    else:
        index_new23 = [a for a in tmp22_min.index]
    tmp22_min['pars_tex'] = insert_opt_lbreak(index_new23)
    max_txt_rows.append(1)
    for idx, val in tmp22_min['pars_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows[-1]:
            max_txt_rows[-1] = txt_rows
    meta_col4.append('-'.join(second_grp2))
    tmp22_min[meta_col4[-1]] = tmp22_min.loc[:, second_grp2[0]].apply(lambda x: str(x) + ' - ') + \
                               tmp22_min.loc[:, second_grp2[1]].apply(lambda x: str(x))
    gloss = add_to_glossary(tmp22_min[second_grp2].stack().tolist(), gloss)
    tmp22_min.drop(second_grp2, axis=1, inplace=True)
    meta_col4.append(meta_col4[-1])
    min_val = np.abs(tmp22_min[col_name].min())
    max_val = np.abs(tmp22_min[col_name].max())
    use_log4.append(use_log_axis(min_val, max_val))
    exp_value4.append(is_exp_used(min_val, max_val, use_log4[-1]))

    tmp22_max.set_index(vars['it_parameters'], inplace=True)
    if len(vars['it_parameters']) > 1:
        index_new24 = ['-'.join(a) for a in tmp22_max.index]
        tmp22_max.index = index_new24
        tmp22_max.index.name = index_name
    else:
        index_new24 = [a for a in tmp22_max.index]
    tmp22_max['pars_tex'] = insert_opt_lbreak(index_new24)
    max_txt_rows.append(1)
    for idx, val in tmp22_max['pars_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows[-1]:
            max_txt_rows[-1] = txt_rows
    tmp22_max[meta_col4[-1]] = tmp22_max.loc[:, second_grp2[0]].apply(lambda x: str(x) + ' - ') + \
                               tmp22_max.loc[:, second_grp2[1]].apply(lambda x: str(x))
    gloss = add_to_glossary(tmp22_max[second_grp2].stack().tolist(), gloss)
    tmp22_max.drop(second_grp2, axis=1, inplace=True)
    min_val = np.abs(tmp22_max[col_name].min())
    max_val = np.abs(tmp22_max[col_name].max())
    use_log4.append(use_log_axis(min_val, max_val))
    exp_value4.append(is_exp_used(min_val, max_val, use_log4[-1]))

    t_main_name1 = 'time_on_' + meta_col4[0] + \
                   '_over_accumul_' + str(vars['accum_step_props'][0]) + \
                   '_for_' + str(int(vars['nr_target_kps'])) + \
                   'kpts_for_opts_' + '-'.join(map(str, vars['it_parameters']))
    fnames4 = []
    fnames4.append('data_min_' + t_main_name1 + '.csv')
    ft_min2_name = os.path.join(vars['tdata_folder'], fnames4[-1])
    with open(ft_min2_name, 'a') as f:
        f.write('# Minimum execution times over all ' + meta_col4[0] +
                ' combinations for accumulated execution times over ' +
                str(vars['accum_step_props'][0]) + ' values extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp12_min.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    fnames4.append('data_max_' + t_main_name1 + '.csv')
    ft_max2_name = os.path.join(vars['tdata_folder'], fnames4[-1])
    with open(ft_max2_name, 'a') as f:
        f.write('# Maximum execution times over all ' + meta_col4[0] +
                ' combinations for accumulated execution times over ' +
                str(vars['accum_step_props'][0]) + ' values extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp12_max.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    t_main_name2 = 'time_on_' + meta_col4[2] + \
                   '_over_accumul_' + str(vars['accum_step_props'][1]) + \
                   '_for_' + str(int(vars['nr_target_kps'])) + \
                   'kpts_for_opts_' + '-'.join(map(str, vars['it_parameters']))
    fnames4.append('data_min_' + t_main_name2 + '.csv')
    ft_min2_name = os.path.join(vars['tdata_folder'], fnames4[-1])
    with open(ft_min2_name, 'a') as f:
        f.write('# Minimum execution times over all ' + meta_col4[2] +
                ' combinations for accumulated execution times over ' +
                str(vars['accum_step_props'][1]) + ' values extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp22_min.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    fnames4.append('data_max_' + t_main_name2 + '.csv')
    ft_max2_name = os.path.join(vars['tdata_folder'], fnames4[-1])
    with open(ft_max2_name, 'a') as f:
        f.write('# Maximum execution times over all ' + meta_col4[2] +
                ' combinations for accumulated execution times over ' +
                str(vars['accum_step_props'][1]) + ' values extrapolated for ' +
                str(int(vars['nr_target_kps'])) + ' keypoints' + '\n')
        f.write('# Parameters: ' + '-'.join(vars['it_parameters']) + '\n')
        tmp22_max.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    title = 'Minimum and Maximum Execution Times for Parameter Variations of ' + \
            vars['sub_title_it_pars'] + \
            ' Over All ' + replaceCSVLabels(first_grp2[0], False, True, True) + ' and ' + \
            replaceCSVLabels(first_grp2[1], False, True, True) + ' Combinations in Addition to All ' + \
            replaceCSVLabels(second_grp2[0], False, True, True) + ' and ' + \
            replaceCSVLabels(second_grp2[1], False, True, True) + \
            ' Combinations Extrapolated for ' + str(int(vars['nr_target_kps'])) + ' Keypoints'
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
    section_name = []
    caption = []
    section_name_main1 = 'execution times for parameter variations of\\\\' + \
                         strToLower(vars['sub_title_it_pars']) + \
                         '\\\\over all ' + replaceCSVLabels(first_grp2[0], False, False, True) + ' and ' + \
                         replaceCSVLabels(first_grp2[1], False, False, True) + \
                         ' combinations\\\\extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    section_name.append(split_large_titles('Minimum ' + section_name_main1))
    section_name.append(split_large_titles('Maximum ' + section_name_main1))
    caption_main1 = 'execution times for parameter variations of ' + \
                    strToLower(vars['sub_title_it_pars']) + \
                    ' over all ' + replaceCSVLabels(first_grp2[0]) + ' and ' + \
                    replaceCSVLabels(first_grp2[1]) + \
                    ' combinations (corresponding ' + replaceCSVLabels(first_grp2[0]) + ' -- ' + \
                    replaceCSVLabels(first_grp2[1]) + ' values on top of each bar) extrapolated for ' + \
                    str(int(vars['nr_target_kps'])) + ' keypoints'
    caption.append('Minimum ' + caption_main1)
    caption.append('Maximum ' + caption_main1)

    section_name_main2 = 'execution times for parameter variations of\\\\' + \
                         strToLower(vars['sub_title_it_pars']) + \
                         '\\\\over all ' + replaceCSVLabels(second_grp2[0], False, False, True) + ' and ' + \
                         replaceCSVLabels(second_grp2[1], False, False, True) + \
                         ' combinations\\\\extrapolated for ' + str(int(vars['nr_target_kps'])) + ' keypoints'
    section_name.append(split_large_titles('Minimum ' + section_name_main2))
    section_name.append(split_large_titles('Maximum ' + section_name_main2))
    caption_main2 = 'execution times for parameter variations of ' + \
                    strToLower(vars['sub_title_it_pars']) + \
                    ' over all ' + replaceCSVLabels(second_grp2[0]) + ' and ' + \
                    replaceCSVLabels(second_grp2[1]) + \
                    ' combinations (corresponding ' + replaceCSVLabels(second_grp2[0]) + ' -- ' + \
                    replaceCSVLabels(second_grp2[1]) + ' values on top of each bar) extrapolated for ' + \
                    str(int(vars['nr_target_kps'])) + ' keypoints'
    caption.append('Minimum ' + caption_main2)
    caption.append('Maximum ' + caption_main2)

    exp_value4[0] = enl_space_title(exp_value4[0], section_name[0], tmp12_min, 'pars_tex',
                                    1, 'xbar')
    exp_value4[1] = enl_space_title(exp_value4[1], section_name[1], tmp12_max, 'pars_tex',
                                    1, 'xbar')
    exp_value4[2] = enl_space_title(exp_value4[2], section_name[2], tmp22_min, 'pars_tex',
                                    1, 'xbar')
    exp_value4[3] = enl_space_title(exp_value4[3], section_name[3], tmp22_max, 'pars_tex',
                                    1, 'xbar')

    for i in range(0, 4):
        tex_infos['sections'].append({'file': os.path.join(vars['rel_data_path'], fnames4[i]),
                                      'name': section_name[i].replace('\\\\', ' '),
                                      'title': section_name[i],
                                      'title_rows': section_name[i].count('\\\\'),
                                      'fig_type': 'xbar',
                                      'plots': [col_name],
                                      # Label of the value axis. For xbar it labels the x-axis
                                      'label_y': 'time/$\\mu s$',
                                      # Label/column name of axis with bars. For xbar it labels the y-axis
                                      'label_x': 'Parameter combination',
                                      # Column name of axis with bars. For xbar it is the column for the y-axis
                                      'print_x': 'pars_tex',
                                      # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                      'print_meta': True,
                                      'plot_meta': [meta_col4[i]],
                                      # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                      'rotate_meta': 0,
                                      'limits': None,
                                      # If None, no legend is used, otherwise use a list
                                      'legend': None,
                                      'legend_cols': 1,
                                      'use_marks': False,
                                      # The x/y-axis values are given as strings if True
                                      'use_string_labels': True,
                                      'use_log_y_axis': use_log4[i],
                                      'xaxis_txt_rows': max_txt_rows[i],
                                      'enlarge_lbl_dist': None,
                                      'enlarge_title_space': exp_value4[i],
                                      'large_meta_space_needed': False,
                                      'caption': caption[i]
                                      })

    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
    t_main_name = 'time_on_' + meta_col4[0] + '_and_' + meta_col4[2] + \
                  '_combs_over_sep_accumul_' + str(vars['accum_step_props'][0]) + '_and_' + \
                  str(vars['accum_step_props'][1]) + '_for_' + \
                  str(int(vars['nr_target_kps'])) + 'kpts_for_opts_' + '-'.join(map(str, vars['it_parameters']))
    base_out_name = 'tex_min_max_' + t_main_name
    texf_name = base_out_name + '.tex'
    pdf_name = base_out_name + '.pdf'
    if vars['build_pdf'][2]:
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

    min1_t = tmp12_min.loc[[tmp12_min[col_name].idxmin()]].reset_index()
    min2_t = tmp22_min.loc[[tmp22_min[col_name].idxmin()]].reset_index()

    main_parameter_name = vars['res_par_name']
    # Check if file and parameters exist
    ppar_file, res = check_par_file_exists(main_parameter_name, vars['res_folder'], res)

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl1 = str(min1_t[index_name].values[0]).split('-')
        if len(vars['it_parameters']) != len(alg_comb_bestl1):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w1 = {}
        for i, val in enumerate(vars['it_parameters']):
            alg_w1[val] = alg_comb_bestl1[i]
        par1 = str(min1_t[meta_col4[0]].values[0]).split(' - ')

        par_name2 = meta_col4[2].split('-')
        alg_comb_bestl2 = str(min2_t[index_name].values[0]).split('-')
        if len(vars['it_parameters']) != len(alg_comb_bestl2):
            raise ValueError('Nr of refine algorithms does not match')
        alg_w2 = {}
        for i, val in enumerate(vars['it_parameters']):
            alg_w2[val] = alg_comb_bestl2[i]
        par2 = str(min2_t[meta_col4[2]].values[0]).split(' - ')
        par_name1 = meta_col4[2].split('-')
        yaml.dump({main_parameter_name: {'res1':{'Algorithm': alg_w1,
                                                par_name1[0]: par1[0],
                                                par_name1[1]: par1[1],
                                                'Time_us': float(min1_t[col_name].values[0])},
                                         'res2': {'Algorithm': alg_w2,
                                                  par_name2[0]: par2[0],
                                                  par_name2[1]: par2[1],
                                                  'Time_us': float(min2_t[col_name].values[0])}
                                         }},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    return res


def get_inlrat_diff(**vars):
    if len(vars['eval_columns']) != 2:
        raise ValueError('For calculating the difference of inlier ratios, eval_columns must hold 2 entries')
    if 'inlRat_estimated' not in vars['eval_columns'] or 'inlRat_GT' not in vars['eval_columns']:
        raise ValueError('For calculating inlier ratio differences, eval_columns must hold labels '
                         'inlRat_estimated and inlRat_GT')
    if 'partitions' in vars:
        if 'x_axis_column' in vars:
            needed_columns = vars['eval_columns'] + vars['it_parameters'] + \
                             vars['x_axis_column'] + vars['partitions']
        elif 'xy_axis_columns' in vars:
            needed_columns = vars['eval_columns'] + vars['it_parameters'] + \
                             vars['xy_axis_columns'] + vars['partitions']
        else:
            needed_columns = vars['eval_columns'] + vars['it_parameters'] + \
                             vars['partitions']
    elif 'x_axis_column' in vars:
        needed_columns = vars['eval_columns'] + vars['it_parameters'] + vars['x_axis_column']
    elif 'xy_axis_columns' in vars:
        needed_columns = vars['eval_columns'] + vars['it_parameters'] + vars['xy_axis_columns']
    else:
        needed_columns = vars['eval_columns'] + vars['it_parameters']
    data = vars['data'].loc[:, needed_columns]
    eval_columns = ['inlRat_diff']
    data['inlRat_diff'] = data[vars['eval_columns'][0]] - data[vars['eval_columns'][1]]
    data.drop(vars['eval_columns'], axis=1, inplace=True)
    ret = {'data': data,
           'eval_columns': eval_columns,
           'it_parameters': vars['it_parameters']}
    if 'partitions' in vars:
        ret['partitions'] = vars['partitions']
    if 'x_axis_column' in vars:
        ret['x_axis_column'] = vars['x_axis_column']
    elif 'xy_axis_columns' in vars:
        ret['xy_axis_columns'] = vars['xy_axis_columns']
    return ret


def get_min_inlrat_diff(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    if len(keywords) < 4 or len(keywords) > 7:
        raise ValueError('Wrong number of arguments for function pars_calc_single_fig_partitions')
    if 'data' not in keywords:
        raise ValueError('Missing data argument of function pars_calc_single_fig_partitions')
    if 'partitions' not in keywords:
        raise ValueError('Missing partitions argument of function pars_calc_single_fig_partitions')
    data = keywords['data']
    partitions = keywords['partitions']
    keywords = prepare_io(**keywords)
    grp_names = data.index.names
    nr_partitions = len(partitions)
    it_parameters = grp_names[nr_partitions:-1]
    from statistics_and_plot import tex_string_coding_style, compile_tex, calcNrLegendCols, replaceCSVLabels, strToLower
    from statistics_and_plot import glossary_from_list, add_to_glossary, add_to_glossary_eval, split_large_titles
    from statistics_and_plot import get_limits_log_exp, enl_space_title
    dataf_name_main = str(grp_names[-1]) + '_for_options_' + '-'.join(it_parameters)
    hlp = [a for a in data.columns.values if 'mean' in a]
    if len(hlp) != 1 or len(hlp[0]) != 2:
        raise ValueError('Wrong DataFrame format for inlier ratio difference statistics')
    diff_mean = data[hlp[0]]
    diff_mean.name = hlp[0][0]
    diff_mean = diff_mean.abs().reset_index().drop(partitions, axis=1).groupby(it_parameters +
                                                                               [grp_names[-1]]).mean().unstack()
    if len(keywords['it_parameters']) > 1:
        gloss = glossary_from_list([str(b) for a in diff_mean.index for b in a])
        diff_mean.index = ['-'.join(map(str, a)) for a in diff_mean.index]
        it_parameters_name = '-'.join(it_parameters)
        diff_mean.index.name = it_parameters_name
    else:
        gloss = glossary_from_list([str(a) for a in diff_mean.index])
        it_parameters_name = it_parameters[0]
    gloss = add_to_glossary_eval(grp_names[-1], gloss)
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
                          replaceCSVLabels(str(grp_names[-1]), True, True, True) +
                          ' for Parameter Variations of ' + keywords['sub_title_it_pars'],
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
    _, use_limits, use_log, exp_value = get_limits_log_exp(diff_mean)
    is_numeric = pd.to_numeric(diff_mean.reset_index()[grp_names[-1]], errors='coerce').notnull().all()
    reltex_name = os.path.join(keywords['rel_data_path'], b_name)
    fig_name = 'Absolute mean inlier ratio differences $\\Delta \\epsilon$ vs ' + \
               replaceCSVLabels(str(grp_names[-1]), True, False, True) + \
               ' for parameter variations of \\\\' + keywords['sub_title_it_pars']
    fig_name = split_large_titles(fig_name)
    exp_value = enl_space_title(exp_value, fig_name, diff_mean, grp_names[-1],
                                len(list(diff_mean.columns.values)), 'smooth')
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
                                  'use_log_y_axis': use_log,
                                  'xaxis_txt_rows': 1,
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
    min_mean_diff['options_for_tex'] = insert_opt_lbreak([a for i, a in min_mean_diff[it_parameters_name].iteritems()])
    max_txt_rows = 1
    for idx, val in min_mean_diff['options_for_tex'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows:
            max_txt_rows = txt_rows
    gloss = add_to_glossary(min_mean_diff[grp_names[-1]].tolist(), gloss)
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
                          replaceCSVLabels(str(grp_names[-1]), False, True, True) +
                          ' for Parameter Variations of ' + keywords['sub_title_it_pars'],
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
    section_name = 'Minimum absolute mean inlier ratio difference\\\\and its corresponding ' + \
                   replaceCSVLabels(str(grp_names[-1]), False, False, True) + \
                   ' for parameter variations of\\\\' + strToLower(keywords['sub_title_it_pars'])
    caption = 'Minimum absolute mean inlier ratio difference and its corresponding ' + \
              replaceCSVLabels(str(grp_names[-1])) + \
              ' (on top of each bar) for parameter variations of ' + strToLower(keywords['sub_title_it_pars'])
    section_name = split_large_titles(section_name)
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
                                  'xaxis_txt_rows': max_txt_rows,
                                  'enlarge_lbl_dist': None,
                                  'enlarge_title_space': False,
                                  'large_meta_space_needed': False,
                                  'caption': caption
                                  })
    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    rendered_tex = template.render(title=tex_infos['title'],
                                   make_index=tex_infos['make_index'],
                                   ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                   figs_externalize=tex_infos['figs_externalize'],
                                   fill_bar=tex_infos['fill_bar'],
                                   sections=tex_infos['sections'],
                                   abbreviations=tex_infos['abbreviations'])
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

    main_parameter_name = keywords['res_par_name']#'USAC_opt_search_min_inlrat_diff'
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
