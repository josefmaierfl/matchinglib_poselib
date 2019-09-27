"""
Evaluates results from the autocalibration present in a pandas DataFrame as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np, math
import ruamel.yaml as yaml
# import modin.pandas as pd
import pandas as pd
#from jinja2 import Template as ji
import jinja2 as ji
# import tempfile
# import shutil
from copy import deepcopy
import shutil
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


def compile_tex(rendered_tex,
                out_tex_dir,
                out_tex_file,
                make_fig_index=True,
                out_pdf_filen=None,
                figs_externalize=False,
                build_glossary=False):
    texdf = os.path.join(out_tex_dir, out_tex_file)
    with open(texdf, 'w') as outfile:
        outfile.write(rendered_tex)
    if out_pdf_filen is not None:
        rep_make = 1
        if make_fig_index or figs_externalize:
            rep_make = 2
        if build_glossary:
            rep_make = 3
        pdfpath, pdfname = os.path.split(out_pdf_filen)
        pdfname = os.path.splitext(pdfname)[0]
        stdoutf = os.path.join(pdfpath, 'stdout_' + pdfname + '.txt')
        erroutf = os.path.join(pdfpath, 'error_' + pdfname + '.txt')
        cmdline = ['pdflatex',
                   '--jobname=' + pdfname,
                   '--output-directory=' + pdfpath,
                   '-synctex=1',
                   '--interaction=nonstopmode']
        if figs_externalize:
            cmdline += ['--shell-escape', texdf]
            figs = os.path.join(pdfpath, 'figures')
            try:
                os.mkdir(figs)
            except FileExistsError:
                print('Folder', figs, 'for storing temp images already exists')
            except:
                print("Unexpected error (Unable to create directory for storing temp images):", sys.exc_info()[0])
            try:
                os.symlink(figs, os.path.join(out_tex_dir, 'figures'))
            except OSError:
                print('Unable to create a symlink to the stored images')
            except:
                print("Unexpected error (Unable to create directory for storing temp images):", sys.exc_info()[0])
        else:
            cmdline += [texdf]
        stdoutfh = open(stdoutf, 'w')
        erroutfh = open(erroutf, 'w')
        retcode = 0
        while rep_make > 0 and retcode == 0:
            try:
                retcode = sp.run(cmdline,
                                 shell=False,
                                 check=True,
                                 cwd=out_tex_dir,
                                 stdout=stdoutfh,
                                 stderr=erroutfh).returncode
                if retcode < 0:
                    print("Child pdflatex was terminated by signal", -retcode, file=sys.stderr)
                    retcode = 1
                else:
                    print("PDF generation successful with code", retcode)
            except OSError as e:
                print("Execution of pdflatex failed:", e, file=sys.stderr)
                retcode = 1
            except sp.CalledProcessError as e:
                print("Execution of pdflatex failed:", e, file=sys.stderr)
                retcode = 1
            rep_make -= 1
            if rep_make > 0 and retcode == 0:
                stdoutfh.close()
                erroutfh.close()
                try:
                    os.remove(stdoutf)
                except:
                    print('Unable to remove output log file')
                try:
                    os.remove(erroutf)
                except:
                    print('Unable to remove output log file')
                stdoutfh = open(stdoutf, 'w')
                erroutfh = open(erroutf, 'w')

        stdoutfh.close()
        erroutfh.close()
        auxf = os.path.join(pdfpath, pdfname + '.aux')
        if os.path.exists(auxf):
            try:
                os.remove(auxf)
            except:
                print('Unable to remove aux file')
        loff = os.path.join(pdfpath, pdfname + '.lof')
        if os.path.exists(loff):
            try:
                os.remove(loff)
            except:
                print('Unable to remove lof file')
        synctex = os.path.join(pdfpath, pdfname + '.synctex.gz')
        if os.path.exists(synctex):
            try:
                os.remove(synctex)
            except:
                print('Unable to remove synctex.gz file')
        outf = os.path.join(pdfpath, pdfname + '.out')
        if os.path.exists(outf):
            try:
                os.remove(outf)
            except:
                print('Unable to remove out file')
        if os.path.exists(auxf):
            try:
                os.remove(auxf)
            except:
                print('Unable to remove aux file')
        tocf = os.path.join(pdfpath, pdfname + '.toc')
        if os.path.exists(tocf):
            try:
                os.remove(tocf)
            except:
                print('Unable to remove toc file')
        auxlockf = os.path.join(pdfpath, pdfname + '.auxlock')
        if os.path.exists(auxlockf):
            try:
                os.remove(auxlockf)
            except:
                print('Unable to remove auxlock file')
        if figs_externalize:
            if os.path.exists(figs):
                try:
                    os.unlink(os.path.join(out_tex_dir, 'figures'))
                except IsADirectoryError:
                    print('Unable to remove symlink to images')
                except:
                    print("Unexpected error (Unable to remove directory to temp images):", sys.exc_info()[0])
                try:
                    shutil.rmtree(figs, ignore_errors=True)
                except:
                    print('Unable to remove figures directory')
        if retcode == 0:
            logf = os.path.join(pdfpath, pdfname + '.log')
            try:
                os.remove(logf)
            except:
                print('Unable to remove log file')
            try:
                os.remove(stdoutf)
            except:
                print('Unable to remove output log file')
            try:
                os.remove(erroutf)
            except:
                print('Unable to remove output log file')
        return retcode
    return 0


def calcSatisticAndPlot_2D(data,
                           store_path,
                           tex_file_pre_str,
                           fig_title_pre_str,
                           eval_description_path,
                           eval_columns,
                           units,
                           it_parameters,
                           x_axis_column,
                           pdfsplitentry,# One or more column names present in eval_columns for splitting pdf
                           filter_func = None,
                           filter_func_args = None,
                           special_calcs_func = None,
                           special_calcs_args = None,
                           calc_func = None,
                           calc_func_args = None,
                           compare_source = None,
                           fig_type='smooth',
                           use_marks=True,
                           ctrl_fig_size=True,
                           make_fig_index=True,
                           build_pdf=False,
                           figs_externalize=True):
    if len(x_axis_column) != 1:
        raise ValueError('Only 1 column is allowed to be selected for the x axis')
    fig_types = ['sharp plot', 'smooth', 'const plot', 'ybar', 'xbar']
    if not fig_type in fig_types:
        raise ValueError('Unknown figure type.')
    # if type(data) is not pd.dataframe.DataFrame:
    #     data = pd.utils.from_pandas(data)
    #Filter rows by excluding not successful estimations
    data = data.loc[~((data['R_out(0,0)'] == 0) &
                      (data['R_out(0,1)'] == 0) &
                      (data['R_out(0,2)'] == 0) &
                      (data['R_out(1,0)'] == 0) &
                      (data['R_out(1,1)'] == 0) &
                      (data['R_out(1,2)'] == 0) &
                      (data['R_out(2,0)'] == 0) &
                      (data['R_out(2,1)'] == 0) &
                      (data['R_out(2,2)'] == 0))]
    if filter_func is not None:
        if filter_func_args is None:
            filter_func_args = {'data': data}
        else:
            filter_func_args['data'] = data
        data = filter_func(**filter_func_args)
    if data.empty:
        raise ValueError('No data left after filtering')

    # Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            calc_func_args = {'data': data}
        else:
            calc_func_args['data'] = data
        calc_func_args['eval_columns'] = eval_columns
        calc_func_args['it_parameters'] = it_parameters
        calc_func_args['x_axis_column'] = x_axis_column
        ret = calc_func(**calc_func_args)
        df = ret['data']
        eval_columns = ret['eval_columns']
        it_parameters = ret['it_parameters']
        x_axis_column = ret['x_axis_column']
    else:
        needed_columns = eval_columns + it_parameters + x_axis_column
        df = data[needed_columns]

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + '-'.join(map(str, it_parameters)) + '_vs_' +
                                              '-'.join(map(str, x_axis_column)))
    cnt = 1
    store_path_init = store_path_sub
    while os.path.exists(store_path_sub):
        store_path_sub = store_path_init + '_' + str(int(cnt))
        cnt += 1
    try:
        os.mkdir(store_path_sub)
    except FileExistsError:
        print('Folder', store_path_sub, 'for storing statistics data already exists')
    except:
        print("Unexpected error (Unable to create directory for storing statistics data):", sys.exc_info()[0])
        raise
    if build_pdf:
        pdf_folder = os.path.join(store_path_sub, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            print('Folder', pdf_folder, 'for storing pdf files already exists')
    tex_folder = os.path.join(store_path_sub, 'tex')
    try:
        os.mkdir(tex_folder)
    except FileExistsError:
        print('Folder', tex_folder, 'for storing tex files already exists')
    tdata_folder = os.path.join(tex_folder, 'data')
    try:
        os.mkdir(tdata_folder)
    except FileExistsError:
        print('Folder', tdata_folder, 'for storing data files already exists')

    if compare_source:
        compare_source['full_path'] = os.path.join(compare_source['store_path'],
                                                   compare_source['eval_description_path'] + '_' +
                                                   '-'.join(map(str, compare_source['it_parameters'])) + '_vs_' +
                                                   '-'.join(map(str, x_axis_column)))
        if not os.path.exists(compare_source['full_path']):
            warnings.warn('Path ' + compare_source['full_path'] + ' for comparing results not found. '
                          'Skipping comparison.', UserWarning)
            compare_source = None
        if compare_source:
            compare_source['tdata_folder'] = os.path.join(compare_source['full_path'], 'tex')
            if compare_source and not os.path.exists(compare_source['tdata_folder']):
                warnings.warn('Tex folder ' + compare_source['tdata_folder'] + ' for comparing results not found. '
                              'Skipping comparison.', UserWarning)
                compare_source = None
        if compare_source:
            compare_source['tdata_folder'] = os.path.join(compare_source['tdata_folder'], 'data')
            if compare_source and not os.path.exists(compare_source['tdata_folder']):
                warnings.warn('Data folder ' + compare_source['tdata_folder'] + ' for comparing results not found. '
                              'Skipping comparison.', UserWarning)
                compare_source = None

    #Group by USAC parameters 5&6 and calculate the statistic
    stats = df.groupby(it_parameters + x_axis_column).describe()
    if special_calcs_func is not None and special_calcs_args is not None:
        if 'func_name' in special_calcs_args:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_args['func_name'])
        else:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_func.__name__)
        cnt = 1
        calc_vals = True
        special_path_init = special_path_sub
        while os.path.exists(special_path_sub):
            special_path_sub = special_path_init + '_' + str(int(cnt))
            cnt += 1
        try:
            os.mkdir(special_path_sub)
        except FileExistsError:
            print('Folder', special_path_sub, 'for storing statistics data already exists')
        except:
            print("Unexpected error (Unable to create directory for storing special function data):", sys.exc_info()[0])
            calc_vals = False
        if calc_vals:
            special_calcs_args['data'] = stats
            special_calcs_args['it_parameters'] = it_parameters
            special_calcs_args['eval_columns'] = eval_columns
            special_calcs_args['x_axis_column'] = x_axis_column
            special_calcs_args['res_folder'] = special_path_sub
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Errors occured during calculation of specific results!', UserWarning)
    errvalnames = stats.columns.values # Includes statistic name and error value names
    grp_names = stats.index.names #As used when generating the groups
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    for i in range(0, nr_it_parameters):
        base_out_name += grp_names[i]
        title_name += replaceCSVLabels(grp_names[i], True, True)
        if(nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_name += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_name += ', '
            elif i < nr_it_parameters - 1:
                title_name += ', and '
        if i < nr_it_parameters - 1:
            base_out_name += '-'
    base_out_name += '_combs_vs_' + grp_names[-1]
    title_name += ' Compared to ' + replaceCSVLabels(grp_names[-1], False, True) + ' Values'
    # holds the grouped names/entries within the group names excluding the last entry th
    #grp_values = list(dict.fromkeys([i[0:2] for i in stats.index.values]))
    tex_infos = {'title': title_name,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': make_fig_index,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': ctrl_fig_size,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': figs_externalize,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations of a list of dicts
                 'abbreviations': None}
    pdf_nr = 0
    gloss_calced = False
    for it in errvalnames:
        if it[-1] != 'count':
            tmp = stats[it[0]].unstack()
            tmp = tmp[it[1]]
            tmp = tmp.T
            #tmp.columns = ['%s%s' % (str(a), '-%s' % str(b) if b is not None else '') for a, b in tmp.columns]
            if not gloss_calced:
                if len(it_parameters) > 1:
                    gloss = glossary_from_list([str(b) for a in tmp.columns for b in a])
                else:
                    gloss = glossary_from_list([str(a) for a in tmp.columns])
                if compare_source:
                    add_to_glossary(compare_source['it_par_select'], gloss)
                    gloss.append({'key': 'cmp', 'description': compare_source['cmp']})
                gloss_calced = True
                if gloss:
                    gloss = add_to_glossary_eval(eval_columns + x_axis_column, gloss)
                    tex_infos['abbreviations'] = gloss
                else:
                    gloss = add_to_glossary_eval(eval_columns + x_axis_column)
                    if gloss:
                        tex_infos['abbreviations'] = gloss
            if len(it_parameters) > 1:
                tmp.columns = ['-'.join(map(str, a)) for a in tmp.columns]
                tmp.columns.name = '-'.join(grp_names[0:-1])

            dataf_name = 'data_' + '_'.join(map(str, it)) + '_vs_' + \
                         str(grp_names[-1]) + '.csv'
            dataf_name = dataf_name.replace('%', 'perc')
            if compare_source:
                tmp, succ = add_comparison_column(compare_source, dataf_name, tmp)
            fdataf_name = os.path.join(tdata_folder, dataf_name)
            with open(fdataf_name, 'a') as f:
                f.write('# ' + str(it[-1]) + ' values for ' + str(it[0]) + '\n')
                f.write('# Column parameters: ' + '-'.join(it_parameters) + '\n')
                tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

            #Construct tex-file
            if pdfsplitentry:
                if pdf_nr < len(pdfsplitentry):
                    if pdfsplitentry[pdf_nr] == str(it[0]):
                        pdf_nr += 1
            stats_all = tmp.stack().reset_index()
            stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
            if (np.isclose(stats_all['min'][0], 0, atol=1e-06) and
                np.isclose(stats_all['max'][0], 0, atol=1e-06)) or \
                    np.isclose(stats_all['min'][0], stats_all['max'][0]):
                continue
            #figure types: sharp plot, smooth, const plot, ybar, xbar
            use_limits = {'miny': None, 'maxy': None}
            if np.abs(stats_all['max'][0] - stats_all['min'][0]) < np.abs(stats_all['max'][0] / 200):
                if stats_all['min'][0] < 0:
                    use_limits['miny'] = round(1.01 * stats_all['min'][0], 6)
                else:
                    use_limits['miny'] = round(0.99 * stats_all['min'][0], 6)
                if stats_all['max'][0] < 0:
                    use_limits['maxy'] = round(0.99 * stats_all['max'][0], 6)
                else:
                    use_limits['maxy'] = round(1.01 * stats_all['max'][0], 6)
            else:
                if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
                    use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
                if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
                    use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
            if use_limits['miny'] and use_limits['maxy']:
                use_log = True if np.abs(np.log10(np.abs(use_limits['miny'])) -
                                         np.log10(np.abs(use_limits['maxy']))) > 1 else False
                exp_value = is_exp_used(use_limits['miny'], use_limits['maxy'], use_log)
            elif use_limits['miny']:
                use_log = True if np.abs(np.log10(np.abs(use_limits['miny'])) -
                                         np.log10(np.abs(stats_all['max'][0]))) > 1 else False
                exp_value = is_exp_used(use_limits['miny'], stats_all['max'][0], use_log)
            elif use_limits['maxy']:
                use_log = True if np.abs(np.log10(np.abs(stats_all['min'][0])) -
                                         np.log10(np.abs(use_limits['maxy']))) > 1 else False
                exp_value = is_exp_used(stats_all['min'][0], use_limits['maxy'], use_log)
            else:
                use_log = True if np.abs(np.log10(np.abs(stats_all['min'][0])) -
                                         np.log10(np.abs(stats_all['max'][0]))) > 1 else False
                exp_value = is_exp_used(stats_all['min'][0], stats_all['max'][0], use_log)

            is_numeric = pd.to_numeric(tmp.reset_index()[grp_names[-1]], errors='coerce').notnull().all()
            section_name = replace_stat_names(it[-1]) + ' values for ' +\
                           replaceCSVLabels(str(it[0]), True) +\
                           ' compared to ' + replaceCSVLabels(str(grp_names[-1]), True)
            if exp_value and len(section_name) < 70:
                exp_value = False
            reltex_name = os.path.join(rel_data_path, dataf_name)
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': section_name,
                                          # If caption is None, the field name is used
                                          'caption': None,
                                          'fig_type': fig_type,
                                          'plots': list(tmp.columns.values),
                                          'label_y': replace_stat_names(it[-1]) + findUnit(str(it[0]), units),
                                          'plot_x': str(grp_names[-1]),
                                          'label_x': replaceCSVLabels(str(grp_names[-1])),
                                          'limits': use_limits,
                                          'legend': [tex_string_coding_style(a) for a in list(tmp.columns.values)],
                                          'legend_cols': None,
                                          'use_marks': use_marks,
                                          'use_log_y_axis': use_log,
                                          'enlarge_title_space': exp_value,
                                          'use_string_labels': True if not is_numeric else False,
                                          'xaxis_txt_rows': 1,
                                          'pdf_nr': pdf_nr
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    template = ji_env.get_template('usac-testing_2D_plots.tex')
    #Get number of pdfs to generate
    pdf_nr = tex_infos['sections'][-1]['pdf_nr']
    res = 0
    if pdf_nr == 0:
        rendered_tex = template.render(title=tex_infos['title'],
                                       make_index=tex_infos['make_index'],
                                       ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                       figs_externalize=tex_infos['figs_externalize'],
                                       fill_bar=tex_infos['fill_bar'],
                                       sections=tex_infos['sections'],
                                       abbreviations=tex_infos['abbreviations'])
        texf_name = base_out_name + '.tex'
        if build_pdf:
            pdf_name = base_out_name + '.pdf'
            res += compile_tex(rendered_tex,
                               tex_folder,
                               texf_name,
                               make_fig_index,
                               os.path.join(pdf_folder, pdf_name),
                               tex_infos['figs_externalize'])
        else:
            res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)
    else:
        sections = []
        diff_pdfs = []
        tmp_nr = 0
        for it in tex_infos['sections']:
            if it['pdf_nr'] == tmp_nr:
                sections.append(it)
            else:
                diff_pdfs.append(deepcopy(sections))
                sections = [it]
                tmp_nr += 1
        diff_pdfs.append(sections)
        for it in diff_pdfs:
            rendered_tex = template.render(title=tex_infos['title'],
                                           make_index=tex_infos['make_index'],
                                           ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                           figs_externalize=tex_infos['figs_externalize'],
                                           fill_bar=tex_infos['fill_bar'],
                                           sections=it,
                                           abbreviations=tex_infos['abbreviations'])
            texf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.tex'
            if build_pdf:
                pdf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.pdf'
                res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index,
                                   os.path.join(pdf_folder, pdf_name), tex_infos['figs_externalize'])
            else:
                res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)
    return res


def calcSatisticAndPlot_2D_partitions(data,
                                      store_path,
                                      tex_file_pre_str,
                                      fig_title_pre_str,
                                      eval_description_path,
                                      eval_columns,#Column names for which statistics are calculated (y-axis)
                                      units,# Units in string format for every entry of eval_columns
                                      it_parameters,# Algorithm parameters to evaluate
                                      partitions,# Data properties to calculate statistics seperately
                                      x_axis_column,# x-axis column name
                                      filter_func=None,
                                      filter_func_args=None,
                                      special_calcs_func = None,
                                      special_calcs_args = None,
                                      calc_func = None,
                                      calc_func_args = None,
                                      compare_source = None,
                                      fig_type='smooth',
                                      use_marks=True,
                                      ctrl_fig_size=True,
                                      make_fig_index=True,
                                      build_pdf=False,
                                      figs_externalize=True):
    if len(x_axis_column) != 1:
        raise ValueError('Only 1 column is allowed to be selected for the x axis')
    fig_types = ['sharp plot', 'smooth', 'const plot', 'ybar', 'xbar']
    if not fig_type in fig_types:
        raise ValueError('Unknown figure type.')
    # if type(data) is not pd.dataframe.DataFrame:
    #     data = pd.utils.from_pandas(data)
    #Filter rows by excluding not successful estimations
    data = data.loc[~((data['R_out(0,0)'] == 0) &
                      (data['R_out(0,1)'] == 0) &
                      (data['R_out(0,2)'] == 0) &
                      (data['R_out(1,0)'] == 0) &
                      (data['R_out(1,1)'] == 0) &
                      (data['R_out(1,2)'] == 0) &
                      (data['R_out(2,0)'] == 0) &
                      (data['R_out(2,1)'] == 0) &
                      (data['R_out(2,2)'] == 0))]
    if filter_func is not None:
        if filter_func_args is None:
            filter_func_args = {'data': data}
        else:
            filter_func_args['data'] = data
        data = filter_func(**filter_func_args)
    if data.empty:
        raise ValueError('No data left after filtering')

    # Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            calc_func_args = {'data': data}
        else:
            calc_func_args['data'] = data
        calc_func_args['eval_columns'] = eval_columns
        calc_func_args['it_parameters'] = it_parameters
        calc_func_args['x_axis_column'] = x_axis_column
        calc_func_args['partitions'] = partitions
        ret = calc_func(**calc_func_args)
        df = ret['data']
        eval_columns = ret['eval_columns']
        it_parameters = ret['it_parameters']
        x_axis_column = ret['x_axis_column']
        partitions = ret['partitions']
    else:
        needed_columns = eval_columns + it_parameters + x_axis_column + partitions
        df = data[needed_columns]

    if len(partitions) > 1:
        store_path_sub = os.path.join(store_path, eval_description_path + '_' + '-'.join(map(str, it_parameters)) +
                                                  '_vs_' + '-'.join(map(str, x_axis_column)) + '_for_' +
                                                  '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]))
    else:
        store_path_sub = os.path.join(store_path, eval_description_path + '_' + '-'.join(map(str, it_parameters)) +
                                                  '_vs_' +'-'.join(map(str, x_axis_column)) + '_for_' +
                                                  str(partitions[0]))
    cnt = 1
    store_path_init = store_path_sub
    while os.path.exists(store_path_sub):
        store_path_sub = store_path_init + '_' + str(int(cnt))
        cnt += 1
    try:
        os.mkdir(store_path_sub)
    except FileExistsError:
        print('Folder', store_path_sub, 'for storing statistics data already exists')
    except:
        print("Unexpected error (Unable to create directory for storing statistics data):", sys.exc_info()[0])
        raise
    if build_pdf:
        pdf_folder = os.path.join(store_path_sub, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            print('Folder', pdf_folder, 'for storing pdf files already exists')
    tex_folder = os.path.join(store_path_sub, 'tex')
    try:
        os.mkdir(tex_folder)
    except FileExistsError:
        print('Folder', tex_folder, 'for storing tex files already exists')
    tdata_folder = os.path.join(tex_folder, 'data')
    try:
        os.mkdir(tdata_folder)
    except FileExistsError:
        print('Folder', tdata_folder, 'for storing data files already exists')

    if compare_source:
        if len(partitions) > 1:
            compare_source['full_path'] = os.path.join(compare_source['store_path'],
                                                       compare_source['eval_description_path'] + '_' +
                                                       '-'.join(map(str, compare_source['it_parameters'])) +
                                                       '_vs_' + '-'.join(map(str, x_axis_column)) + '_for_' +
                                                       '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]))
        else:
            compare_source['full_path'] = os.path.join(compare_source['store_path'],
                                                       compare_source['eval_description_path'] + '_' +
                                                       '-'.join(map(str, compare_source['it_parameters'])) +
                                                       '_vs_' + '-'.join(map(str, x_axis_column)) + '_for_' +
                                                       str(partitions[0]))
        if not os.path.exists(compare_source['full_path']):
            warnings.warn('Path ' + compare_source['full_path'] + ' for comparing results not found. '
                          'Skipping comparison.', UserWarning)
            compare_source = None
        if compare_source:
            compare_source['tdata_folder'] = os.path.join(compare_source['full_path'], 'tex')
            if compare_source and not os.path.exists(compare_source['tdata_folder']):
                warnings.warn('Tex folder ' + compare_source['tdata_folder'] + ' for comparing results not found. '
                              'Skipping comparison.', UserWarning)
                compare_source = None
        if compare_source:
            compare_source['tdata_folder'] = os.path.join(compare_source['tdata_folder'], 'data')
            if compare_source and not os.path.exists(compare_source['tdata_folder']):
                warnings.warn('Data folder ' + compare_source['tdata_folder'] + ' for comparing results not found. '
                              'Skipping comparison.', UserWarning)
                compare_source = None

    #Group by USAC parameters 5&6 and calculate the statistic
    stats = df.groupby(partitions + it_parameters + x_axis_column).describe()
    if special_calcs_func is not None and special_calcs_args is not None:
        if 'func_name' in special_calcs_args:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_args['func_name'])
        else:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_func.__name__)
        cnt = 1
        calc_vals = True
        special_path_init = special_path_sub
        while os.path.exists(special_path_sub):
            special_path_sub = special_path_init + '_' + str(int(cnt))
            cnt += 1
        try:
            os.mkdir(special_path_sub)
        except FileExistsError:
            print('Folder', special_path_sub, 'for storing statistics data already exists')
        except:
            print("Unexpected error (Unable to create directory for storing special function data):", sys.exc_info()[0])
            calc_vals = False
        if calc_vals:
            special_calcs_args['data'] = stats
            special_calcs_args['partitions'] = partitions
            special_calcs_args['eval_columns'] = eval_columns
            special_calcs_args['x_axis_column'] = x_axis_column
            special_calcs_args['it_parameters'] = it_parameters
            special_calcs_args['res_folder'] = special_path_sub
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Errors occured during calculation of specific results!', UserWarning)
    errvalnames = stats.columns.values # Includes statistic name and error value names
    grp_names = stats.index.names #As used when generating the groups
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    nr_partitions = len(partitions)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    for i in range(0, nr_it_parameters):
        base_out_name += grp_names[nr_partitions + i]
        title_name += replaceCSVLabels(grp_names[nr_partitions + i], True, True)
        if(nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_name += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_name += ', '
            elif i < nr_it_parameters - 1:
                title_name += ', and '
        if i < nr_it_parameters - 1:
            base_out_name += '-'
    if len(partitions) > 1:
        base_out_name += '_combs_vs_' + grp_names[-1] + '_for_' + \
                         '-'.join([a[:min(3, len(a))] for a in map(str, partitions)])
    else:
        base_out_name += '_combs_vs_' + grp_names[-1] + '_for_' + str(partitions)
    title_name += ' Compared to ' + replaceCSVLabels(grp_names[-1], False, True) + ' Values Separately for '
    for i in range(0, nr_partitions):
        title_name += replaceCSVLabels(grp_names[i], True, True)
        if(nr_partitions <= 2):
            if i < nr_partitions - 1:
                title_name += ' and '
        else:
            if i < nr_partitions - 2:
                title_name += ', '
            elif i < nr_partitions - 1:
                title_name += ', and '
    # holds the grouped names/entries within the group names excluding the last entry th
    #grp_values = list(dict.fromkeys([i[0:2] for i in stats.index.values]))
    tex_infos = {'title': title_name,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': make_fig_index,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': ctrl_fig_size,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': figs_externalize,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': None}
    stat_names = list(dict.fromkeys([i[-1] for i in errvalnames if i[-1] != 'count']))
    gloss_calced = False
    for it in errvalnames:
        if it[-1] != 'count':
            tmp = stats[it[0]].unstack()
            tmp = tmp[it[1]]
            tmp1 = tmp.reset_index().set_index(partitions)
            idx_old = None
            for p in tmp1.index:
                if idx_old is not None and idx_old == p:
                    continue
                idx_old = p
                tmp2 = tmp1.loc[p]
                part_props = deepcopy(tmp2.index[0])
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
                tmp2 = tmp2.reset_index().drop(partitions, axis=1)
                tmp2 = tmp2.set_index(it_parameters).T
                if not gloss_calced:
                    if len(it_parameters) > 1:
                        gloss = glossary_from_list([str(b) for a in tmp2.columns for b in a])
                    else:
                        gloss = glossary_from_list([str(a) for a in tmp2.columns])
                    if compare_source:
                        add_to_glossary(compare_source['it_par_select'], gloss)
                        gloss.append({'key': 'cmp', 'description': compare_source['cmp']})
                    gloss_calced = True
                    if gloss:
                        gloss = add_to_glossary_eval(eval_columns + x_axis_column + partitions, gloss)
                        tex_infos['abbreviations'] = gloss
                    else:
                        gloss = add_to_glossary_eval(eval_columns + x_axis_column + partitions)
                        if gloss:
                            tex_infos['abbreviations'] = gloss
                tex_infos['abbreviations'] = add_to_glossary(part_props, tex_infos['abbreviations'])
                if len(it_parameters) > 1:
                    tmp2.columns = ['-'.join(map(str, a)) for a in tmp2.columns]
                    tmp2.columns.name = '-'.join(it_parameters)
                dataf_name = 'data_' + '_'.join(map(str, it)) + '_vs_' + \
                             str(grp_names[-1]) + '_for_' + part_name.replace('.','d') + '.csv'
                dataf_name = dataf_name.replace('%', 'perc')
                if compare_source:
                    tmp2, succ = add_comparison_column(compare_source, dataf_name, tmp2)
                fdataf_name = os.path.join(tdata_folder, dataf_name)
                with open(fdataf_name, 'a') as f:
                    f.write('# ' + str(it[-1]) + ' values for ' + str(it[0]) + ' and properties ' + part_name + '\n')
                    f.write('# Column parameters: ' + '-'.join(it_parameters) + '\n')
                    tmp2.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

                #Construct tex-file
                stats_all = tmp2.stack().reset_index()
                stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
                if (np.isclose(stats_all['min'][0], 0, atol=1e-06) and
                    np.isclose(stats_all['max'][0], 0, atol=1e-06)) or \
                        np.isclose(stats_all['min'][0], stats_all['max'][0]):
                    continue
                #figure types: sharp plot, smooth, const plot, ybar, xbar
                use_limits = {'miny': None, 'maxy': None}
                if np.abs(stats_all['max'][0] - stats_all['min'][0]) < np.abs(stats_all['max'][0] / 200):
                    if stats_all['min'][0] < 0:
                        use_limits['miny'] = round(1.01 * stats_all['min'][0], 6)
                    else:
                        use_limits['miny'] = round(0.99 * stats_all['min'][0], 6)
                    if stats_all['max'][0] < 0:
                        use_limits['maxy'] = round(0.99 * stats_all['max'][0], 6)
                    else:
                        use_limits['maxy'] = round(1.01 * stats_all['max'][0], 6)
                else:
                    if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
                        use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
                    if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
                        use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
                if use_limits['miny'] and use_limits['maxy']:
                    use_log = True if np.abs(np.log10(np.abs(use_limits['miny'])) -
                                             np.log10(np.abs(use_limits['maxy']))) > 1 else False
                    exp_value = is_exp_used(use_limits['miny'], use_limits['maxy'], use_log)
                elif use_limits['miny']:
                    use_log = True if np.abs(np.log10(np.abs(use_limits['miny'])) -
                                             np.log10(np.abs(stats_all['max'][0]))) > 1 else False
                    exp_value = is_exp_used(use_limits['miny'], stats_all['max'][0], use_log)
                elif use_limits['maxy']:
                    use_log = True if np.abs(np.log10(np.abs(stats_all['min'][0])) -
                                             np.log10(np.abs(use_limits['maxy']))) > 1 else False
                    exp_value = is_exp_used(stats_all['min'][0], use_limits['maxy'], use_log)
                else:
                    use_log = True if np.abs(np.log10(np.abs(stats_all['min'][0])) -
                                             np.log10(np.abs(stats_all['max'][0]))) > 1 else False
                    exp_value = is_exp_used(stats_all['min'][0], stats_all['max'][0], use_log)
                is_numeric = pd.to_numeric(tmp2.reset_index()[grp_names[-1]], errors='coerce').notnull().all()
                section_name = replace_stat_names(it[-1]) + ' values for ' +\
                               replaceCSVLabels(str(it[0]), True) +\
                               ' compared to ' + replaceCSVLabels(str(grp_names[-1]), True) +\
                               '\\\\for properties ' + part_name.replace('_', '\\_')
                section_name = split_large_titles(section_name)
                if exp_value and len(section_name.split('\\\\')[-1]) < 70:
                    exp_value = False
                reltex_name = os.path.join(rel_data_path, dataf_name)
                tex_infos['sections'].append({'file': reltex_name,
                                              'name': section_name,
                                              # If caption is None, the field name is used
                                              'caption': replace_stat_names(it[-1]) + ' values for ' +
                                                      replaceCSVLabels(str(it[0]), True) +
                                                      ' compared to ' + replaceCSVLabels(str(grp_names[-1]), True) +
                                                      ' for properties ' + part_name_title,
                                              'fig_type': fig_type,
                                              'plots': list(tmp2.columns.values),
                                              'label_y': replace_stat_names(it[-1]) + findUnit(str(it[0]), units),
                                              'plot_x': str(grp_names[-1]),
                                              'label_x': replaceCSVLabels(str(grp_names[-1])),
                                              'limits': use_limits,
                                              'legend': [tex_string_coding_style(a) for a in list(tmp2.columns.values)],
                                              'legend_cols': None,
                                              'use_marks': use_marks,
                                              'use_log_y_axis': use_log,
                                              'enlarge_title_space': exp_value,
                                              'use_string_labels': True if not is_numeric else False,
                                              'xaxis_txt_rows': 1,
                                              'stat_name': it[-1],
                                              })
                tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    pdfs_info = []
    max_figs_pdf = 50
    if tex_infos['ctrl_fig_size']:  # and not figs_externalize:
        max_figs_pdf = 30
    for st in stat_names:
        # Get list of results using the same statistic
        st_list = list(filter(lambda stat: stat['stat_name'] == st, tex_infos['sections']))
        if len(st_list) > max_figs_pdf:
            st_list2 = [{'figs': st_list[i:i + max_figs_pdf],
                         'pdf_nr': i1 + 1} for i1, i in enumerate(range(0, len(st_list), max_figs_pdf))]
        else:
            st_list2 = [{'figs': st_list, 'pdf_nr': 1}]
        for it in st_list2:
            if len(st_list2) == 1:
                title = replace_stat_names(st) + ' ' + tex_infos['title']
            else:
                title = replace_stat_names(st) + ' ' + tex_infos['title'] + ' -- Part ' + str(it['pdf_nr'])
            pdfs_info.append({'title': title,
                              'texf_name': replace_stat_names(st, False).replace(' ', '_') +
                                           '_' + base_out_name + '_' + str(it['pdf_nr']),
                              'figs_externalize': figs_externalize,
                              'sections': it['figs'],
                              'make_index': tex_infos['make_index'],
                              'ctrl_fig_size': tex_infos['ctrl_fig_size'],
                              'fill_bar': True,
                              'abbreviations': tex_infos['abbreviations']})

    template = ji_env.get_template('usac-testing_2D_plots.tex')
    res = 0
    for it in pdfs_info:
        rendered_tex = template.render(title=it['title'],
                                       make_index=it['make_index'],
                                       ctrl_fig_size=it['ctrl_fig_size'],
                                       figs_externalize=it['figs_externalize'],
                                       fill_bar=it['fill_bar'],
                                       sections=it['sections'],
                                       abbreviations=it['abbreviations'])
        texf_name = it['texf_name'] + '.tex'
        if build_pdf:
            pdf_name = it['texf_name'] + '.pdf'
            res += compile_tex(rendered_tex,
                               tex_folder,
                               texf_name,
                               make_fig_index,
                               os.path.join(pdf_folder, pdf_name),
                               it['figs_externalize'])
        else:
            res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)

    return res


def calcFromFuncAndPlot_2D_partitions(data,
                                      store_path,
                                      tex_file_pre_str,
                                      fig_title_pre_str,
                                      eval_description_path,
                                      eval_columns,#Column names for which statistics are calculated (y-axis)
                                      units,# Units in string format for every entry of eval_columns
                                      it_parameters,# Algorithm parameters to evaluate
                                      partitions,# Data properties to calculate statistics seperately
                                      x_axis_column,# x-axis column name
                                      filter_func=None,
                                      filter_func_args=None,
                                      special_calcs_func = None,
                                      special_calcs_args = None,
                                      calc_func = None,
                                      calc_func_args = None,
                                      compare_source = None,
                                      fig_type='smooth',
                                      use_marks=True,
                                      ctrl_fig_size=True,
                                      make_fig_index=True,
                                      build_pdf=False,
                                      figs_externalize=True):
    if len(x_axis_column) != 1:
        raise ValueError('Only 1 column is allowed to be selected for the x axis')
    fig_types = ['sharp plot', 'smooth', 'const plot', 'ybar', 'xbar']
    if not fig_type in fig_types:
        raise ValueError('Unknown figure type.')
    # if type(data) is not pd.dataframe.DataFrame:
    #     data = pd.utils.from_pandas(data)
    #Filter rows by excluding not successful estimations
    data = data.loc[~((data['R_out(0,0)'] == 0) &
                      (data['R_out(0,1)'] == 0) &
                      (data['R_out(0,2)'] == 0) &
                      (data['R_out(1,0)'] == 0) &
                      (data['R_out(1,1)'] == 0) &
                      (data['R_out(1,2)'] == 0) &
                      (data['R_out(2,0)'] == 0) &
                      (data['R_out(2,1)'] == 0) &
                      (data['R_out(2,2)'] == 0))]
    if filter_func is not None:
        if filter_func_args is None:
            filter_func_args = {'data': data}
        else:
            filter_func_args['data'] = data
        data = filter_func(**filter_func_args)
    if data.empty:
        raise ValueError('No data left after filtering')

    # Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            calc_func_args = {'data': data}
        else:
            calc_func_args['data'] = data
        calc_func_args['eval_columns'] = eval_columns
        calc_func_args['it_parameters'] = it_parameters
        calc_func_args['x_axis_column'] = x_axis_column
        calc_func_args['partitions'] = partitions
        calc_func_args['units'] = units
        ret = calc_func(**calc_func_args)
        df = ret['data']
        eval_columns = ret['eval_columns']
        eval_cols_lname = ret['eval_cols_lname']
        eval_cols_log_scaling = ret['eval_cols_log_scaling']
        eval_init_input = ret['eval_init_input']
        units = ret['units']
        it_parameters = ret['it_parameters']
        x_axis_column = ret['x_axis_column']
        partitions = ret['partitions']
    else:
        raise ValueError('No function for calculating results provided')

    if len(partitions) > 1:
        store_path_sub = os.path.join(store_path, eval_description_path + '_' +
                                      '-'.join(map(str, it_parameters)) + '_vs_' +
                                      '-'.join(map(str, x_axis_column)) + '_for_' +
                                      '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]))
    else:
        store_path_sub = os.path.join(store_path, eval_description_path + '_' +
                                      '-'.join(map(str, it_parameters)) +
                                      '_vs_' + '-'.join(map(str, x_axis_column)) + '_for_' +
                                      str(partitions[0]))
    cnt = 1
    store_path_init = store_path_sub
    while os.path.exists(store_path_sub):
        store_path_sub = store_path_init + '_' + str(int(cnt))
        cnt += 1
    try:
        os.mkdir(store_path_sub)
    except FileExistsError:
        print('Folder', store_path_sub, 'for storing statistics data already exists')
    except:
        print("Unexpected error (Unable to create directory for storing statistics data):", sys.exc_info()[0])
        raise

    if build_pdf:
        pdf_folder = os.path.join(store_path_sub, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            print('Folder', pdf_folder, 'for storing pdf files already exists')
    tex_folder = os.path.join(store_path_sub, 'tex')
    try:
        os.mkdir(tex_folder)
    except FileExistsError:
        print('Folder', tex_folder, 'for storing tex files already exists')
    tdata_folder = os.path.join(tex_folder, 'data')
    try:
        os.mkdir(tdata_folder)
    except FileExistsError:
        print('Folder', tdata_folder, 'for storing data files already exists')

    if compare_source:
        if len(partitions) > 1:
            compare_source['full_path'] = os.path.join(compare_source['store_path'],
                                                       compare_source['eval_description_path'] + '_' +
                                                       '-'.join(map(str, compare_source['it_parameters'])) +
                                                       '_vs_' + '-'.join(map(str, x_axis_column)) + '_for_' +
                                                       '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]))
        else:
            compare_source['full_path'] = os.path.join(compare_source['store_path'],
                                                       compare_source['eval_description_path'] + '_' +
                                                       '-'.join(map(str, compare_source['it_parameters'])) +
                                                       '_vs_' + '-'.join(map(str, x_axis_column)) + '_for_' +
                                                       str(partitions[0]))
        if not os.path.exists(compare_source['full_path']):
            warnings.warn('Path ' + compare_source['full_path'] + ' for comparing results not found. '
                          'Skipping comparison.', UserWarning)
            compare_source = None
        if compare_source:
            compare_source['tdata_folder'] = os.path.join(compare_source['full_path'], 'tex')
            if compare_source and not os.path.exists(compare_source['tdata_folder']):
                warnings.warn('Tex folder ' + compare_source['tdata_folder'] + ' for comparing results not found. '
                              'Skipping comparison.', UserWarning)
                compare_source = None
        if compare_source:
            compare_source['tdata_folder'] = os.path.join(compare_source['tdata_folder'], 'data')
            if compare_source and not os.path.exists(compare_source['tdata_folder']):
                warnings.warn('Data folder ' + compare_source['tdata_folder'] + ' for comparing results not found. '
                              'Skipping comparison.', UserWarning)
                compare_source = None

    if special_calcs_func is not None and special_calcs_args is not None:
        if 'func_name' in special_calcs_args:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_args['func_name'])
        else:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_func.__name__)
        cnt = 1
        calc_vals = True
        special_path_init = special_path_sub
        while os.path.exists(special_path_sub):
            special_path_sub = special_path_init + '_' + str(int(cnt))
            cnt += 1
        try:
            os.mkdir(special_path_sub)
        except FileExistsError:
            print('Folder', special_path_sub, 'for storing statistics data already exists')
        except:
            print("Unexpected error (Unable to create directory for storing special function data):", sys.exc_info()[0])
            calc_vals = False
        if calc_vals:
            special_calcs_args['data'] = df
            special_calcs_args['eval_columns'] = eval_columns
            special_calcs_args['eval_cols_lname'] = eval_cols_lname
            special_calcs_args['units'] = units
            special_calcs_args['x_axis_column'] = x_axis_column
            special_calcs_args['partitions'] = partitions
            special_calcs_args['it_parameters'] = it_parameters
            special_calcs_args['res_folder'] = special_path_sub
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Calculation of specific results failed!', UserWarning)

    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    nr_partitions = len(partitions)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    it_title_part = ''
    for i, val in enumerate(it_parameters):
        base_out_name += val
        it_title_part += replaceCSVLabels(val, True, True)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                it_title_part += ' and '
        else:
            if i < nr_it_parameters - 2:
                it_title_part += ', '
            elif i < nr_it_parameters - 1:
                it_title_part += ', and '
        if i < nr_it_parameters - 1:
            base_out_name += '-'
    title_name += it_title_part

    init_pars_title = ''
    init_pars_out_name = ''
    nr_eval_init_input = len(eval_init_input)
    if nr_eval_init_input > 1:
        for i, val in enumerate(eval_init_input):
            init_pars_out_name += val
            init_pars_title += replaceCSVLabels(val, True, True)
            if (nr_eval_init_input <= 2):
                if i < nr_eval_init_input - 1:
                    init_pars_title += ' and '
            else:
                if i < nr_eval_init_input - 2:
                    init_pars_title += ', '
                elif i < nr_eval_init_input - 1:
                    init_pars_title += ', and '
            if i < nr_eval_init_input - 1:
                init_pars_out_name += '-'
    else:
        init_pars_out_name = eval_init_input[0]
        init_pars_title = replaceCSVLabels(eval_init_input[0], True, True)
    base_out_name += '_combs_vs_' + x_axis_column[0] + '_based_on_' + init_pars_out_name + \
                     '_for_' + '-'.join([a[:min(3, len(a))] for a in map(str, partitions)])
    title_name += ' Compared to ' + replaceCSVLabels(x_axis_column[0], True, True) + \
                  ' Based On ' + init_pars_title + ' Separately for '
    partition_text = ''
    for i, val in enumerate(partitions):
        partition_text += replaceCSVLabels(val, True, True)
        if(nr_partitions <= 2):
            if i < nr_partitions - 1:
                partition_text += ' and '
        else:
            if i < nr_partitions - 2:
                partition_text += ', '
            elif i < nr_partitions - 1:
                partition_text += ', and '
    title_name += partition_text
    tex_infos = {'title': title_name,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': make_fig_index,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': ctrl_fig_size,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': figs_externalize,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': None}
    partition_text_val = []
    for i, val in enumerate(partitions):
        partition_text_val.append([replaceCSVLabels(val)])
        if (nr_partitions <= 2):
            if i < nr_partitions - 1:
                partition_text_val[-1].append(' and ')
        else:
            if i < nr_partitions - 2:
                partition_text_val[-1].append(', ')
            elif i < nr_partitions - 1:
                partition_text_val[-1].append(', and ')
    df = df.groupby(partitions)
    grp_keys = df.groups.keys()
    gloss_calced = False
    for grp in grp_keys:
        partition_text_val1 = ''
        partition_text_val_tmp = deepcopy(partition_text_val)
        if len(partitions) > 1:
            for i, ptv in enumerate(partition_text_val_tmp):
                if '$' == ptv[0][-1]:
                    partition_text_val_tmp[i][0][-1] = '='
                    partition_text_val_tmp[i][0] += str(grp[i]) + '$'
                elif '}' == ptv[0][-1]:
                    partition_text_val_tmp[i][0] += '=' + str(grp[i])
                else:
                    partition_text_val_tmp[i][0] += ' equal to ' + str(grp[i])
            partition_text_val1 = ''.join([''.join(a) for a in partition_text_val_tmp])
        else:
            if '$' == partition_text_val_tmp[0][0][-1]:
                partition_text_val_tmp[0][0][-1] = '='
                partition_text_val_tmp[0][0] += str(grp) + '$'
            elif '}' == partition_text_val_tmp[0][0][-1]:
                partition_text_val_tmp[0][0] += '=' + str(grp)
            else:
                partition_text_val_tmp[0][0] += ' equal to ' + str(grp)
            partition_text_val1 = ''.join(partition_text_val_tmp[0])
        df1 = df.get_group(grp)
        df1.set_index(it_parameters, inplace=True)
        df1 = df1.T
        if not gloss_calced:
            if len(it_parameters) > 1:
                gloss = glossary_from_list([str(b) for a in df1.columns for b in a])
            else:
                gloss = glossary_from_list([str(a) for a in df1.columns])
            if compare_source:
                add_to_glossary(compare_source['it_par_select'], gloss)
                gloss.append({'key': 'cmp', 'description': compare_source['cmp']})
            gloss_calced = True
            if gloss:
                gloss = add_to_glossary_eval(eval_columns + x_axis_column + partitions, gloss)
                tex_infos['abbreviations'] = gloss
            else:
                gloss = add_to_glossary_eval(eval_columns + x_axis_column + partitions)
                if gloss:
                    tex_infos['abbreviations'] = gloss
        if len(partitions) > 1:
            tex_infos['abbreviations'] = add_to_glossary(grp, tex_infos['abbreviations'])
        else:
            tex_infos['abbreviations'] = add_to_glossary([grp], tex_infos['abbreviations'])
        if len(it_parameters) > 1:
            par_cols = ['-'.join(map(str, a)) for a in df1.columns]
            df1.columns = par_cols
            it_pars_cols_name = '-'.join(map(str, it_parameters))
            df1.columns.name = it_pars_cols_name
        else:
            par_cols = [str(a) for a in df1.columns]
            it_pars_cols_name = it_parameters[0]
        tmp = df1.T.drop(partitions, axis=1).reset_index().set_index(x_axis_column + [it_pars_cols_name]).unstack()
        par_cols1 = ['-'.join(map(str, a)) for a in tmp.columns]
        tmp.columns = par_cols1
        tmp.columns.name = 'eval-' + it_pars_cols_name
        if compare_source:
            cmp_fname = 'data_evals_' + init_pars_out_name + '_for_pars_'
            if len(compare_source['it_parameters']) > 1:
                cmp_fname += '-'.join(map(str, compare_source['it_parameters']))
            else:
                cmp_fname += str(compare_source['it_parameters'][0])
            cmp_fname += '_on_partition_'
            cmp_fname += '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]) + '_'
        dataf_name = 'data_evals_' + init_pars_out_name + '_for_pars_' + it_pars_cols_name + '_on_partition_'
        dataf_name += '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]) + '_'
        grp_name = '-'.join([a[:min(4, len(a))] for a in map(str, grp)]) if len(partitions) > 1 else str(grp)
        dataf_name += grp_name.replace('.', 'd')
        dataf_name += '_vs_' + x_axis_column[0] + '.csv'
        if compare_source:
            cmp_fname += grp_name.replace('.', 'd')
            cmp_fname += '_vs_' + x_axis_column[0] + '.csv'
            cmp_its = '-'.join(map(str, compare_source['it_par_select']))
            mult_cols = [a.replace(par_cols[0], cmp_its) for a in par_cols1 if par_cols[0] in a]
            tmp, succ = add_comparison_column(compare_source, cmp_fname, tmp, mult_cols)
            if succ:
                par_cols1 += mult_cols

        fdataf_name = os.path.join(tdata_folder, dataf_name)
        with open(fdataf_name, 'a') as f:
            f.write('# Evaluations on ' + init_pars_out_name + ' for parameter variations of ' +
                    it_pars_cols_name + '\n')
            f.write('# Used data part of ' + '-'.join(map(str, partitions)) + ': ' + grp_name + '\n')
            f.write('# Column parameters: ' + ', '.join(eval_cols_lname) + '\n')
            tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
        for i, ev in enumerate(eval_columns):
            sel_cols = [a for a in par_cols1 if ev in a]
            legend = ['-'.join([b for b in a.split('-') if ev not in b]) for a in sel_cols]

            # Construct tex-file
            stats_all = tmp.loc[:, sel_cols].stack().reset_index()
            stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
            if (np.isclose(stats_all['min'][0], 0, atol=1e-06) and
                np.isclose(stats_all['max'][0], 0, atol=1e-06)) or \
                    np.isclose(stats_all['min'][0], stats_all['max'][0]):
                continue
            # figure types: sharp plot, smooth, const plot, ybar, xbar
            use_limits = {'miny': None, 'maxy': None}
            if np.abs(stats_all['max'][0] - stats_all['min'][0]) < np.abs(stats_all['max'][0] / 200):
                if stats_all['min'][0] < 0:
                    use_limits['miny'] = round(1.01 * stats_all['min'][0], 6)
                else:
                    use_limits['miny'] = round(0.99 * stats_all['min'][0], 6)
                if stats_all['max'][0] < 0:
                    use_limits['maxy'] = round(0.99 * stats_all['max'][0], 6)
                else:
                    use_limits['maxy'] = round(1.01 * stats_all['max'][0], 6)
            else:
                if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 3.291):
                    use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 3.291, 6)
                if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 3.291):
                    use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 3.291, 6)
            if use_limits['miny'] and use_limits['maxy']:
                exp_value = is_exp_used(use_limits['miny'], use_limits['maxy'], eval_cols_log_scaling[i])
            elif use_limits['miny']:
                exp_value = is_exp_used(use_limits['miny'], stats_all['max'][0], eval_cols_log_scaling[i])
            elif use_limits['maxy']:
                exp_value = is_exp_used(stats_all['min'][0], use_limits['maxy'], eval_cols_log_scaling[i])
            else:
                exp_value = is_exp_used(stats_all['min'][0], stats_all['max'][0], eval_cols_log_scaling[i])
            is_numeric = pd.to_numeric(tmp.reset_index()[x_axis_column[0]], errors='coerce').notnull().all()
            reltex_name = os.path.join(rel_data_path, dataf_name)
            fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' based on ' + strToLower(init_pars_title)
            fig_name += '\\\\for '
            fig_name += partition_text_val1 + ' in addition to ' + \
                        '\\\\parameter variations of ' + strToLower(it_title_part) + \
                        '\\\\compared to ' + \
                        replaceCSVLabels(x_axis_column[0], True)
            fig_name = split_large_titles(fig_name)
            if exp_value and len(fig_name.split('\\\\')[-1]) < 70:
                exp_value = False
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': fig_name,
                                          # If caption is None, the field name is used
                                          'caption': fig_name.replace('\\\\', ' '),
                                          'fig_type': fig_type,
                                          'plots': sel_cols,
                                          'label_y': eval_cols_lname[i] + findUnit(ev, units),
                                          'plot_x': x_axis_column[0],
                                          'label_x': replaceCSVLabels(x_axis_column[0]),
                                          'limits': use_limits,
                                          'legend': [tex_string_coding_style(a) for a in legend],
                                          'legend_cols': None,
                                          'use_marks': use_marks,
                                          'use_log_y_axis': eval_cols_log_scaling[i],
                                          'enlarge_title_space': exp_value,
                                          'use_string_labels': True if not is_numeric else False,
                                          'xaxis_txt_rows': 1,
                                          'stat_name': ev,
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    pdfs_info = []
    max_figs_pdf = 50
    if tex_infos['ctrl_fig_size']:  # and not figs_externalize:
        max_figs_pdf = 30
    for i, st in enumerate(eval_columns):
        # Get list of results using the same statistic
        st_list = list(filter(lambda stat: stat['stat_name'] == st, tex_infos['sections']))
        if len(st_list) > max_figs_pdf:
            st_list2 = [{'figs': st_list[i:i + max_figs_pdf],
                         'pdf_nr': i1 + 1} for i1, i in enumerate(range(0, len(st_list), max_figs_pdf))]
        else:
            st_list2 = [{'figs': st_list, 'pdf_nr': 1}]
        for it in st_list2:
            if len(st_list2) == 1:
                title = tex_infos['title'] + ': ' + capitalizeStr(eval_cols_lname[i])
            else:
                title = tex_infos['title'] + ' -- Part ' + str(it['pdf_nr']) + \
                        ' for ' + capitalizeStr(eval_cols_lname[i])
            pdfs_info.append({'title': title,
                              'texf_name': base_out_name + '_' + str(st) + '_' + str(it['pdf_nr']),
                              'figs_externalize': figs_externalize,
                              'sections': it['figs'],
                              'make_index': tex_infos['make_index'],
                              'ctrl_fig_size': tex_infos['ctrl_fig_size'],
                              'fill_bar': True,
                              'abbreviations': tex_infos['abbreviations']})

    template = ji_env.get_template('usac-testing_2D_plots.tex')
    res = 0
    for it in pdfs_info:
        rendered_tex = template.render(title=it['title'],
                                       make_index=it['make_index'],
                                       ctrl_fig_size=it['ctrl_fig_size'],
                                       figs_externalize=it['figs_externalize'],
                                       fill_bar=it['fill_bar'],
                                       sections=it['sections'],
                                       abbreviations=it['abbreviations'])
        texf_name = it['texf_name'] + '.tex'
        if build_pdf:
            pdf_name = it['texf_name'] + '.pdf'
            res += compile_tex(rendered_tex,
                               tex_folder,
                               texf_name,
                               make_fig_index,
                               os.path.join(pdf_folder, pdf_name),
                               it['figs_externalize'])
        else:
            res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)

    return res


def calcSatisticAndPlot_3D(data,
                           store_path,
                           tex_file_pre_str,
                           fig_title_pre_str,
                           eval_description_path,
                           eval_columns,
                           units,
                           it_parameters,
                           xy_axis_columns,
                           filter_func=None,
                           filter_func_args=None,
                           special_calcs_func = None,
                           special_calcs_args = None,
                           calc_func = None,
                           calc_func_args = None,
                           fig_type='surface',
                           use_marks=True,
                           ctrl_fig_size=True,
                           make_fig_index=True,
                           build_pdf=False,
                           figs_externalize=True):
    if len(xy_axis_columns) != 2:
        raise ValueError('Only 2 columns are allowed to be selected for the x and y axis')
    fig_types = ['scatter', 'mesh', 'mesh-scatter', 'mesh', 'surf', 'surf-scatter', 'surf-interior',
                 'surface', 'contour', 'surface-contour']
    if not fig_type in fig_types:
        raise ValueError('Unknown figure type.')
    # if type(data) is not pd.dataframe.DataFrame:
    #     data = pd.utils.from_pandas(data)
    # startt = time.time()
    #Filter rows by excluding not successful estimations
    data = data.loc[~((data['R_out(0,0)'] == 0) &
                      (data['R_out(0,1)'] == 0) &
                      (data['R_out(0,2)'] == 0) &
                      (data['R_out(1,0)'] == 0) &
                      (data['R_out(1,1)'] == 0) &
                      (data['R_out(1,2)'] == 0) &
                      (data['R_out(2,0)'] == 0) &
                      (data['R_out(2,1)'] == 0) &
                      (data['R_out(2,2)'] == 0))]
    if filter_func is not None:
        if filter_func_args is None:
            filter_func_args = {'data': data}
        else:
            filter_func_args['data'] = data
        data = filter_func(**filter_func_args)
    if data.empty:
        raise ValueError('No data left after filtering')

    # Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            calc_func_args = {'data': data}
        else:
            calc_func_args['data'] = data
        calc_func_args['eval_columns'] = eval_columns
        calc_func_args['it_parameters'] = it_parameters
        calc_func_args['xy_axis_columns'] = xy_axis_columns
        ret = calc_func(**calc_func_args)
        df = ret['data']
        eval_columns = ret['eval_columns']
        it_parameters = ret['it_parameters']
        xy_axis_columns = ret['xy_axis_columns']
    else:
        needed_columns = eval_columns + it_parameters + xy_axis_columns
        df = data[needed_columns]

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + '-'.join(map(str, it_parameters)) + '_vs_' +
                                              '-'.join(map(str, xy_axis_columns)))
    cnt = 1
    store_path_init = store_path_sub
    while os.path.exists(store_path_sub):
        store_path_sub = store_path_init + '_' + str(int(cnt))
        cnt += 1
    try:
        os.mkdir(store_path_sub)
    except FileExistsError:
        print('Folder', store_path_sub, 'for storing statistics data already exists')
    except:
        print("Unexpected error (Unable to create directory for storing statistics data):", sys.exc_info()[0])
        raise
    if build_pdf:
        pdf_folder = os.path.join(store_path_sub, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            print('Folder', pdf_folder, 'for storing pdf files already exists')
    tex_folder = os.path.join(store_path_sub, 'tex')
    try:
        os.mkdir(tex_folder)
    except FileExistsError:
        print('Folder', tex_folder, 'for storing tex files already exists')
    tdata_folder = os.path.join(tex_folder, 'data')
    try:
        os.mkdir(tdata_folder)
    except FileExistsError:
        print('Folder', tdata_folder, 'for storing data files already exists')

    #Group by USAC parameters 5&6 and calculate the statistic
    stats = df.groupby(it_parameters + xy_axis_columns).describe()
    if special_calcs_func is not None and special_calcs_args is not None:
        if 'func_name' in special_calcs_args:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_args['func_name'])
        else:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_func.__name__)
        cnt = 1
        calc_vals = True
        special_path_init = special_path_sub
        while os.path.exists(special_path_sub):
            special_path_sub = special_path_init + '_' + str(int(cnt))
            cnt += 1
        try:
            os.mkdir(special_path_sub)
        except FileExistsError:
            print('Folder', special_path_sub, 'for storing statistics data already exists')
        except:
            print("Unexpected error (Unable to create directory for storing special function data):", sys.exc_info()[0])
            calc_vals = False
        if calc_vals:
            special_calcs_args['data'] = stats
            special_calcs_args['it_parameters'] = it_parameters
            special_calcs_args['eval_columns'] = eval_columns
            special_calcs_args['xy_axis_columns'] = xy_axis_columns
            special_calcs_args['res_folder'] = special_path_sub
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Calculation of specific results failed!', UserWarning)
    errvalnames = stats.columns.values # Includes statistic name and error value names
    grp_names = stats.index.names #As used when generating the groups
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    for i in range(0, nr_it_parameters):
        base_out_name += grp_names[i]
        title_name += replaceCSVLabels(grp_names[i], True, True)
        if(nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_name += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_name += ', '
            elif i < nr_it_parameters - 1:
                title_name += ', and '
        if i < nr_it_parameters - 1:
            base_out_name += '-'
    base_out_name += '_combs_vs_' + grp_names[-2] + '_and_' + grp_names[-1]
    title_name += ' Compared to ' + replaceCSVLabels(grp_names[-2], False, True) + ' and ' + \
                  replaceCSVLabels(grp_names[-1], False, True) + ' Values'
    tex_infos = {'title': title_name,
                 'sections': [],
                 'use_fixed_caption': False,
                 'make_index': make_fig_index,#Builds an index with hyperrefs on the beginning of the pdf
                 'ctrl_fig_size': ctrl_fig_size,#If True, the figures are adapted to the page height if they are too big
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': None}
    # Get names of statistics
    stat_names = list(dict.fromkeys([i[-1] for i in errvalnames if i[-1] != 'count']))
    gloss_calced = False
    for it in errvalnames:
        if it[-1] != 'count':
            tmp = stats[it[0]].unstack()
            tmp = tmp[it[1]]
            tmp = tmp.unstack()
            tmp = tmp.T
            if not gloss_calced:
                if len(it_parameters) > 1:
                    gloss = glossary_from_list([str(b) for a in tmp.columns for b in a])
                else:
                    gloss = glossary_from_list([str(a) for a in tmp.columns])
                gloss_calced = True
                if gloss:
                    gloss = add_to_glossary_eval(eval_columns + xy_axis_columns, gloss)
                    tex_infos['abbreviations'] = gloss
                else:
                    gloss = add_to_glossary_eval(eval_columns + xy_axis_columns)
                    if gloss:
                        tex_infos['abbreviations'] = gloss
            if len(it_parameters) > 1:
                tmp.columns = ['-'.join(map(str, a)) for a in tmp.columns]
                tmp.columns.name = '-'.join(grp_names[0:-2])
            tmp = tmp.reset_index()
            nr_equal_ss = int(tmp.groupby(tmp.columns.values[0]).size().array[0])
            dataf_name = 'data_' + '_'.join(map(str, it)) + '_vs_' + \
                       str(grp_names[-2]) + '_and_' + str(grp_names[-1]) + '.csv'
            dataf_name = dataf_name.replace('%', 'perc')
            fdataf_name = os.path.join(tdata_folder, dataf_name)
            with open(fdataf_name, 'a') as f:
                f.write('# ' + str(it[-1]) + ' values for ' + str(it[0]) + '\n')
                f.write('# Column parameters: ' + '-'.join(it_parameters) + '\n')
                tmp.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

            #Construct tex-file information
            stats_all = tmp.drop(tmp.columns.values[0:2], axis=1).stack().reset_index()
            stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
            if (np.isclose(stats_all['min'][0], 0, atol=1e-06) and
                np.isclose(stats_all['max'][0], 0, atol=1e-06)) or \
                    np.isclose(stats_all['min'][0], stats_all['max'][0]):
                continue
            use_limits = {'minz': None, 'maxz': None}
            if np.abs(stats_all['max'][0] - stats_all['min'][0]) < np.abs(stats_all['max'][0] / 100):
                if stats_all['min'][0] < 0:
                    use_limits['minz'] = round(1.01 * stats_all['min'][0], 6)
                else:
                    use_limits['minz'] = round(0.99 * stats_all['min'][0], 6)
                if stats_all['max'][0] < 0:
                    use_limits['maxz'] = round(0.99 * stats_all['max'][0], 6)
                else:
                    use_limits['maxz'] = round(1.01 * stats_all['max'][0], 6)
            #figure types:
            # scatter, mesh, mesh-scatter, mesh, surf, surf-scatter, surf-interior, surface, contour, surface-contour
            reltex_name = os.path.join(rel_data_path, dataf_name)
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': replace_stat_names(it[-1]) + ' value for ' +
                                                  replaceCSVLabels(str(it[0]), True) +
                                                  ' compared to ' +
                                                  replaceCSVLabels(str(grp_names[-2]), True) +
                                                  ' and ' + replaceCSVLabels(str(grp_names[-1]), True),
                                          'fig_type': fig_type,
                                          'stat_name': it[-1],
                                          'plots_z': list(tmp.columns.values)[2:],
                                          'diff_z_labels': False,
                                          'label_z': replace_stat_names(it[-1]) + findUnit(str(it[0]), units),
                                          'plot_x': str(tmp.columns.values[1]),
                                          'label_x': replaceCSVLabels(str(tmp.columns.values[1])) +
                                                     findUnit(str(tmp.columns.values[1]), units),
                                          'plot_y': str(tmp.columns.values[0]),
                                          'label_y': replaceCSVLabels(str(tmp.columns.values[0])) +
                                                     findUnit(str(tmp.columns.values[0]), units),
                                          'legend': [tex_string_coding_style(a) for a in list(tmp.columns.values)[2:]],
                                          'use_marks': use_marks,
                                          'mesh_cols': nr_equal_ss,
                                          'use_log_z_axis': False,
                                          'limits': use_limits
                                          })

    pdfs_info = []
    max_figs_pdf = 40
    if tex_infos['ctrl_fig_size']:# and not figs_externalize:
        max_figs_pdf = 30
    # elif figs_externalize:
    #     max_figs_pdf = 40
    for st in stat_names:
        #Get list of results using the same statistic
        st_list = list(filter(lambda stat: stat['stat_name'] == st, tex_infos['sections']))
        act_figs = 0
        cnt = 1
        i_old = 0
        i_new = 0
        st_list2 = []
        for i_new, a in enumerate(st_list):
            act_figs += len(a['plots_z'])
            if act_figs > max_figs_pdf:
                st_list2.append({'figs': st_list[i_old:(i_new + 1)], 'pdf_nr': cnt})
                cnt += 1
                i_old = i_new + 1
                act_figs = 0
        if (i_new + 1) != i_old:
            st_list2.append({'figs': st_list[i_old:(i_new + 1)], 'pdf_nr': cnt})
        for it in st_list2:
            if len(st_list2) == 1:
                title = replace_stat_names(st) + ' ' + tex_infos['title']
            else:
                title = replace_stat_names(st) + ' ' + tex_infos['title'] + ' -- Part ' + str(it['pdf_nr'])
            pdfs_info.append({'title': title,
                              'texf_name': replace_stat_names(st, False).replace(' ', '_') +
                                           '_' + base_out_name + '_' + str(it['pdf_nr']),
                              'figs_externalize': figs_externalize,
                              'sections': it['figs'],
                              'make_index': tex_infos['make_index'],
                              'use_fixed_caption': tex_infos['use_fixed_caption'],
                              'ctrl_fig_size': tex_infos['ctrl_fig_size'],
                              'abbreviations': tex_infos['abbreviations']})

    # endt = time.time()
    # print(endt - startt)

    template = ji_env.get_template('usac-testing_3D_plots.tex')
    res = 0
    for it in pdfs_info:
        rendered_tex = template.render(title=it['title'],
                                       make_index=it['make_index'],
                                       use_fixed_caption=it['use_fixed_caption'],
                                       ctrl_fig_size=it['ctrl_fig_size'],
                                       figs_externalize=it['figs_externalize'],
                                       sections=it['sections'],
                                       abbreviations=it['abbreviations'])
        texf_name = it['texf_name'] + '.tex'
        if build_pdf:
            pdf_name = it['texf_name'] + '.pdf'
            res += compile_tex(rendered_tex,
                               tex_folder,
                               texf_name,
                               make_fig_index,
                               os.path.join(pdf_folder, pdf_name),
                               it['figs_externalize'])
        else:
            res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)

    return res


def calcFromFuncAndPlot_3D(data,
                           store_path,
                           tex_file_pre_str,
                           fig_title_pre_str,
                           eval_description_path,
                           eval_columns,
                           units,
                           it_parameters,
                           xy_axis_columns,
                           filter_func=None,
                           filter_func_args=None,
                           special_calcs_func = None,
                           special_calcs_args = None,
                           calc_func = None,
                           calc_func_args = None,
                           fig_type='surface',
                           use_marks=True,
                           ctrl_fig_size=True,
                           make_fig_index=True,
                           build_pdf=False,
                           figs_externalize=True):
    # if len(xy_axis_columns) != 2:
    #     raise ValueError('Only 2 columns are allowed to be selected for the x and y axis')
    fig_types = ['scatter', 'mesh', 'mesh-scatter', 'mesh', 'surf', 'surf-scatter', 'surf-interior',
                 'surface', 'contour', 'surface-contour']
    if not fig_type in fig_types:
        raise ValueError('Unknown figure type.')
    #Filter rows by excluding not successful estimations
    data = data.loc[~((data['R_out(0,0)'] == 0) &
                      (data['R_out(0,1)'] == 0) &
                      (data['R_out(0,2)'] == 0) &
                      (data['R_out(1,0)'] == 0) &
                      (data['R_out(1,1)'] == 0) &
                      (data['R_out(1,2)'] == 0) &
                      (data['R_out(2,0)'] == 0) &
                      (data['R_out(2,1)'] == 0) &
                      (data['R_out(2,2)'] == 0))]
    if filter_func is not None:
        if filter_func_args is None:
            filter_func_args = {'data': data}
        else:
            filter_func_args['data'] = data
        data = filter_func(**filter_func_args)
    if data.empty:
        raise ValueError('No data left after filtering')

    # Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            calc_func_args = {'data': data}
        else:
            calc_func_args['data'] = data
        calc_func_args['eval_columns'] = eval_columns
        calc_func_args['it_parameters'] = it_parameters
        calc_func_args['xy_axis_columns'] = xy_axis_columns
        calc_func_args['units'] = units
        ret = calc_func(**calc_func_args)
        df = ret['data']
        eval_columns = ret['eval_columns']
        eval_cols_lname = ret['eval_cols_lname']
        eval_cols_log_scaling = ret['eval_cols_log_scaling']
        eval_init_input = ret['eval_init_input']
        units = ret['units']
        it_parameters = ret['it_parameters']
        xy_axis_columns = ret['xy_axis_columns']
    else:
        raise ValueError('No function for calculating results provided')

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + '-'.join(map(str, it_parameters)) + '_vs_' +
                                              '-'.join(map(str, xy_axis_columns)))
    cnt = 1
    store_path_init = store_path_sub
    while os.path.exists(store_path_sub):
        store_path_sub = store_path_init + '_' + str(int(cnt))
        cnt += 1
    try:
        os.mkdir(store_path_sub)
    except FileExistsError:
        print('Folder', store_path_sub, 'for storing statistics data already exists')
    except:
        print("Unexpected error (Unable to create directory for storing statistics data):", sys.exc_info()[0])
        raise

    if build_pdf:
        pdf_folder = os.path.join(store_path_sub, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            print('Folder', pdf_folder, 'for storing pdf files already exists')
    tex_folder = os.path.join(store_path_sub, 'tex')
    try:
        os.mkdir(tex_folder)
    except FileExistsError:
        print('Folder', tex_folder, 'for storing tex files already exists')
    tdata_folder = os.path.join(tex_folder, 'data')
    try:
        os.mkdir(tdata_folder)
    except FileExistsError:
        print('Folder', tdata_folder, 'for storing data files already exists')
    if special_calcs_func is not None and special_calcs_args is not None:
        if 'func_name' in special_calcs_args:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_args['func_name'])
        else:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_func.__name__)
        cnt = 1
        calc_vals = True
        special_path_init = special_path_sub
        while os.path.exists(special_path_sub):
            special_path_sub = special_path_init + '_' + str(int(cnt))
            cnt += 1
        try:
            os.mkdir(special_path_sub)
        except FileExistsError:
            print('Folder', special_path_sub, 'for storing statistics data already exists')
        except:
            print("Unexpected error (Unable to create directory for storing special function data):", sys.exc_info()[0])
            calc_vals = False
        if calc_vals:
            special_calcs_args['data'] = df
            special_calcs_args['eval_columns'] = eval_columns
            special_calcs_args['eval_cols_lname'] = eval_cols_lname
            special_calcs_args['units'] = units
            special_calcs_args['xy_axis_columns'] = xy_axis_columns
            special_calcs_args['it_parameters'] = it_parameters
            special_calcs_args['res_folder'] = special_path_sub
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Calculation of specific results failed!', UserWarning)
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    for i, val in enumerate(it_parameters):
        base_out_name += val
        title_name += replaceCSVLabels(val, True, True)
        if(nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_name += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_name += ', '
            elif i < nr_it_parameters - 1:
                title_name += ', and '
        if i < nr_it_parameters - 1:
            base_out_name += '-'
    init_pars_title = ''
    init_pars_out_name = ''
    nr_eval_init_input = len(eval_init_input)
    if nr_eval_init_input > 1:
        for i, val in enumerate(eval_init_input):
            init_pars_out_name += val
            init_pars_title += replaceCSVLabels(val, True, True)
            if(nr_eval_init_input <= 2):
                if i < nr_eval_init_input - 1:
                    init_pars_title += ' and '
            else:
                if i < nr_eval_init_input - 2:
                    init_pars_title += ', '
                elif i < nr_eval_init_input - 1:
                    init_pars_title += ', and '
            if i < nr_eval_init_input - 1:
                init_pars_out_name += '-'
    else:
        init_pars_out_name = eval_init_input[0]
        init_pars_title = replaceCSVLabels(eval_init_input[0], True, True)
    base_out_name += '_combs_vs_' + xy_axis_columns[0] + '_and_' + xy_axis_columns[1] + '_based_on_' + \
                     init_pars_out_name
    title_name += ' Compared to ' + replaceCSVLabels(xy_axis_columns[0], True, True) + ' and ' + \
                  replaceCSVLabels(xy_axis_columns[1], True, True) + ' Based On ' + init_pars_title
    tex_infos = {'title': title_name,
                 'sections': [],
                 'use_fixed_caption': True,
                 'make_index': make_fig_index,#Builds an index with hyperrefs on the beginning of the pdf
                 'ctrl_fig_size': ctrl_fig_size,#If True, the figures are adapted to the page height if they are too big
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': None}
    df = df.groupby(it_parameters)
    grp_keys = df.groups.keys()
    if len(it_parameters) > 1:
        gloss = glossary_from_list([str(b) for a in grp_keys for b in a])
    else:
        gloss = glossary_from_list([str(a) for a in grp_keys])
    if gloss:
        gloss = add_to_glossary_eval(eval_columns + xy_axis_columns, gloss)
        tex_infos['abbreviations'] = gloss
    else:
        gloss = add_to_glossary_eval(eval_columns + xy_axis_columns)
        if gloss:
            tex_infos['abbreviations'] = gloss
    for grp in grp_keys:
        if len(it_parameters) > 1:
            dataf_name = 'data_evals_' + init_pars_out_name + '_for_pars_' + \
                         '-'.join(map(str, grp)) + '_vs_' + xy_axis_columns[0] + \
                         '_and_' + xy_axis_columns[1] + '.csv'
        else:
            dataf_name = 'data_evals_' + init_pars_out_name + '_for_pars_' + \
                         str(grp) + '_vs_' + xy_axis_columns[0] + \
                         '_and_' + xy_axis_columns[1] + '.csv'
        fdataf_name = os.path.join(tdata_folder, dataf_name)
        tmp = df.get_group(grp)
        tmp = tmp.drop(it_parameters, axis=1)
        nr_equal_ss = int(tmp.groupby(xy_axis_columns[0]).size().array[0])
        with open(fdataf_name, 'a') as f:
            f.write('# Evaluations on ' + init_pars_out_name + ' for parameter variations of ' +
                    '-'.join(map(str, it_parameters)) + '\n')
            if len(it_parameters) > 1:
                f.write('# Used parameter values: ' + '-'.join(map(str, grp)) + '\n')
            else:
                f.write('# Used parameter values: ' + str(grp) + '\n')
            f.write('# Column parameters: ' + ', '.join(eval_cols_lname) + '\n')
            tmp.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')
        for i, it in enumerate(eval_columns):
            #Construct tex-file information
            stats_all = {'min': tmp[it].min(), 'max': tmp[it].max()}
            if (np.isclose(stats_all['min'], 0, atol=1e-06) and
                np.isclose(stats_all['max'], 0, atol=1e-06)) or \
                    np.isclose(stats_all['min'], stats_all['max']):
                continue
            use_limits = {'minz': None, 'maxz': None}
            if np.abs(stats_all['max'] - stats_all['min']) < np.abs(stats_all['max'] / 200):
                if stats_all['min'] < 0:
                    use_limits['minz'] = round(1.01 * stats_all['min'], 6)
                else:
                    use_limits['minz'] = round(0.99 * stats_all['min'], 6)
                if stats_all['max'] < 0:
                    use_limits['maxz'] = round(0.99 * stats_all['max'], 6)
                else:
                    use_limits['maxz'] = round(1.01 * stats_all['max'], 6)
            #figure types:
            # scatter, mesh, mesh-scatter, mesh, surf, surf-scatter, surf-interior, surface, contour, surface-contour
            reltex_name = os.path.join(rel_data_path, dataf_name)
            if len(it_parameters) > 1:
                fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' based on ' + strToLower(init_pars_title) +\
                           ' for parameters ' + tex_string_coding_style('-'.join(map(str, grp))) + ' compared to ' + \
                           replaceCSVLabels(xy_axis_columns[0], True) + ' and ' + \
                           replaceCSVLabels(xy_axis_columns[1], True)
                legends = [eval_cols_lname[i] + ' for ' + tex_string_coding_style('-'.join(map(str, grp)))]
            else:
                fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' based on ' + strToLower(init_pars_title) + \
                           ' for parameters ' + tex_string_coding_style(str(grp)) + ' compared to ' + \
                           replaceCSVLabels(xy_axis_columns[0], True) + ' and ' + \
                           replaceCSVLabels(xy_axis_columns[1], True)
                legends = [eval_cols_lname[i] + ' for ' + tex_string_coding_style(str(grp))]
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': fig_name,
                                          'fig_type': fig_type,
                                          'stat_name': it,
                                          'plots_z': [it],
                                          'diff_z_labels': False,
                                          'label_z': eval_cols_lname[i] + findUnit(str(it), units),
                                          'plot_x': str(xy_axis_columns[0]),
                                          'label_x': replaceCSVLabels(str(xy_axis_columns[0])) +
                                                     findUnit(str(xy_axis_columns[0]), units),
                                          'plot_y': str(xy_axis_columns[1]),
                                          'label_y': replaceCSVLabels(str(xy_axis_columns[1])) +
                                                     findUnit(str(xy_axis_columns[1]), units),
                                          'legend': legends,
                                          'use_marks': use_marks,
                                          'mesh_cols': nr_equal_ss,
                                          'use_log_z_axis': eval_cols_log_scaling[i],
                                          'limits': use_limits
                                          })

    pdfs_info = []
    max_figs_pdf = 50
    if tex_infos['ctrl_fig_size']:
        max_figs_pdf = 30
    for st in eval_columns:
        # Get list of results using the same statistic
        st_list = list(filter(lambda stat: stat['stat_name'] == st, tex_infos['sections']))
        if len(st_list) > max_figs_pdf:
            st_list2 = [{'figs': st_list[i:i + max_figs_pdf],
                         'pdf_nr': i1 + 1} for i1, i in enumerate(range(0, len(st_list), max_figs_pdf))]
        else:
            st_list2 = [{'figs': st_list, 'pdf_nr': 1}]
        for i, it in enumerate(st_list2):
            if len(st_list2) == 1:
                title = tex_infos['title'] + ': ' + capitalizeStr(eval_cols_lname[i])
            else:
                title = tex_infos['title'] + ' -- Part ' + str(it['pdf_nr']) + \
                        ' for ' + capitalizeStr(eval_cols_lname[i])
            pdfs_info.append({'title': title,
                              'texf_name': base_out_name + '_' + str(st) + '_' + str(it['pdf_nr']),
                              'figs_externalize': figs_externalize,
                              'sections': it['figs'],
                              'make_index': tex_infos['make_index'],
                              'use_fixed_caption': tex_infos['use_fixed_caption'],
                              'ctrl_fig_size': tex_infos['ctrl_fig_size'],
                              'abbreviations': tex_infos['abbreviations']})

    template = ji_env.get_template('usac-testing_3D_plots.tex')
    res = 0
    for it in pdfs_info:
        rendered_tex = template.render(title=it['title'],
                                       make_index=it['make_index'],
                                       use_fixed_caption=it['use_fixed_caption'],
                                       ctrl_fig_size=it['ctrl_fig_size'],
                                       figs_externalize=it['figs_externalize'],
                                       sections=it['sections'],
                                       abbreviations=it['abbreviations'])
        texf_name = it['texf_name'] + '.tex'
        if build_pdf:
            pdf_name = it['texf_name'] + '.pdf'
            res += compile_tex(rendered_tex,
                               tex_folder,
                               texf_name,
                               make_fig_index,
                               os.path.join(pdf_folder, pdf_name),
                               it['figs_externalize'])
        else:
            res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)

    return res


def calcFromFuncAndPlot_3D_partitions(data,
                                      store_path,
                                      tex_file_pre_str,
                                      fig_title_pre_str,
                                      eval_description_path,
                                      eval_columns,#Column names for which statistics are calculated (y-axis)
                                      units,# Units in string format for every entry of eval_columns
                                      it_parameters,# Algorithm parameters to evaluate
                                      partitions,# Data properties to calculate statistics seperately
                                      xy_axis_columns,# x- and y-axis column names
                                      filter_func=None,
                                      filter_func_args=None,
                                      special_calcs_func = None,
                                      special_calcs_args = None,
                                      calc_func = None,
                                      calc_func_args = None,
                                      fig_type='surface',
                                      use_marks=True,
                                      ctrl_fig_size=True,
                                      make_fig_index=True,
                                      build_pdf=False,
                                      figs_externalize=True):
    fig_types = ['scatter', 'mesh', 'mesh-scatter', 'mesh', 'surf', 'surf-scatter', 'surf-interior',
                 'surface', 'contour', 'surface-contour']
    if not fig_type in fig_types:
        raise ValueError('Unknown figure type.')
    # if type(data) is not pd.dataframe.DataFrame:
    #     data = pd.utils.from_pandas(data)
    #Filter rows by excluding not successful estimations
    data = data.loc[~((data['R_out(0,0)'] == 0) &
                      (data['R_out(0,1)'] == 0) &
                      (data['R_out(0,2)'] == 0) &
                      (data['R_out(1,0)'] == 0) &
                      (data['R_out(1,1)'] == 0) &
                      (data['R_out(1,2)'] == 0) &
                      (data['R_out(2,0)'] == 0) &
                      (data['R_out(2,1)'] == 0) &
                      (data['R_out(2,2)'] == 0))]
    if filter_func is not None:
        if filter_func_args is None:
            filter_func_args = {'data': data}
        else:
            filter_func_args['data'] = data
        data = filter_func(**filter_func_args)
    if data.empty:
        raise ValueError('No data left after filtering')

    # Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            calc_func_args = {'data': data}
        else:
            calc_func_args['data'] = data
        calc_func_args['eval_columns'] = eval_columns
        calc_func_args['it_parameters'] = it_parameters
        calc_func_args['xy_axis_columns'] = xy_axis_columns
        calc_func_args['partitions'] = partitions
        calc_func_args['units'] = units
        ret = calc_func(**calc_func_args)
        df = ret['data']
        eval_columns = ret['eval_columns']
        eval_cols_lname = ret['eval_cols_lname']
        eval_cols_log_scaling = ret['eval_cols_log_scaling']
        eval_init_input = ret['eval_init_input']
        units = ret['units']
        it_parameters = ret['it_parameters']
        xy_axis_columns = ret['xy_axis_columns']
        partitions = ret['partitions']
    else:
        raise ValueError('No function for calculating results provided')

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + '-'.join(map(str, it_parameters)) + '_vs_' +
                                              '-'.join(map(str, xy_axis_columns)) + '_for_' +
                                              '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]))
    cnt = 1
    store_path_init = store_path_sub
    while os.path.exists(store_path_sub):
        store_path_sub = store_path_init + '_' + str(int(cnt))
        cnt += 1
    try:
        os.mkdir(store_path_sub)
    except FileExistsError:
        print('Folder', store_path_sub, 'for storing statistics data already exists')
    except:
        print("Unexpected error (Unable to create directory for storing statistics data):", sys.exc_info()[0])
        raise

    if build_pdf:
        pdf_folder = os.path.join(store_path_sub, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            print('Folder', pdf_folder, 'for storing pdf files already exists')
    tex_folder = os.path.join(store_path_sub, 'tex')
    try:
        os.mkdir(tex_folder)
    except FileExistsError:
        print('Folder', tex_folder, 'for storing tex files already exists')
    tdata_folder = os.path.join(tex_folder, 'data')
    try:
        os.mkdir(tdata_folder)
    except FileExistsError:
        print('Folder', tdata_folder, 'for storing data files already exists')

    if special_calcs_func is not None and special_calcs_args is not None:
        if 'func_name' in special_calcs_args:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_args['func_name'])
        else:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_func.__name__)
        cnt = 1
        calc_vals = True
        special_path_init = special_path_sub
        while os.path.exists(special_path_sub):
            special_path_sub = special_path_init + '_' + str(int(cnt))
            cnt += 1
        try:
            os.mkdir(special_path_sub)
        except FileExistsError:
            print('Folder', special_path_sub, 'for storing statistics data already exists')
        except:
            print("Unexpected error (Unable to create directory for storing special function data):", sys.exc_info()[0])
            calc_vals = False
        if calc_vals:
            special_calcs_args['data'] = df
            special_calcs_args['eval_columns'] = eval_columns
            special_calcs_args['eval_cols_lname'] = eval_cols_lname
            special_calcs_args['units'] = units
            special_calcs_args['xy_axis_columns'] = xy_axis_columns
            special_calcs_args['partitions'] = partitions
            special_calcs_args['it_parameters'] = it_parameters
            special_calcs_args['res_folder'] = special_path_sub
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Calculation of specific results failed!', UserWarning)

    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    nr_partitions = len(partitions)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    it_title_part = ''
    for i, val in enumerate(it_parameters):
        base_out_name += val
        it_title_part += replaceCSVLabels(val, True, True)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                it_title_part += ' and '
        else:
            if i < nr_it_parameters - 2:
                it_title_part += ', '
            elif i < nr_it_parameters - 1:
                it_title_part += ', and '
        if i < nr_it_parameters - 1:
            base_out_name += '-'
    title_name += it_title_part

    if eval_init_input:
        init_pars_title = ''
        init_pars_out_name = ''
        nr_eval_init_input = len(eval_init_input)
        if nr_eval_init_input > 1:
            for i, val in enumerate(eval_init_input):
                init_pars_out_name += val
                init_pars_title += replaceCSVLabels(val, True, True)
                if (nr_eval_init_input <= 2):
                    if i < nr_eval_init_input - 1:
                        init_pars_title += ' and '
                else:
                    if i < nr_eval_init_input - 2:
                        init_pars_title += ', '
                    elif i < nr_eval_init_input - 1:
                        init_pars_title += ', and '
                if i < nr_eval_init_input - 1:
                    init_pars_out_name += '-'
        else:
            init_pars_out_name = eval_init_input[0]
            init_pars_title = replaceCSVLabels(eval_init_input[0], True, True)
        base_out_name += '_combs_vs_' + xy_axis_columns[0] + '_and_' + xy_axis_columns[1] + '_based_on_' + \
                         init_pars_out_name + '_for_' + '-'.join([a[:min(3, len(a))] for a in map(str, partitions)])
        title_name += ' Compared to ' + replaceCSVLabels(xy_axis_columns[0], True, True) + ' and ' + \
                      replaceCSVLabels(xy_axis_columns[1], True, True) + ' Based On ' + init_pars_title + \
                      ' Separately for '
    else:
        base_out_name += '_combs_vs_' + xy_axis_columns[0] + '_and_' + xy_axis_columns[1] + \
                         '_for_' + '-'.join([a[:min(3, len(a))] for a in map(str, partitions)])
        title_name += ' Compared to ' + replaceCSVLabels(xy_axis_columns[0], True, True) + ' and ' + \
                      replaceCSVLabels(xy_axis_columns[1], True, True) + ' Separately for '

    partition_text = ''
    for i, val in enumerate(partitions):
        partition_text += replaceCSVLabels(val, True, True)
        if(nr_partitions <= 2):
            if i < nr_partitions - 1:
                partition_text += ' and '
        else:
            if i < nr_partitions - 2:
                partition_text += ', '
            elif i < nr_partitions - 1:
                partition_text += ', and '
    title_name += partition_text
    tex_infos = {'title': title_name,
                 'sections': [],
                 'use_fixed_caption': True,
                 'make_index': make_fig_index,  # Builds an index with hyperrefs on the beginning of the pdf
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': ctrl_fig_size,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': None}
    partition_text_val = []
    for i, val in enumerate(partitions):
        partition_text_val.append([replaceCSVLabels(val)])
        if (nr_partitions <= 2):
            if i < nr_partitions - 1:
                partition_text_val[-1].append(' and ')
        else:
            if i < nr_partitions - 2:
                partition_text_val[-1].append(', ')
            elif i < nr_partitions - 1:
                partition_text_val[-1].append(', and ')
    df = df.groupby(partitions)
    grp_keys = df.groups.keys()
    gloss_calced = False
    for grp in grp_keys:
        partition_text_val1 = ''
        partition_text_val_tmp = deepcopy(partition_text_val)
        if len(partitions) > 1:
            for i, ptv in enumerate(partition_text_val_tmp):
                if '$' == ptv[0][-1]:
                    partition_text_val_tmp[i][0][-1] = '='
                    partition_text_val_tmp[i][0] += str(grp[i]) + '$'
                elif '}' == ptv[0][-1]:
                    partition_text_val_tmp[i][0] += '=' + str(grp[i])
                else:
                    partition_text_val_tmp[i][0] += ' equal to ' + str(grp[i])
            partition_text_val1 = ''.join([''.join(a) for a in partition_text_val_tmp])
        else:
            if '$' == partition_text_val_tmp[0][0][-1]:
                partition_text_val_tmp[0][0][-1] = '='
                partition_text_val_tmp[0][0] += str(grp) + '$'
            elif '}' == partition_text_val_tmp[0][0][-1]:
                partition_text_val_tmp[0][0] += '=' + str(grp)
            else:
                partition_text_val_tmp[0][0] += ' equal to ' + str(grp)
            partition_text_val1 = ''.join(partition_text_val_tmp[0])
        df1 = df.get_group(grp)
        df1.set_index(it_parameters, inplace=True)
        df1 = df1.T
        if not gloss_calced:
            if len(it_parameters) > 1:
                gloss = glossary_from_list([str(b) for a in df1.columns for b in a])
            else:
                gloss = glossary_from_list([str(a) for a in df1.columns])
            gloss_calced = True
            if gloss:
                gloss = add_to_glossary_eval(eval_columns + xy_axis_columns + partitions, gloss)
                tex_infos['abbreviations'] = gloss
            else:
                gloss = add_to_glossary_eval(eval_columns + xy_axis_columns + partitions)
                if gloss:
                    tex_infos['abbreviations'] = gloss
        if len(partitions) > 1:
            tex_infos['abbreviations'] = add_to_glossary(grp, tex_infos['abbreviations'])
        else:
            tex_infos['abbreviations'] = add_to_glossary([grp], tex_infos['abbreviations'])
        if len(it_parameters) > 1:
            par_cols = ['-'.join(map(str, a)) for a in df1.columns]
            df1.columns = par_cols
            it_pars_cols_name = '-'.join(map(str, it_parameters))
            df1.columns.name = it_pars_cols_name
        else:
            it_pars_cols_name = it_parameters[0]
        tmp = df1.T.drop(partitions, axis=1).reset_index()
        tmp = tmp.groupby(it_pars_cols_name)
        grp_keys_it = tmp.groups.keys()
        for grp_it in grp_keys_it:
            tmp1 = tmp.get_group(grp_it)
            tmp1 = tmp1.drop(it_pars_cols_name, axis=1)
            nr_equal_ss = int(tmp1.groupby(xy_axis_columns[0]).size().array[0])

            if eval_init_input:
                dataf_name = 'data_evals_' + init_pars_out_name + '_for_pars_' + grp_it + \
                             '_on_partition_'
            else:
                dataf_name = 'data_evals_for_pars_' + grp_it + '_on_partition_'
            dataf_name += '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]) + '_'
            grp_name = '-'.join([a[:min(4, len(a))] for a in map(str, grp)]) if len(partitions) > 1 else str(grp)
            dataf_name += grp_name.replace('.', 'd')
            dataf_name += '_vs_' + xy_axis_columns[0] + '_and_' + xy_axis_columns[1] + '.csv'
            fdataf_name = os.path.join(tdata_folder, dataf_name)
            with open(fdataf_name, 'a') as f:
                if eval_init_input:
                    f.write('# Evaluations on ' + init_pars_out_name + ' for parameter variations of ' +
                            it_pars_cols_name + '\n')
                else:
                    f.write('# Evaluations for parameter variations of ' +
                            it_pars_cols_name + '\n')
                f.write('# Used parameter values: ' + grp_it + '\n')
                f.write('# Used data part of ' + '-'.join(map(str, partitions)) + ': ' + grp_name + '\n')
                f.write('# Column parameters: ' + ', '.join(eval_cols_lname) + '\n')
                tmp1.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

            for i, it in enumerate(eval_columns):
                # Construct tex-file information
                stats_all = {'min': tmp1[it].min(), 'max': tmp1[it].max()}
                if (np.isclose(stats_all['min'], 0, atol=1e-06) and
                    np.isclose(stats_all['max'], 0, atol=1e-06)) or \
                        np.isclose(stats_all['min'], stats_all['max']):
                    continue
                use_limits = {'minz': None, 'maxz': None}
                if np.abs(stats_all['max'] - stats_all['min']) < np.abs(stats_all['max'] / 200):
                    if stats_all['min'] < 0:
                        use_limits['minz'] = round(1.01 * stats_all['min'], 6)
                    else:
                        use_limits['minz'] = round(0.99 * stats_all['min'], 6)
                    if stats_all['max'] < 0:
                        use_limits['maxz'] = round(0.99 * stats_all['max'], 6)
                    else:
                        use_limits['maxz'] = round(1.01 * stats_all['max'], 6)
                reltex_name = os.path.join(rel_data_path, dataf_name)
                if eval_init_input:
                    fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' based on ' + strToLower(init_pars_title)
                else:
                    fig_name = capitalizeFirstChar(eval_cols_lname[i])
                fig_name += ' for '
                fig_name += partition_text_val1 + ' in addition to ' + \
                            ' parameters ' + tex_string_coding_style(grp_it) + \
                            ' compared to ' + \
                            replaceCSVLabels(xy_axis_columns[0], True) + ' and ' + \
                            replaceCSVLabels(xy_axis_columns[1], True)
                tex_infos['sections'].append({'file': reltex_name,
                                              'name': fig_name,
                                              'fig_type': fig_type,
                                              'stat_name': it,
                                              'plots_z': [it],
                                              'diff_z_labels': False,
                                              'label_z': eval_cols_lname[i] + findUnit(str(it), units),
                                              'plot_x': str(xy_axis_columns[0]),
                                              'label_x': replaceCSVLabels(str(xy_axis_columns[0])) +
                                                         findUnit(str(xy_axis_columns[0]), units),
                                              'plot_y': str(xy_axis_columns[1]),
                                              'label_y': replaceCSVLabels(str(xy_axis_columns[1])) +
                                                         findUnit(str(xy_axis_columns[1]), units),
                                              'legend': [eval_cols_lname[i] + ' for ' +
                                                         tex_string_coding_style(grp_it)],
                                              'use_marks': use_marks,
                                              'mesh_cols': nr_equal_ss,
                                              'use_log_z_axis': eval_cols_log_scaling[i],
                                              'limits': use_limits
                                              })

    pdfs_info = []
    max_figs_pdf = 50
    if tex_infos['ctrl_fig_size']:  # and not figs_externalize:
        max_figs_pdf = 30
    for i, st in enumerate(eval_columns):
        # Get list of results using the same statistic
        st_list = list(filter(lambda stat: stat['stat_name'] == st, tex_infos['sections']))
        if len(st_list) > max_figs_pdf:
            st_list2 = [{'figs': st_list[i:i + max_figs_pdf],
                         'pdf_nr': i1 + 1} for i1, i in enumerate(range(0, len(st_list), max_figs_pdf))]
        else:
            st_list2 = [{'figs': st_list, 'pdf_nr': 1}]
        for it in st_list2:
            if len(st_list2) == 1:
                title = tex_infos['title'] + ': ' + capitalizeStr(eval_cols_lname[i])
            else:
                title = tex_infos['title'] + ' -- Part ' + str(it['pdf_nr']) + \
                        ' for ' + capitalizeStr(eval_cols_lname[i])
            pdfs_info.append({'title': title,
                              'texf_name': base_out_name + '_' + str(st) + '_' + str(it['pdf_nr']),
                              'figs_externalize': figs_externalize,
                              'sections': it['figs'],
                              'make_index': tex_infos['make_index'],
                              'use_fixed_caption': tex_infos['use_fixed_caption'],
                              'ctrl_fig_size': tex_infos['ctrl_fig_size'],
                              'abbreviations': tex_infos['abbreviations']})

    template = ji_env.get_template('usac-testing_3D_plots.tex')
    res = 0
    for it in pdfs_info:
        rendered_tex = template.render(title=it['title'],
                                       make_index=it['make_index'],
                                       use_fixed_caption=it['use_fixed_caption'],
                                       ctrl_fig_size=it['ctrl_fig_size'],
                                       figs_externalize=it['figs_externalize'],
                                       sections=it['sections'],
                                       abbreviations=it['abbreviations'])
        texf_name = it['texf_name'] + '.tex'
        if build_pdf:
            pdf_name = it['texf_name'] + '.pdf'
            res += compile_tex(rendered_tex,
                               tex_folder,
                               texf_name,
                               make_fig_index,
                               os.path.join(pdf_folder, pdf_name),
                               it['figs_externalize'])
        else:
            res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)

    return res


def calcFromFuncAndPlot_aggregate(data,
                                  store_path,
                                  tex_file_pre_str,
                                  fig_title_pre_str,
                                  eval_description_path,
                                  eval_columns,
                                  units,
                                  it_parameters,
                                  x_axis_column,
                                  filter_func=None,
                                  filter_func_args=None,
                                  special_calcs_func = None,
                                  special_calcs_args = None,
                                  calc_func = None,
                                  calc_func_args = None,
                                  compare_source = None,
                                  fig_type='smooth',
                                  use_marks=True,
                                  ctrl_fig_size=True,
                                  make_fig_index=True,
                                  build_pdf=False,
                                  figs_externalize=False):
    fig_types = ['sharp plot', 'smooth', 'const plot', 'ybar', 'xbar']
    if not fig_type in fig_types:
        raise ValueError('Unknown figure type.')
    #Filter rows by excluding not successful estimations
    data = data.loc[~((data['R_out(0,0)'] == 0) &
                      (data['R_out(0,1)'] == 0) &
                      (data['R_out(0,2)'] == 0) &
                      (data['R_out(1,0)'] == 0) &
                      (data['R_out(1,1)'] == 0) &
                      (data['R_out(1,2)'] == 0) &
                      (data['R_out(2,0)'] == 0) &
                      (data['R_out(2,1)'] == 0) &
                      (data['R_out(2,2)'] == 0))]
    if filter_func is not None:
        if filter_func_args is None:
            filter_func_args = {'data': data}
        else:
            filter_func_args['data'] = data
        data = filter_func(**filter_func_args)
    if data.empty:
        raise ValueError('No data left after filtering')

    # Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            calc_func_args = {'data': data}
        else:
            calc_func_args['data'] = data
        calc_func_args['eval_columns'] = eval_columns
        calc_func_args['it_parameters'] = it_parameters
        calc_func_args['x_axis_column'] = x_axis_column
        calc_func_args['units'] = units
        ret = calc_func(**calc_func_args)
        df = ret['data']
        eval_columns = ret['eval_columns']
        eval_cols_lname = ret['eval_cols_lname']
        eval_cols_log_scaling = ret['eval_cols_log_scaling']
        eval_init_input = ret['eval_init_input']
        units = ret['units']
        it_parameters = ret['it_parameters']
    else:
        raise ValueError('No function for calculating results provided')

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + '-'.join(map(str, it_parameters)))
    cnt = 1
    store_path_init = store_path_sub
    while os.path.exists(store_path_sub):
        store_path_sub = store_path_init + '_' + str(int(cnt))
        cnt += 1
    try:
        os.mkdir(store_path_sub)
    except FileExistsError:
        print('Folder', store_path_sub, 'for storing statistics data already exists')
    except:
        print("Unexpected error (Unable to create directory for storing statistics data):", sys.exc_info()[0])
        raise

    if build_pdf:
        pdf_folder = os.path.join(store_path_sub, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            print('Folder', pdf_folder, 'for storing pdf files already exists')
    tex_folder = os.path.join(store_path_sub, 'tex')
    try:
        os.mkdir(tex_folder)
    except FileExistsError:
        print('Folder', tex_folder, 'for storing tex files already exists')
    tdata_folder = os.path.join(tex_folder, 'data')
    try:
        os.mkdir(tdata_folder)
    except FileExistsError:
        print('Folder', tdata_folder, 'for storing data files already exists')

    if compare_source:
        compare_source['full_path'] = os.path.join(compare_source['store_path'],
                                                   compare_source['eval_description_path'] + '_' +
                                                   '-'.join(map(str, compare_source['it_parameters'])))
        if not os.path.exists(compare_source['full_path']):
            warnings.warn('Path ' + compare_source['full_path'] + ' for comparing results not found. '
                          'Skipping comparison.', UserWarning)
            compare_source = None
        if compare_source:
            compare_source['tdata_folder'] = os.path.join(compare_source['full_path'], 'tex')
            if compare_source and not os.path.exists(compare_source['tdata_folder']):
                warnings.warn('Tex folder ' + compare_source['tdata_folder'] + ' for comparing results not found. '
                              'Skipping comparison.', UserWarning)
                compare_source = None
        if compare_source:
            compare_source['tdata_folder'] = os.path.join(compare_source['tdata_folder'], 'data')
            if compare_source and not os.path.exists(compare_source['tdata_folder']):
                warnings.warn('Data folder ' + compare_source['tdata_folder'] + ' for comparing results not found. '
                              'Skipping comparison.', UserWarning)
                compare_source = None

    if special_calcs_func is not None and special_calcs_args is not None:
        if 'func_name' in special_calcs_args:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_args['func_name'])
        else:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_func.__name__)
        cnt = 1
        calc_vals = True
        special_path_init = special_path_sub
        while os.path.exists(special_path_sub):
            special_path_sub = special_path_init + '_' + str(int(cnt))
            cnt += 1
        try:
            os.mkdir(special_path_sub)
        except FileExistsError:
            print('Folder', special_path_sub, 'for storing statistics data already exists')
        except:
            print("Unexpected error (Unable to create directory for storing special function data):", sys.exc_info()[0])
            calc_vals = False
        if calc_vals:
            special_calcs_args['data'] = df
            special_calcs_args['eval_columns'] = eval_columns
            special_calcs_args['eval_cols_lname'] = eval_cols_lname
            special_calcs_args['units'] = units
            special_calcs_args['it_parameters'] = it_parameters
            special_calcs_args['res_folder'] = special_path_sub
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Calculation of specific results failed!', UserWarning)
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    title_it_pars = ''
    for i, val in enumerate(it_parameters):
        base_out_name += val
        title_it_pars += replaceCSVLabels(val, True, True)
        if(nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_it_pars += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_it_pars += ', '
            elif i < nr_it_parameters - 1:
                title_it_pars += ', and '
        if i < nr_it_parameters - 1:
            base_out_name += '-'
    title_name += title_it_pars
    init_pars_title = ''
    init_pars_out_name = ''
    nr_eval_init_input = len(eval_init_input)
    if nr_eval_init_input > 1:
        for i, val in enumerate(eval_init_input):
            init_pars_out_name += val
            init_pars_title += replaceCSVLabels(val, True, True)
            if(nr_eval_init_input <= 2):
                if i < nr_eval_init_input - 1:
                    init_pars_title += ' and '
            else:
                if i < nr_eval_init_input - 2:
                    init_pars_title += ', '
                elif i < nr_eval_init_input - 1:
                    init_pars_title += ', and '
            if i < nr_eval_init_input - 1:
                init_pars_out_name += '-'
    else:
        init_pars_out_name = eval_init_input[0]
        init_pars_title = replaceCSVLabels(eval_init_input[0], True, True)
    base_out_name += '_combs_based_on_' + init_pars_out_name
    title_name += ' Based On ' + init_pars_title
    df.set_index(it_parameters, inplace=True)
    if len(it_parameters) > 1:
        gloss = glossary_from_list([str(b) for a in df.index for b in a])
        it_pars_name = '-'.join(it_parameters)
        it_pars_index = ['-'.join(map(str, a)) for a in df.index]
        df.index = it_pars_index
        df.index.name = it_pars_name
    else:
        it_pars_index = [str(a) for a in df.index]
        gloss = glossary_from_list(it_pars_index)
        it_pars_name = it_parameters[0]
    if compare_source:
        add_to_glossary(compare_source['it_par_select'], gloss)
        gloss.append({'key': 'cmp', 'description': compare_source['cmp']})
    if gloss:
        gloss = add_to_glossary_eval(eval_columns, gloss)
    else:
        gloss = add_to_glossary_eval(eval_columns)
    from usac_eval import insert_opt_lbreak
    df['tex_it_pars'] = insert_opt_lbreak(it_pars_index)
    max_txt_rows = 1
    for idx, val in df['tex_it_pars'].iteritems():
        txt_rows = str(val).count('\\\\') + 1
        if txt_rows > max_txt_rows:
            max_txt_rows = txt_rows
    tex_infos = {'title': title_name,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': make_fig_index,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': ctrl_fig_size,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': figs_externalize,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations from a list of dicts
                 'abbreviations': gloss
                 }
    if compare_source:
        cmp_col_name = '-'.join(compare_source['it_parameters'])
        comp_fname = 'data_evals_' + init_pars_out_name + '_for_pars_' + \
                     cmp_col_name + '.csv'
        df, succ = add_comparison_column(compare_source, comp_fname, df, None, cmp_col_name)
    dataf_name = 'data_evals_' + init_pars_out_name + '_for_pars_' + \
                 it_pars_name + '.csv'
    fdataf_name = os.path.join(tdata_folder, dataf_name)
    with open(fdataf_name, 'a') as f:
        f.write('# Evaluations on ' + init_pars_out_name + ' for parameter variations of ' +
                it_pars_name + '\n')
        f.write('# Column parameters: ' + ', '.join(eval_cols_lname) + '\n')
        df.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')
    for i, it in enumerate(eval_columns):
        # Construct tex-file information
        stats_all = df[it].describe().T
        if (np.isclose(stats_all['min'], 0, atol=1e-06) and
            np.isclose(stats_all['max'], 0, atol=1e-06)) or \
                np.isclose(stats_all['min'], stats_all['max']):
            continue
        use_limits = {'miny': None, 'maxy': None}
        if np.abs(stats_all['max'] - stats_all['min']) < np.abs(stats_all['max'] / 200):
            if stats_all['min'] < 0:
                use_limits['miny'] = round(1.01 * stats_all['min'], 6)
            else:
                use_limits['miny'] = round(0.99 * stats_all['min'], 6)
            if stats_all['max'] < 0:
                use_limits['maxy'] = round(0.99 * stats_all['max'], 6)
            else:
                use_limits['maxy'] = round(1.01 * stats_all['max'], 6)
        else:
            if stats_all['min'] < (stats_all['mean'] - stats_all['std'] * 3.291):
                use_limits['miny'] = round(stats_all['mean'] - stats_all['std'] * 3.291, 6)
            if stats_all['max'] > (stats_all['mean'] + stats_all['std'] * 3.291):
                use_limits['maxy'] = round(stats_all['mean'] + stats_all['std'] * 3.291, 6)
        if use_limits['miny'] and use_limits['maxy']:
            exp_value = is_exp_used(use_limits['miny'], use_limits['maxy'], eval_cols_log_scaling[i])
        elif use_limits['miny']:
            exp_value = is_exp_used(use_limits['miny'], stats_all['max'], eval_cols_log_scaling[i])
        elif use_limits['maxy']:
            exp_value = is_exp_used(stats_all['min'], use_limits['maxy'], eval_cols_log_scaling[i])
        else:
            exp_value = is_exp_used(stats_all['min'], stats_all['max'], eval_cols_log_scaling[i])
        reltex_name = os.path.join(rel_data_path, dataf_name)
        fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' based on ' + strToLower(init_pars_title) + \
                   ' for parameter variations of\\\\' + strToLower(title_it_pars)
        fig_name = split_large_titles(fig_name)
        if exp_value and len(fig_name.split('\\\\')[-1]) < 70:
            exp_value = False
        tex_infos['sections'].append({'file': reltex_name,
                                      'name': fig_name.replace('\\\\', ' '),
                                      'title': fig_name,
                                      'title_rows': fig_name.count('\\\\'),
                                      'fig_type': fig_type,
                                      'plots': [it],
                                      'label_y': eval_cols_lname[i] + findUnit(str(it), units),
                                      # Label of the value axis. For xbar it labels the x-axis
                                      # Label/column name of axis with bars. For xbar it labels the y-axis
                                      'label_x': 'Parameter',
                                      # Column name of axis with bars. For xbar it is the column for the y-axis
                                      'print_x': 'tex_it_pars',
                                      # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                      'print_meta': False,
                                      'plot_meta': None,
                                      # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                      'rotate_meta': 0,
                                      'limits': use_limits,
                                      # If None, no legend is used, otherwise use a list
                                      'legend': None,
                                      'legend_cols': 1,
                                      'use_marks': use_marks,
                                      # The x/y-axis values are given as strings if True
                                      'use_string_labels': True,
                                      'use_log_y_axis': eval_cols_log_scaling[i],
                                      'xaxis_txt_rows': max_txt_rows,
                                      'enlarge_title_space': exp_value,
                                      'large_meta_space_needed': False,
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
    texf_name = base_out_name + '.tex'
    if build_pdf:
        pdf_name = base_out_name + '.pdf'
        res = abs(compile_tex(rendered_tex,
                              tex_folder,
                              texf_name,
                              tex_infos['make_index'],
                              os.path.join(pdf_folder, pdf_name),
                              tex_infos['figs_externalize']))
    else:
        res = abs(compile_tex(rendered_tex, tex_folder, texf_name, False))

    return res


def calcSatisticAndPlot_aggregate(data,
                                  store_path,
                                  tex_file_pre_str,
                                  fig_title_pre_str,
                                  eval_description_path,
                                  eval_columns,
                                  units,
                                  it_parameters,
                                  pdfsplitentry,  # One or more column names present in eval_columns for splitting pdf
                                  filter_func=None,
                                  filter_func_args=None,
                                  special_calcs_func=None,
                                  special_calcs_args=None,
                                  calc_func=None,
                                  calc_func_args=None,
                                  compare_source = None,
                                  fig_type='smooth',
                                  use_marks=True,
                                  ctrl_fig_size=True,
                                  make_fig_index=True,
                                  build_pdf=False,
                                  figs_externalize=True):
    fig_types = ['sharp plot', 'smooth', 'const plot', 'ybar', 'xbar']
    if not fig_type in fig_types:
        raise ValueError('Unknown figure type.')
    # if type(data) is not pd.dataframe.DataFrame:
    #     data = pd.utils.from_pandas(data)
    # Filter rows by excluding not successful estimations
    data = data.loc[~((data['R_out(0,0)'] == 0) &
                      (data['R_out(0,1)'] == 0) &
                      (data['R_out(0,2)'] == 0) &
                      (data['R_out(1,0)'] == 0) &
                      (data['R_out(1,1)'] == 0) &
                      (data['R_out(1,2)'] == 0) &
                      (data['R_out(2,0)'] == 0) &
                      (data['R_out(2,1)'] == 0) &
                      (data['R_out(2,2)'] == 0))]
    if filter_func is not None:
        if filter_func_args is None:
            filter_func_args = {'data': data}
        else:
            filter_func_args['data'] = data
        data = filter_func(**filter_func_args)
    if data.empty:
        raise ValueError('No data left after filtering')

    # Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            calc_func_args = {'data': data}
        else:
            calc_func_args['data'] = data
        calc_func_args['eval_columns'] = eval_columns
        calc_func_args['it_parameters'] = it_parameters
        ret = calc_func(**calc_func_args)
        df = ret['data']
        eval_columns = ret['eval_columns']
        it_parameters = ret['it_parameters']
    else:
        needed_columns = eval_columns + it_parameters
        df = data[needed_columns]

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + '-'.join(map(str, it_parameters)))
    cnt = 1
    store_path_init = store_path_sub
    while os.path.exists(store_path_sub):
        store_path_sub = store_path_init + '_' + str(int(cnt))
        cnt += 1
    try:
        os.mkdir(store_path_sub)
    except FileExistsError:
        print('Folder', store_path_sub, 'for storing statistics data already exists')
    except:
        print("Unexpected error (Unable to create directory for storing statistics data):", sys.exc_info()[0])
        raise
    if build_pdf:
        pdf_folder = os.path.join(store_path_sub, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            print('Folder', pdf_folder, 'for storing pdf files already exists')
    tex_folder = os.path.join(store_path_sub, 'tex')
    try:
        os.mkdir(tex_folder)
    except FileExistsError:
        print('Folder', tex_folder, 'for storing tex files already exists')
    tdata_folder = os.path.join(tex_folder, 'data')
    try:
        os.mkdir(tdata_folder)
    except FileExistsError:
        print('Folder', tdata_folder, 'for storing data files already exists')

    if compare_source:
        compare_source['full_path'] = os.path.join(compare_source['store_path'],
                                                   compare_source['eval_description_path'] + '_' +
                                                   '-'.join(map(str, compare_source['it_parameters'])))
        if not os.path.exists(compare_source['full_path']):
            warnings.warn('Path ' + compare_source['full_path'] + ' for comparing results not found. '
                          'Skipping comparison.', UserWarning)
            compare_source = None
        if compare_source:
            compare_source['tdata_folder'] = os.path.join(compare_source['full_path'], 'tex')
            if compare_source and not os.path.exists(compare_source['tdata_folder']):
                warnings.warn('Tex folder ' + compare_source['tdata_folder'] + ' for comparing results not found. '
                              'Skipping comparison.', UserWarning)
                compare_source = None
        if compare_source:
            compare_source['tdata_folder'] = os.path.join(compare_source['tdata_folder'], 'data')
            if compare_source and not os.path.exists(compare_source['tdata_folder']):
                warnings.warn('Data folder ' + compare_source['tdata_folder'] + ' for comparing results not found. '
                              'Skipping comparison.', UserWarning)
                compare_source = None

    # Group by USAC parameters 5&6 and calculate the statistic
    stats = df.groupby(it_parameters).describe()
    if special_calcs_func is not None and special_calcs_args is not None:
        if 'func_name' in special_calcs_args:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_args['func_name'])
        else:
            special_path_sub = os.path.join(store_path, 'evals_function_' + special_calcs_func.__name__)
        cnt = 1
        calc_vals = True
        special_path_init = special_path_sub
        while os.path.exists(special_path_sub):
            special_path_sub = special_path_init + '_' + str(int(cnt))
            cnt += 1
        if 'mk_no_folder' not in special_calcs_args or not special_calcs_args['mk_no_folder']:
            try:
                os.mkdir(special_path_sub)
            except FileExistsError:
                print('Folder', special_path_sub, 'for storing statistics data already exists')
            except:
                print("Unexpected error "
                      "(Unable to create directory for storing special function data):", sys.exc_info()[0])
                calc_vals = False
        if calc_vals:
            special_calcs_args['data'] = stats
            special_calcs_args['it_parameters'] = it_parameters
            special_calcs_args['eval_columns'] = eval_columns
            special_calcs_args['res_folder'] = special_path_sub
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Errors occured during calculation of specific results!', UserWarning)
    errvalnames = stats.columns.values  # Includes statistic name and error value names
    grp_names = stats.index.names  # As used when generating the groups
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    title_it_pars = ''
    for i in range(0, nr_it_parameters):
        base_out_name += grp_names[i]
        title_it_pars += replaceCSVLabels(grp_names[i], True, True)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_it_pars += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_it_pars += ', '
            elif i < nr_it_parameters - 1:
                title_it_pars += ', and '
        if i < nr_it_parameters - 1:
            base_out_name += '-'
    title_name += title_it_pars
    base_out_name += '_combs_aggregated'
    # holds the grouped names/entries within the group names excluding the last entry th
    # grp_values = list(dict.fromkeys([i[0:2] for i in stats.index.values]))
    tex_infos = {'title': title_name,
                 'sections': [],
                 # Builds an index with hyperrefs on the beginning of the pdf
                 'make_index': make_fig_index,
                 # If True, the figures are adapted to the page height if they are too big
                 'ctrl_fig_size': ctrl_fig_size,
                 # If true, a pdf is generated for every figure and inserted as image in a second run
                 'figs_externalize': figs_externalize,
                 # If true and a bar chart is chosen, the bars a filled with color and markers are turned off
                 'fill_bar': True,
                 # Builds a list of abbrevations of a list of dicts
                 'abbreviations': None}
    pdf_nr = 0
    gloss_calced = False
    from usac_eval import insert_opt_lbreak
    for it in errvalnames:
        if it[-1] != 'count':
            tmp = stats[it[0]]
            tmp = tmp.loc[:, [it[-1]]]
            col_name = replace_stat_names_col_tex(it[-1])
            tmp.rename(columns={it[-1]: col_name}, inplace=True)
            # tmp.columns = ['%s%s' % (str(a), '-%s' % str(b) if b is not None else '') for a, b in tmp.columns]
            if not gloss_calced:
                if len(it_parameters) > 1:
                    gloss = glossary_from_list([str(b) for a in tmp.index for b in a])
                else:
                    gloss = glossary_from_list([str(a) for a in tmp.index])
                gloss_calced = True
                if compare_source:
                    add_to_glossary(compare_source['it_par_select'], gloss)
                    gloss.append({'key': 'cmp', 'description': compare_source['cmp']})
                if gloss:
                    gloss = add_to_glossary_eval(eval_columns, gloss)
                    tex_infos['abbreviations'] = gloss
                else:
                    gloss = add_to_glossary_eval(eval_columns)
                    if gloss:
                        tex_infos['abbreviations'] = gloss
            if len(it_parameters) > 1:
                it_pars_index = ['-'.join(map(str, a)) for a in tmp.index]
                tmp.index = it_pars_index
                index_name = '-'.join(it_parameters)
                tmp.index.name = index_name
            else:
                it_pars_index = [str(a) for a in tmp.index]
                index_name = it_parameters[0]
            tmp['tex_it_pars'] = insert_opt_lbreak(it_pars_index)
            max_txt_rows = 1
            for idx, val in tmp['tex_it_pars'].iteritems():
                txt_rows = str(val).count('\\\\') + 1
                if txt_rows > max_txt_rows:
                    max_txt_rows = txt_rows
            dataf_name = 'data_' + '_'.join(map(str, it)) + '.csv'
            dataf_name = dataf_name.replace('%', 'perc')
            if compare_source:
                cmp_col_name = '-'.join(compare_source['it_parameters'])
                df, succ = add_comparison_column(compare_source, dataf_name, df, None, cmp_col_name)
            fdataf_name = os.path.join(tdata_folder, dataf_name)
            with open(fdataf_name, 'a') as f:
                f.write('# ' + str(it[-1]) + ' values for ' + str(it[0]) + '\n')
                f.write('# Parameters: ' + '-'.join(it_parameters) + '\n')
                tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

            # Construct tex-file
            if pdfsplitentry:
                if pdf_nr < len(pdfsplitentry):
                    if pdfsplitentry[pdf_nr] == str(it[0]):
                        pdf_nr += 1
            stats_all = tmp.drop('tex_it_pars', axis=1).stack().reset_index()
            stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
            if (np.isclose(stats_all['min'][0], 0, atol=1e-06) and
                np.isclose(stats_all['max'][0], 0, atol=1e-06)) or \
                    np.isclose(stats_all['min'][0], stats_all['max'][0]):
                continue
            # figure types: sharp plot, smooth, const plot, ybar, xbar
            use_limits = {'miny': None, 'maxy': None}
            if np.abs(stats_all['max'][0] - stats_all['min'][0]) < np.abs(stats_all['max'][0] / 200):
                if stats_all['min'][0] < 0:
                    use_limits['miny'] = round(1.01 * stats_all['min'][0], 6)
                else:
                    use_limits['miny'] = round(0.99 * stats_all['min'][0], 6)
                if stats_all['max'][0] < 0:
                    use_limits['maxy'] = round(0.99 * stats_all['max'][0], 6)
                else:
                    use_limits['maxy'] = round(1.01 * stats_all['max'][0], 6)
            use_log = True if np.abs(np.log10(np.abs(stats_all['min'][0])) -
                                     np.log10(np.abs(stats_all['max'][0]))) > 1 else False
            exp_value = is_exp_used(stats_all['min'][0], stats_all['max'][0], use_log)

            fig_name = replace_stat_names(it[-1]) + ' values for ' +\
                       replaceCSVLabels(str(it[0]), True) + ' comparing parameter variations of\\\\' + \
                       strToLower(title_it_pars)
            fig_name = split_large_titles(fig_name)
            if exp_value and len(fig_name.split('\\\\')[-1]) < 70:
                exp_value = False
            reltex_name = os.path.join(rel_data_path, dataf_name)
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': fig_name.replace('\\\\', ' '),
                                          'title': fig_name,
                                          'title_rows': fig_name.count('\\\\'),
                                          'fig_type': fig_type,
                                          'plots': [col_name],
                                          'label_y': replace_stat_names(it[-1]) + findUnit(str(it[0]), units),
                                          # Label of the value axis. For xbar it labels the x-axis
                                          # Label/column name of axis with bars. For xbar it labels the y-axis
                                          'label_x': 'Parameter',
                                          # Column name of axis with bars. For xbar it is the column for the y-axis
                                          'print_x': 'tex_it_pars',
                                          # Set print_meta to True if values from column plot_meta should be printed next to each bar
                                          'print_meta': False,
                                          'plot_meta': None,
                                          # A value in degrees can be specified to rotate the text (Use only 0, 45, and 90)
                                          'rotate_meta': 0,
                                          'limits': use_limits,
                                          # If None, no legend is used, otherwise use a list
                                          'legend': None,
                                          'legend_cols': 1,
                                          'use_marks': use_marks,
                                          # The x/y-axis values are given as strings if True
                                          'use_string_labels': True,
                                          'use_log_y_axis': use_log,
                                          'xaxis_txt_rows': max_txt_rows,
                                          'enlarge_title_space': exp_value,
                                          'large_meta_space_needed': False,
                                          'caption': fig_name.replace('\\\\', ' '),
                                          'pdf_nr': pdf_nr
                                          })

    template = ji_env.get_template('usac-testing_2D_bar_chart_and_meta.tex')
    # Get number of pdfs to generate
    pdf_nr = tex_infos['sections'][-1]['pdf_nr']
    res = 0
    if pdf_nr == 0:
        rendered_tex = template.render(title=tex_infos['title'],
                                       make_index=tex_infos['make_index'],
                                       ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                       figs_externalize=tex_infos['figs_externalize'],
                                       fill_bar=tex_infos['fill_bar'],
                                       sections=tex_infos['sections'],
                                       abbreviations=tex_infos['abbreviations'])
        texf_name = base_out_name + '.tex'
        if build_pdf:
            pdf_name = base_out_name + '.pdf'
            res += compile_tex(rendered_tex,
                               tex_folder,
                               texf_name,
                               make_fig_index,
                               os.path.join(pdf_folder, pdf_name),
                               tex_infos['figs_externalize'])
        else:
            res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)
    else:
        sections = []
        diff_pdfs = []
        tmp_nr = 0
        for it in tex_infos['sections']:
            if it['pdf_nr'] == tmp_nr:
                sections.append(it)
            else:
                diff_pdfs.append(deepcopy(sections))
                sections = [it]
                tmp_nr += 1
        diff_pdfs.append(sections)
        for it in diff_pdfs:
            rendered_tex = template.render(title=tex_infos['title'],
                                           make_index=tex_infos['make_index'],
                                           ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                           figs_externalize=tex_infos['figs_externalize'],
                                           fill_bar=tex_infos['fill_bar'],
                                           sections=it,
                                           abbreviations=tex_infos['abbreviations'])
            texf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.tex'
            if build_pdf:
                pdf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.pdf'
                res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index,
                                   os.path.join(pdf_folder, pdf_name), tex_infos['figs_externalize'])
            else:
                res += compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)
    return res


def add_comparison_column(compare_source, comp_fname, data_new, mult_cols=None, it_col_name=None):
    comp_col_name = '-'.join(map(str, compare_source['it_par_select']))
    comp_fdataf_name = os.path.join(compare_source['tdata_folder'], comp_fname)
    succ = True
    if not os.path.isfile(comp_fdataf_name):
        warnings.warn('CSV file ' + comp_fdataf_name + ' for comparing results not found. '
                                                       'Skipping comparison.', UserWarning)
        succ = False
    else:
        comp_data = pd.read_csv(comp_fdataf_name, delimiter=';', comment='#')
        if not mult_cols and not it_col_name:
            if comp_col_name not in comp_data.columns:
                warnings.warn('Column ' + comp_col_name + ' not found in csv file ' + comp_fdataf_name +
                              ' for comparing results not found. Skipping comparison.', UserWarning)
                succ = False
            else:
                comp_col = comp_data[comp_col_name]
                if comp_col.shape[0] != data_new.shape[0]:
                    warnings.warn(' Number of rows in column ' + comp_col_name +
                                  ' of csv file ' + comp_fdataf_name +
                                  ' for comparing results does not match. Skipping comparison.', UserWarning)
                    succ = False
                else:
                    data_new['cmp-' + comp_col_name] = comp_col.values
        elif not it_col_name:
            data_tmp = data_new.copy(deep=True)
            for col in mult_cols:
                if col not in comp_data.columns:
                    warnings.warn('Column ' + col + ' not found in csv file ' + comp_fdataf_name +
                                  ' for comparing results not found. Skipping comparison.', UserWarning)
                    succ = False
                    break
                else:
                    comp_col = comp_data[col]
                    if comp_col.shape[0] != data_tmp.shape[0]:
                        warnings.warn(' Number of rows in column ' + comp_col_name +
                                      ' of csv file ' + comp_fdataf_name +
                                      ' for comparing results does not match. Skipping comparison.', UserWarning)
                        succ = False
                        break
                    else:
                        data_tmp['cmp-' + col] = comp_col.values
            if succ:
                data_new = data_tmp
        else:
            if it_col_name not in comp_data.columns:
                warnings.warn('Column ' + it_col_name + ' not found in csv file ' + comp_fdataf_name +
                              ' for comparing results not found. Skipping comparison.', UserWarning)
                succ = False
            else:
                comp_data.set_index(it_col_name, inplace=True)
                for col1, col2 in zip(data_new.columns, comp_data.columns):
                    if col1 != col2:
                        succ = False
                        break
                if succ:
                    if comp_col_name not in comp_data.index:
                        warnings.warn('Parameter ' + comp_col_name + ' not found in csv file ' + comp_fdataf_name +
                                      ' for comparing results not found. Skipping comparison.', UserWarning)
                        succ = False
                    else:
                        line = comp_data.loc[comp_col_name, :]
                        line.name = 'cmp-' + line.name
                        if 'tex_it_pars' in line.index:
                            line['tex_it_pars'] = 'cmp-' + str(line['tex_it_pars'])
                        data_new = data_new.append(line, ignore_index=False)
    return data_new, succ


def replace_stat_names_col_tex(name):
    if name == r'25%':
        return '25percentile'
    elif name == '50%':
        return 'median'
    elif name == r'75%':
        return '75percentile'
    else:
        return str(name).replace('%', 'perc')

def replace_stat_names(name, for_tex=True):
    if name == 'max':
        return 'Maximum'
    elif name == 'min':
        return 'Minimum'
    elif name == 'mean':
        return name.capitalize()
    elif name == 'std':
        return 'Standard deviation'
    elif name == r'25%':
        if for_tex:
            return r'25\% percentile'
        else:
            return r'25perc percentile'
    elif name == '50%':
        return 'Median'
    elif name == r'75%':
        if for_tex:
            return r'75\% percentile'
        else:
            return r'75perc percentile'
    else:
        if for_tex:
            return str(name).replace('%', '\%').capitalize()
        else:
            return str(name).replace('%', 'perc').capitalize()


def is_exp_used(min_val, max_val, use_log=False):
    if use_log:
        return False
    m_val = min(float(np.log10(np.abs(min_val))), float(np.log10(np.abs(max_val))))
    m_val2 = max(float(np.log10(np.abs(min_val))), float(np.log10(np.abs(max_val))))
    if m_val < 0 and abs(m_val) > 1.01:
        return True
    elif m_val2 >= 4:
        return True
    return False


def split_large_titles(title_str):
    if '\\\\' in title_str:
        substr = title_str.split('\\\\')
        substr1 = []
        for s in substr:
            substr1.append(split_large_str(s))
        return '\\\\'.join(substr1)
    else:
        return split_large_str(title_str)


def split_large_str(large_str):
    ls = len(large_str)
    if ls <= 100:
        return large_str
    else:
        nr_splits = int(ls / 100)
        ls_part = round(ls / float(nr_splits + 1), 0)
        ls_sum = ls_part
        it_nr = 0
        s1 = large_str.split(' ')
        s1_tmp = []
        l1 = 0
        for s2 in s1:
            l1 += len(s2)
            if l1 < ls_sum:
                s1_tmp.append(s2)
            elif l1 >= ls_sum and it_nr < nr_splits:
                it_nr += 1
                ls_sum += ls_part
                if s1_tmp:
                    if '$' in s1_tmp[-1] or\
                       '\\' in s1_tmp[-1] or\
                       len(s1_tmp[-1]) == 1 or\
                       '}' in s1_tmp[-1] or\
                       '{' in s1_tmp[-1] or\
                       '_' in s1_tmp[-1]:
                        it_nr -= 1
                        ls_sum -= ls_part
                        s1_tmp.append(s2)
                    else:
                        s1_tmp[-1] += '\\\\' + s2
                else:
                    s1_tmp.append(s2)
            else:
                s1_tmp.append(s2)
        return ' '.join(s1_tmp)


def tex_string_coding_style(text):
    text = text.replace('_', '\\_')
    return '\\texttt{' + text + '}'

def findUnit(key, units):
    for i in units:
        if key in i:
            return i[1]
    return ''

def calcNrLegendCols(tex_infos_section):
    nr_plots = len(tex_infos_section['plots'])
    max_cols = int(235 / (len(max(tex_infos_section['plots'], key=len)) * 3 + 16))
    if max_cols < 1:
        max_cols = 1
    if max_cols > 10:
        max_cols = 10
    use_cols = max_cols
    rem = float(nr_plots) / float(use_cols) - math.floor(float(nr_plots) / float(use_cols))
    while rem < 0.5 and not np.isclose(rem, 0) and use_cols > 1:
        use_cols -= 1
        rem = float(nr_plots) / float(use_cols) - math.floor(float(nr_plots) / float(use_cols))
    if use_cols == 1:
        use_cols = max_cols
    return use_cols


def add_to_glossary_eval(entry_ies, gloss=None):
    entries = None
    if isinstance(entry_ies, str):
        entries = [entry_ies]
    else:
        try:
            oit = iter(entry_ies)
            entries = entry_ies
        except TypeError as te:
            entries = [entry_ies]
    mylist = list(dict.fromkeys(entries))
    for elem in mylist:
        elem_tex = replaceCSVLabels(str(elem))
        if gloss:
            found = False
            for entry in gloss:
                if entry['key'] == elem_tex:
                    found = True
                    break
            if found:
                continue
        else:
            gloss = []
        ret = getSymbolDescription(str(elem))
        if ret[2]:
            gloss.append({'key': ret[0], 'description': ret[1]})
    return gloss


def getSymbolDescription(label):
    if label == 'R_diffAll':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth and estimated relative stereo camera rotation matrices '
                '(calculated using quaternion notations)', True)
    elif label == 'R_diff_roll_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth and estimated relative stereo camera rotation about the '
                'x-axis', True)
    elif label == 'R_diff_pitch_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth and estimated relative stereo camera rotation about the '
                'y-axis', True)
    elif label == 'R_diff_yaw_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth and estimated relative stereo camera rotation about the '
                'z-axis', True)
    elif label == 't_angDiff_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth and estimated relative stereo camera translation vectors',
                True)
    elif label == 't_distDiff':
        return (replaceCSVLabels(label),
                'L2-norm on the difference between ground truth and estimated relative stereo camera '
                'translation vectors', True)
    elif label == 't_diff_tx':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{x}=\\tilde{t}_{x}/\\lvert\\tilde{\\bm{t}}\\rvert '
                '-t_{x}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized x-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$', True)
    elif label == 't_diff_ty':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{y}=\\tilde{t}_{y}/\\lvert\\tilde{\\bm{t}}\\rvert '
                '-t_{y}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized y-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$', True)
    elif label == 't_diff_tz':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{z}=\\tilde{t}_{z}/\\lvert\\tilde{\\bm{t}}\\rvert '
                '-t_{z}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized z-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$', True)
    if label == 'R_diffAll_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of angular differences between ground truth and estimated relative '
                'stereo camera rotation matrices (calculated using quaternion notations)', True)
    elif label == 'R_diff_roll_deg_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of angular differences between ground truth and estimated '
                'relative stereo camera rotation about the x-axis', True)
    elif label == 'R_diff_pitch_deg_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of angular differences between ground truth and estimated '
                'relative stereo camera rotation about the y-axis', True)
    elif label == 'R_diff_yaw_deg_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of angular differences between ground truth and estimated '
                'relative stereo camera rotation about the z-axis', True)
    elif label == 't_angDiff_deg_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of angular differences between ground truth and estimated '
                'relative stereo camera translation vectors',
                True)
    elif label == 't_distDiff_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of L2-norms on the difference between ground truth and estimated '
                'relative stereo camera translation vectors', True)
    elif label == 't_diff_tx_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of differences '
                '$\\Delta t_{x}=\\tilde{t}_{x}/\\lvert\\tilde{\\bm{t}}\\rvert -t_{x}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized x-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$', True)
    elif label == 't_diff_ty_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of differences '
                '$\\Delta t_{y}=\\tilde{t}_{y}/\\lvert\\tilde{\\bm{t}}\\rvert -t_{y}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized y-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$', True)
    elif label == 't_diff_tz_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of differences '
                '$\\Delta t_{z}=\\tilde{t}_{z}/\\lvert\\tilde{\\bm{t}}\\rvert -t_{z}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized z-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$', True)
    elif label == 'th':
        return (replaceCSVLabels(label), 'Threshold on point correspondences in pixels', True)
    elif label == 'R_mostLikely_diffAll':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth and estimated relative stereo camera rotation matrices. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few estimated '
                'rotation matrices over the last pose estimations (calculated using quaternion notations).', True)
    elif label == 'R_mostLikely_diff_roll_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth and estimated relative stereo camera rotation about the '
                'x-axis. The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True)
    elif label == 'R_mostLikely_diff_pitch_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth and estimated relative stereo camera rotation about the '
                'y-axis. The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True)
    elif label == 'R_mostLikely_diff_yaw_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth and estimated relative stereo camera rotation about the '
                'z-axis. The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True)
    elif label == 't_mostLikely_angDiff_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth and estimated relative stereo camera translation vectors. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_distDiff':
        return (replaceCSVLabels(label),
                'L2-norm on the difference between ground truth and estimated relative stereo camera '
                'translation vectors. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_diff_tx':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{x}=\\tilde{t}_{x}/\\lvert\\tilde{\\bm{t}}\\rvert '
                '-t_{x}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized x-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_diff_ty':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{y}=\\tilde{t}_{y}/\\lvert\\tilde{\\bm{t}}\\rvert '
                '-t_{y}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized y-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_diff_tz':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{z}=\\tilde{t}_{z}/\\lvert\\tilde{\\bm{t}}\\rvert '
                '-t_{z}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized z-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 'R_mostLikely_diffAll_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of angular differences between ground truth and estimated '
                'relative stereo camera rotation matrices. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few estimated '
                'rotation matrices over the last pose estimations (calculated using quaternion notations).', True)
    elif label == 'R_mostLikely_diff_roll_deg_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of angular differences between ground truth and estimated '
                'relative stereo camera rotation about the '
                'x-axis. The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True)
    elif label == 'R_mostLikely_diff_pitch_deg_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of angular differences between ground truth and estimated '
                'relative stereo camera rotation about the '
                'y-axis. The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True)
    elif label == 'R_mostLikely_diff_yaw_deg_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of angular differences between ground truth and estimated '
                'relative stereo camera rotation about the '
                'z-axis. The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True)
    elif label == 't_mostLikely_angDiff_deg_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of angular differences between ground truth and estimated '
                'relative stereo camera translation vectors. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_distDiff_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of L2-norms on the difference between ground truth and '
                'estimated relative stereo camera translation vectors. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_diff_tx_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of Differences '
                '$\\Delta t_{x}=\\tilde{t}_{x}/\\lvert\\tilde{\\bm{t}}\\rvert -t_{x}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized x-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_diff_ty_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of differences '
                '$\\Delta t_{y}=\\tilde{t}_{y}/\\lvert\\tilde{\\bm{t}}\\rvert -t_{y}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized y-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_diff_tz_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of differences '
                '$\\Delta t_{z}=\\tilde{t}_{z}/\\lvert\\tilde{\\bm{t}}\\rvert -t_{z}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized z-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z},\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z},\\right]^{T}$. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 'K1_fxDiff':
        return (replaceCSVLabels(label),
                'Difference between focal length x-components in pixels of ground truth and estimated camera matrices '
                'from the first/left/top camera', True)
    elif label == 'K1_fyDiff':
        return (replaceCSVLabels(label),
                'Difference between focal length y-components in pixels of ground truth and estimated camera matrices '
                'from the first/left/top camera', True)
    elif label == 'K1_fxyDiffNorm':
        return (replaceCSVLabels(label),
                'L2-norm of focal length x- and y-component differences in pixels of ground truth and estimated '
                'camera matrices from the first/left/top camera', True)
    elif label == 'K1_cxDiff':
        return (replaceCSVLabels(label),
                'Difference between principal point x-components in pixels of ground truth and estimated camera '
                'matrices from the first/left/top camera', True)
    elif label == 'K1_cyDiff':
        return (replaceCSVLabels(label),
                'Difference between principal point y-components in pixels of ground truth and estimated camera '
                'matrices from the first/left/top camera', True)
    elif label == 'K1_cxyDiffNorm':
        return (replaceCSVLabels(label),
                'L2-norm of principal point x- and y-component differences in pixels of ground truth and estimated '
                'camera matrices from the first/left/top camera', True)
    elif label == 'K1_cxyfxfyNorm':
        return (replaceCSVLabels(label),
                'L2-norm of principal point in addition to focal length x- and y-component '
                'differences in pixels of ground truth and estimated '
                'camera matrices from the first/left/top camera', True)
    elif label == 'K2_fxDiff':
        return (replaceCSVLabels(label),
                'Difference between focal length x-components in pixels of ground truth and estimated camera matrices '
                'from the second/right/bottom camera', True)
    elif label == 'K2_fyDiff':
        return (replaceCSVLabels(label),
                'Difference between focal length y-components in pixels of ground truth and estimated camera matrices '
                'from the second/right/bottom camera', True)
    elif label == 'K2_fxyDiffNorm':
        return (replaceCSVLabels(label),
                'L2-norm of focal length x- and y-component differences in pixels of ground truth and estimated '
                'camera matrices from the second/right/bottom camera', True)
    elif label == 'K2_cxDiff':
        return (replaceCSVLabels(label),
                'Difference between principal point x-components in pixels of ground truth and estimated camera '
                'matrices from the second/right/bottom camera', True)
    elif label == 'K2_cyDiff':
        return (replaceCSVLabels(label),
                'Difference between principal point y-components in pixels of ground truth and estimated camera '
                'matrices from the second/right/bottom camera', True)
    elif label == 'K2_cxyDiffNorm':
        return (replaceCSVLabels(label),
                'L2-norm of principal point x- and y-component differences in pixels of ground truth and estimated '
                'camera matrices from the second/right/bottom camera', True)
    elif label == 'K2_cxyfxfyNorm':
        return (replaceCSVLabels(label),
                'L2-norm of principal point in addition to focal length x- and y-component '
                'differences in pixels of ground truth and estimated '
                'camera matrices from the second/right/bottom camera', True)
    elif label == 'Rt_diff':
        return (replaceCSVLabels(label),
                'Combined rotation $R$ and translation $\\bm{t}$ error '
                '$e_{f\\left( R\\bm{t}\\right)_{i}}=\\left( e_{\\Sigma ,i}-'
                '\\text{min}\\left( \\bm{e}_{\\Sigma}\\right)\\right)/r_{R\\bm{t}}$ with '
                '$e_{\\Sigma, i}=e_{f\\left( R\\right)_{i}}+e_{f\\left( \\bm{t}\\right)_{i}}$, '
                '$r_{R\\bm{t}}=\\text{max}\\left( \\bm{e}_{\\Sigma}\\right)'
                '-\\text{min}\\left( \\bm{e}_{\\Sigma}\\right)$, '
                '$e_{f\\left( R\\right)_{i}}=\\lvert \\mu_{\\Delta R_{\\Sigma ,i}}\\rvert '
                '+\\sigma_{\\Delta R_{\\Sigma ,i}}$, '
                '$e_{f\\left( \\bm{t}\\right)_{i}}=\\lvert \\mu_{\\angle{\\Delta \\bm{t}_{i}}}\\rvert '
                '+\\sigma_{\\angle{\\Delta \\bm{t}_{i}}}$, '
                '$\\bm{e}_{\\Sigma}=\\left[ e_{\\Sigma, 1}, \\ldots ,e_{\\Sigma, n_{R\\bm{t}}}\\right]$, and '
                'number $n_{R\\bm{t}}$ of available $R$ and $\\bm{t}$ error statistics. '                
                # '$e_{f\\left( R\\bm{t}\\right) }=\\left( e_{f\\left( R\\right) }r_{\\bm{t}}+'
                # 'e_{f\\left( \\bm{t}\\right) }r_{R}\\right) /2r_{R}r_{\\bm{t}}$ with '
                # '$e_{f\\left( R\\right) }=\\lvert \\mu_{\\Delta R_{\\Sigma}}\\rvert +\\sigma_{\\Delta R_{\\Sigma}}$, '
                # '$e_{f\\left( \\bm{t}\\right) }=\\lvert \\mu_{\\angle{\\Delta \\bm{t}}}\\rvert '
                # '+\\sigma_{\\angle{\\Delta \\bm{t}}}$, '
                # '$r_{R}=\\text{max}\\left( e_{f\\left( R\\right) }\\right) '
                # '-\\text{min}\\left( e_{f\\left( R\\right) }\\right)$, and '
                # '$r_{\\bm{t}}=\\text{max}\\left( e_{f\\left( \\bm{t}\\right) }\\right) '
                # '-\\text{min}\\left( e_{f\\left( \\bm{t}\\right) }\\right)$. '
                '$\\mu_{\\Delta R_{\\Sigma}}$ and $\\mu_{\\angle{\\Delta \\bm{t}}}$ indicate the corresponding mean '
                'values of differential angles $\\Delta R_{\\Sigma}$ and $\\angle{\\Delta \\bm{t}}$. '
                '$\\sigma_{\\Delta R_{\\Sigma}}$ and $\\sigma_{\\angle{\\Delta \\bm{t}}}$ stand for the '
                'standard deviations of afore mentioned data.', True)
    elif label == 'Rt_mostLikely_diff':
        return (replaceCSVLabels(label),
                'Combined rotation $\\hat{R}$ and translation $\\hat{\\bm{t}}$ error '
                '$\\hat{e}_{f\\left( \\hat{R}\\hat{\\bm{t}}\\right)_{i}}='
                '\\left( \\hat{e}_{\\Sigma ,i}-\\text{min}\\left( \\hat{\\bm{e}}_{\\Sigma}\\right)\\right)'
                '/r_{\\hat{R}\\hat{\\bm{t}}}$ with '
                '$\\hat{e}_{\\Sigma ,i}=e_{f\\left( \\hat{R}\\right)_{i}}+e_{f\\left( \\hat{\\bm{t}}\\right)_{i}}$, '
                '$r_{\\hat{R}\\hat{\\bm{t}}}=\\text{max}\\left( \\hat{\\bm{e}}_{\\Sigma}\\right) '
                '-\\text{min}\\left( \\hat{\\bm{e}}_{\\Sigma}\\right)$, '
                '$e_{f\\left( \\hat{R}\\right)_{i}}=\\lvert \\mu_{\\Delta \\hat{R}_{\\Sigma ,i}}\\rvert '
                '+\\sigma_{\\Delta \\hat{R}_{\\Sigma ,i}}$, '
                '$e_{f\\left( \\hat{\\bm{t}}\\right)_{i}}=\\lvert \\mu_{\\angle{\\Delta \\hat{\\bm{t}}_{i}}}\\rvert '
                '+\\sigma_{\\angle{\\Delta \\hat{\\bm{t}}_{i}}}$, '
                '$\\hat{\\bm{e}}_{\\Sigma}=\\left[ \\hat{e}_{\\Sigma ,i}, '
                '\\ldots ,\\hat{e}_{\\Sigma ,n_{R\\bm{t}}}\\right]$, and '
                'number $n_{R\\bm{t}}$ of available $R$ and $\\bm{t}$ error statistics. '                
                # '\\left( e_{f\\left( \\hat{R}\\right) }r_{\\hat{\\bm{t}}}+'
                # 'e_{f\\left( \\hat{\\bm{t}}\\right) }r_{\\hat{R}}\\right) /2r_{\\hat{R}}r_{\\hat{\\bm{t}}}$ with '
                # '$e_{f\\left( \\hat{R}\\right) }=\\lvert \\mu_{\\Delta \\hat{R}_{\\Sigma}}\\rvert '
                # '+\\sigma_{\\Delta \\hat{R}_{\\Sigma}}$, '
                # '$e_{f\\left( \\hat{\\bm{t}}\\right) }=\\lvert \\mu_{\\angle{\\Delta \\hat{\\bm{t}}}}\\rvert '
                # '+\\sigma_{\\angle{\\Delta \\hat{\\bm{t}}}}$, '
                # '$r_{\\hat{R}}=\\text{max}\\left( e_{f\\left( \\hat{R}\\right) }\\right) '
                # '-\\text{min}\\left( e_{f\\left( \\hat{R}\\right) }\\right)$, and '
                # '$r_{\\hat{\\bm{t}}}=\\text{max}\\left( e_{f\\left( \\hat{\\bm{t}}\\right) }\\right) '
                # '-\\text{min}\\left( e_{f\\left( \\hat{\\bm{t}}\\right) }\\right)$. '
                '$\\mu_{\\Delta \\hat{R}_{\\Sigma}}$ and $\\mu_{\\angle{\\Delta \\hat{\\bm{t}}}}$ '
                'indicate the corresponding mean '
                'values of differential angles $\\Delta \\hat{R}_{\\Sigma}$ and $\\angle{\\Delta \\hat{\\bm{t}}}$. '
                '$\\sigma_{\\Delta \\hat{R}_{\\Sigma}}$ and $\\sigma_{\\angle{\\Delta \\hat{\\bm{t}}}}$ stand for the '
                'standard deviations of afore mentioned data.', True)
    elif label == 'Rt_diff_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of combined rotation $R$ and translation $\\bm{t}$ errors '
                '$e_{f\\left( R\\bm{t}\\right)_{i}}=\\left( e_{\\Sigma ,i}-'
                '\\text{min}\\left( \\bm{e}_{\\Sigma}\\right)\\right)/r_{R\\bm{t}}$ with '
                '$e_{\\Sigma, i}=e_{f\\left( R\\right)_{i}}+e_{f\\left( \\bm{t}\\right)_{i}}$, '
                '$r_{R\\bm{t}}=\\text{max}\\left( \\bm{e}_{\\Sigma}\\right)'
                '-\\text{min}\\left( \\bm{e}_{\\Sigma}\\right)$, '
                '$e_{f\\left( R\\right)_{i}}=\\lvert \\mu_{\\Delta R_{\\Sigma ,i}}\\rvert '
                '+\\sigma_{\\Delta R_{\\Sigma ,i}}$, '
                '$e_{f\\left( \\bm{t}\\right)_{i}}=\\lvert \\mu_{\\angle{\\Delta \\bm{t}_{i}}}\\rvert '
                '+\\sigma_{\\angle{\\Delta \\bm{t}_{i}}}$, '
                '$\\bm{e}_{\\Sigma}=\\left[ e_{\\Sigma, 1}, \\ldots ,e_{\\Sigma, n_{R\\bm{t}}}\\right]$, and '
                'number $n_{R\\bm{t}}$ of available $R$ and $\\bm{t}$ error statistics. '
                '$\\mu_{\\Delta R_{\\Sigma}}$ and $\\mu_{\\angle{\\Delta \\bm{t}}}$ indicate the corresponding mean '
                'values of differential angles $\\Delta R_{\\Sigma}$ and $\\angle{\\Delta \\bm{t}}$. '
                '$\\sigma_{\\Delta R_{\\Sigma}}$ and $\\sigma_{\\angle{\\Delta \\bm{t}}}$ stand for the '
                'standard deviations of afore mentioned data.', True)
    elif label == 'Rt_diff2':
        return (replaceCSVLabels(label),
                'Difference from frame to frame $\\Delta e_{R\\bm{t}}='
                # '\\left( \\Delta e_{\\Sigma}-'
                # '\\text{min}\\left( \\Delta \\bm{e}_{\\Sigma}\\right)\\right)/r_{\\Delta e}$ '
                '\\Delta e_{\\Sigma}/r_{\\Delta e}$ '
                'of combined rotation $R$ and translation $\\bm{t}$ error differences '
                '$\\Delta^{2}R_{i}=\\Delta R_{i}-'
                '\\Delta R_{i-1}\\; \\forall i \\in \\left[ 1,\\; n_{I}\\; \\right]$ and '
                '$\\angle{\\Delta^{2} \\bm{t}_{i}}=\\angle{\\Delta \\bm{t}_{i}}-\\angle{\\Delta \\bm{t}_{i-1}}\\; '
                '\\forall i \\in \\left[ 1,\\; n_{I}\\; \\right]$ with '
                '$\\Delta e_{\\Sigma ,i}=\\text{sgn}\\left( \\Delta R_{i}\\right)\\Delta^{2}R_{i}+'
                '\\text{sgn}\\left( \\angle{\\Delta \\bm{t}_{i}}\\right)\\angle{\\Delta^{2} \\bm{t}_{i}}$, '
                '$r_{\\Delta e}=\\text{max}\\left( \\Delta \\bm{e}_{\\Sigma}\\right)'
                '-\\text{min}\\left( \\Delta \\bm{e}_{\\Sigma}\\right)$, '
                '$\\Delta \\bm{e}_{\\Sigma}=\\left[ \\Delta e_{\\Sigma ,1}, '
                '\\ldots ,\\Delta e_{\\Sigma ,n_{I}}\\right]$, sign function $\\text{sgn}\\left( \\right)$, '
                'and number of stereo image pairs $n_{I}$.', True)
    elif label == 'Rt_mostLikely_diff_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of combined rotation $\\hat{R}$ and '
                'translation $\\hat{\\bm{t}}$ errors '                
                '$\\hat{e}_{f\\left( \\hat{R}\\hat{\\bm{t}}\\right)_{i}}='
                '\\left( \\hat{e}_{\\Sigma ,i}-\\text{min}\\left( \\hat{\\bm{e}}_{\\Sigma}\\right)\\right)'
                '/r_{\\hat{R}\\hat{\\bm{t}}}$ with '
                '$\\hat{e}_{\\Sigma ,i}=e_{f\\left( \\hat{R}\\right)_{i}}+e_{f\\left( \\hat{\\bm{t}}\\right)_{i}}$, '
                '$r_{\\hat{R}\\hat{\\bm{t}}}=\\text{max}\\left( \\hat{\\bm{e}}_{\\Sigma}\\right) '
                '-\\text{min}\\left( \\hat{\\bm{e}}_{\\Sigma}\\right)$, '
                '$e_{f\\left( \\hat{R}\\right)_{i}}=\\lvert \\mu_{\\Delta \\hat{R}_{\\Sigma ,i}}\\rvert '
                '+\\sigma_{\\Delta \\hat{R}_{\\Sigma ,i}}$, '
                '$e_{f\\left( \\hat{\\bm{t}}\\right)_{i}}=\\lvert \\mu_{\\angle{\\Delta \\hat{\\bm{t}}_{i}}}\\rvert '
                '+\\sigma_{\\angle{\\Delta \\hat{\\bm{t}}_{i}}}$, '
                '$\\hat{\\bm{e}}_{\\Sigma}=\\left[ \\hat{e}_{\\Sigma ,i}, '
                '\\ldots ,\\hat{e}_{\\Sigma ,n_{R\\bm{t}}}\\right]$, and '
                'number $n_{R\\bm{t}}$ of available $R$ and $\\bm{t}$ error statistics. '
                '$\\mu_{\\Delta \\hat{R}_{\\Sigma}}$ and $\\mu_{\\angle{\\Delta \\hat{\\bm{t}}}}$ '
                'indicate the corresponding mean '
                'values of differential angles $\\Delta \\hat{R}_{\\Sigma}$ and $\\angle{\\Delta \\hat{\\bm{t}}}$. '
                '$\\sigma_{\\Delta \\hat{R}_{\\Sigma}}$ and $\\sigma_{\\angle{\\Delta \\hat{\\bm{t}}}}$ stand for the '
                'standard deviations of afore mentioned data.', True)
    elif label == 'K12_cxyfxfyNorm':
        return (replaceCSVLabels(label),
                'Combined camera matrix parameter differences $e_{\\mli{K1,2}}='
                '\\left( \\mu_{\\Delta \\mli{K1}}+'
                '\\mu_{\\Delta \\mli{K2}}+'
                '\\sigma_{\\Delta \\mli{K1}}+'
                '\\sigma_{\\Delta \\mli{K2}}\\right) /2$ with '
                'mean values $\\mu_{\\Delta \\mli{K1}}$ and '
                '$\\mu_{\\Delta \\mli{K2}}$ '
                'in addition to standard deviations '
                '$\\sigma_{\\Delta \\mli{K1}}$ and '
                '$\\sigma_{\\Delta \\mli{K2}}$ of '
                '$\\lvert\\Delta c_{x,y}^{\\mli{K1}}\\, \\Delta f_{x,y}^{\\mli{K1}}\\rvert$ and '
                '$\\lvert\\Delta c_{x,y}^{\\mli{K2}}\\, \\Delta f_{x,y}^{\\mli{K2}}\\rvert$.', True)
    elif label == 'poolSize_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame on the number of matches $\\Delta n_{pool}$ '
                'within the correspondence pool', True)
    else:
        return (replaceCSVLabels(label), replaceCSVLabels(label), False)

def replaceCSVLabels(label, use_plural=False, str_capitalize=False):
    if label == 'R_diffAll':
        return '$\\Delta R_{\\Sigma}$'
    elif label == 'R_diff_roll_deg':
        return '$\\Delta R_{x}$'
    elif label == 'R_diff_pitch_deg':
        return '$\\Delta R_{y}$'
    elif label == 'R_diff_yaw_deg':
        return '$\\Delta R_{z}$'
    elif label == 't_angDiff_deg':
        return '$\\angle{\\Delta \\bm{t}}$'
    elif label == 't_distDiff':
        return '$\\lvert\\Delta \\bm{t}\\rvert$'
    elif label == 't_diff_tx':
        return '$\\Delta t_{x}$'
    elif label == 't_diff_ty':
        return '$\\Delta t_{y}$'
    elif label == 't_diff_tz':
        return '$\\Delta t_{z}$'
    elif label == 'Rt_diff':
        return '$e_{f\\left( R\\bm{t}\\right) }$'
    elif label == 'R_diffAll_diff':
        return '$\\Delta^{2} R_{\\Sigma}$'
    elif label == 'R_diff_roll_deg_diff':
        return '$\\Delta^{2} R_{x}$'
    elif label == 'R_diff_pitch_deg_diff':
        return '$\\Delta^{2} R_{y}$'
    elif label == 'R_diff_yaw_deg_diff':
        return '$\\Delta^{2} R_{z}$'
    elif label == 't_angDiff_deg_diff':
        return '$\\angle{\\Delta^{2} \\bm{t}}$'
    elif label == 't_distDiff_diff':
        return '$\\Delta\\lvert\\Delta \\bm{t}\\rvert$'
    elif label == 't_diff_tx_diff':
        return '$\\Delta^{2} t_{x}$'
    elif label == 't_diff_ty_diff':
        return '$\\Delta^{2} t_{y}$'
    elif label == 't_diff_tz_diff':
        return '$\\Delta^{2} t_{z}$'
    elif label == 'Rt_diff_diff':
        return '$\\Delta e_{f\\left( R\\bm{t}\\right) }$'
    elif label == 'Rt_diff2':
        return '$\\Delta e_{R\\bm{t}}$'
    elif label == 'th':
        if use_plural:
            str_val = 'thresholds \\texttt{th}'
        else:
            str_val = 'threshold \\texttt{th}'
    elif label == 'kpAccSd':
        if use_plural:
            str_val = 'key point accuracies $\\epsilon_{kp}^{\\sigma}$'
        else:
            str_val = 'key point accuracy $\\epsilon_{kp}^{\\sigma}$'
    elif label == 'inlratMin' or label == 'inlratMax':
        if use_plural:
            str_val = 'inlier ratios $\\epsilon$'
        else:
            str_val = 'inlier ratio $\\epsilon$'
    elif label == 'R_mostLikely_diffAll':
        return '$\\Delta \\hat{R}_{\\Sigma}$'
    elif label == 'R_mostLikely_diff_roll_deg':
        return '$\\Delta \\hat{R}_{x}$'
    elif label == 'R_mostLikely_diff_pitch_deg':
        return '$\\Delta \\hat{R}_{y}$'
    elif label == 'R_mostLikely_diff_yaw_deg':
        return '$\\Delta \\hat{R}_{z}$'
    elif label == 't_mostLikely_angDiff_deg':
        return '$\\angle{\\Delta \\hat{\\bm{t}}}$'
    elif label == 't_mostLikely_distDiff':
        return '$\\lvert\\Delta \\hat{\\bm{t}}\\rvert$'
    elif label == 't_mostLikely_diff_tx':
        return '$\\Delta \\hat{t}_{x}$'
    elif label == 't_mostLikely_diff_ty':
        return '$\\Delta \\hat{t}_{y}$'
    elif label == 't_mostLikely_diff_tz':
        return '$\\Delta \\hat{t}_{z}$'
    elif label == 'Rt_mostLikely_diff':
        return '$\\hat{e}_{f\\left( \\hat{R}\\hat{\\bm{t}}\\right) }$'
    elif label == 'R_mostLikely_diffAll_diff':
        return '$\\Delta^{2} \\hat{R}_{\\Sigma}$'
    elif label == 'R_mostLikely_diff_roll_deg_diff':
        return '$\\Delta^{2} \\hat{R}_{x}$'
    elif label == 'R_mostLikely_diff_pitch_deg_diff':
        return '$\\Delta^{2} \\hat{R}_{y}$'
    elif label == 'R_mostLikely_diff_yaw_deg_diff':
        return '$\\Delta^{2} \\hat{R}_{z}$'
    elif label == 't_mostLikely_angDiff_deg_diff':
        return '$\\angle{\\Delta^{2} \\hat{\\bm{t}}}$'
    elif label == 't_mostLikely_distDiff_diff':
        return '$\\Delta\\lvert\\Delta \\hat{\\bm{t}}\\rvert$'
    elif label == 't_mostLikely_diff_tx_diff':
        return '$\\Delta^{2} \\hat{t}_{x}$'
    elif label == 't_mostLikely_diff_ty_diff':
        return '$\\Delta^{2} \\hat{t}_{y}$'
    elif label == 't_mostLikely_diff_tz_diff':
        return '$\\Delta^{2} \\hat{t}_{z}$'
    elif label == 'Rt_mostLikely_diff_diff':
        return '$\\Delta\\hat{e}_{f\\left( \\hat{R}\\hat{\\bm{t}}\\right) }$'
    elif label == 'K1_fxDiff':
        return '$\\Delta f_{x}^{\\mli{K1}}$'
    elif label == 'K1_fyDiff':
        return '$\\Delta f_{y}^{\\mli{K1}}$'
    elif label == 'K1_fxyDiffNorm':
        return '$\\lvert\\Delta f_{x,y}^{\\mli{K1}}\\rvert$'
    elif label == 'K1_cxDiff':
        return '$\\Delta c_{x}^{\\mli{K1}}$'
    elif label == 'K1_cyDiff':
        return '$\\Delta c_{y}^{\\mli{K1}}$'
    elif label == 'K1_cxyDiffNorm':
        return '$\\lvert\\Delta c_{x,y}^{\\mli{K1}}\\rvert$'
    elif label == 'K1_cxyfxfyNorm':
        return '$\\lvert\\Delta c_{x,y}^{\\mli{K1}}\\, \\Delta f_{x,y}^{\\mli{K1}}\\rvert$'
    elif label == 'K2_fxDiff':
        return '$\\Delta f_{x}^{\\mli{K2}}$'
    elif label == 'K2_fyDiff':
        return '$\\Delta f_{y}^{\\mli{K2}}$'
    elif label == 'K2_fxyDiffNorm':
        return '$\\lvert\\Delta f_{x,y}^{\\mli{K2}}\\rvert$'
    elif label == 'K2_cxDiff':
        return '$\\Delta c_{x}^{\\mli{K2}}$'
    elif label == 'K2_cyDiff':
        return '$\\Delta c_{y}^{\\mli{K2}}$'
    elif label == 'K2_cxyDiffNorm':
        return '$\\lvert\\Delta c_{x,y}^{\\mli{K2}}\\rvert$'
    elif label == 'K2_cxyfxfyNorm':
        return '$\\lvert\\Delta c_{x,y}^{\\mli{K2}}\\, \\Delta f_{x,y}^{\\mli{K2}}\\rvert$'
    elif label == 'K12_cxyfxfyNorm':
        # return '$e_{\\lvert\\Delta c_{x,y}^{\\mli{K2}}\\, \\Delta f_{x,y}^{\\mli{K2}}\\rvert }$'
        return '$e_{\\mli{K1,2}}$'
    elif label == 'inlRat_estimated':
        if use_plural:
            str_val = 'estimated inlier ratios $\\tilde{\\epsilon}$'
        else:
            str_val = 'estimated inlier ratio $\\tilde{\\epsilon}$'
    elif label == 'inlRat_GT':
        if use_plural:
            str_val = 'GT inlier ratios $\\breve{\\epsilon}$'
        else:
            str_val = 'GT inlier ratio $\\breve{\\epsilon}$'
    elif label == 'inlRat_diff':
        if use_plural:
            str_val = 'inlier ratio differences $\\Delta \\epsilon = \\tilde{\\epsilon} - \\breve{\\epsilon}$'
        else:
            str_val = 'inlier ratio difference $\\Delta \\epsilon = \\tilde{\\epsilon} - \\breve{\\epsilon}$'
    elif label == 'nrCorrs_filtered':
        if use_plural:
            str_val = 'numbers of filtered correspondences $n_{fc}$'
        else:
            str_val = '# filtered correspondences $n_{fc}$'
    elif label == 'nrCorrs_estimated':
        if use_plural:
            str_val = 'numbers of estimated correspondences $\\tilde{n}_{c}$'
        else:
            str_val = '# estimated correspondences $\\tilde{n}_{c}$'
    elif label == 'nrCorrs_GT':
        if use_plural:
            str_val = 'numbers of GT correspondences $n_{GT}$'
        else:
            str_val = '# GT correspondences $n_{GT}$'
    elif label == 'filtering_us':
        if use_plural:
            str_val = 'filtering times $t_{f}$'
        else:
            str_val = 'filtering time $t_{f}$'
    elif label == 'robEstimationAndRef_us':
        if use_plural:
            str_val = 'estimation and refinement times $t_{e}$'
        else:
            str_val = 'estimation and refinement time $t_{e}$'
    elif label == 'linRefinement_us':
        if use_plural:
            str_val = 'linear refinement times $t_{l}$'
        else:
            str_val = 'linear refinement time $t_{l}$'
    elif label == 'bundleAdjust_us':
        if use_plural:
            str_val = 'BA times $t_{BA}$'
        else:
            str_val = 'BA time $t_{BA}$'
    elif label == 'stereoRefine_us':
        if use_plural:
            str_val = 'stereo refinement times $t_{s}$'
        else:
            str_val = 'stereo refinement time $t_{s}$'
    elif label == 'linRef_BA_us':
        if use_plural:
            str_val = 'linear refinement and BA times $t_{l,BA}=t_{l}+t_{BA}$'
        else:
            str_val = 'linear refinement and BA time $t_{l,BA}=t_{l}+t_{BA}$'
    elif label == 'poolSize':
        if use_plural:
            str_val = 'numbers of matches $n_{pool}$ within the correspondence pool'
        else:
            str_val = '# of matches $n_{pool}$ within the correspondence pool'
    elif label == 'poolSize_diff':
        str_val = '$\\Delta n_{pool}$'
    elif label == 'kpDistr':
        if use_plural:
            str_val = 'keypoint distributions'
        else:
            str_val = 'keypoint distribution'
    elif label == 'depthDistr':
        if use_plural:
            str_val = 'depth distributions'
        else:
            str_val = 'depth distribution'
    elif label == 'nrTP':
        if use_plural:
            str_val = 'numbers of TP'
        else:
            return '# TP'
    elif label == 'inlratCRate':
        if use_plural:
            str_val = 'inlier ratio change rates $c_{\\breve{\\epsilon}}$'
        else:
            str_val = 'inlier ratio change rate $c_{\\breve{\\epsilon}}$'
    elif label == 'USAC_parameters_th_pixels':
        if use_plural:
            str_val = 'USAC thresholds $\\texttt{th}_{USAC}$'
        else:
            str_val = 'USAC threshold $\\texttt{th}_{USAC}$'
    elif label == 'USAC_parameters_USACInlratFilt':
        if use_plural:
            str_val = 'USAC inlier ratio filters'
        else:
            str_val = 'USAC inlier ratio filter'
    elif label == 'USAC_parameters_automaticSprtInit':
        if use_plural:
            str_val = 'USAC automatic SPRT initializations'
        else:
            str_val = 'USAC automatic SPRT initialization'
    elif label == 'USAC_parameters_automaticProsacParameters':
        if use_plural:
            str_val = 'automatic PROSAC parameter estimations'
        else:
            str_val = 'automatic PROSAC parameter estimation'
    elif label == 'USAC_parameters_prevalidateSample':
        if use_plural:
            str_val = 'sample prevalidations'
        else:
            str_val = 'sample prevalidation'
    elif label == 'USAC_parameters_estimator':
        if use_plural:
            str_val = 'USAC estimators'
        else:
            str_val = 'USAC estimator'
    elif label == 'USAC_parameters_refinealg':
        if use_plural:
            str_val = 'USAC refinement algorithms'
        else:
            str_val = 'USAC refinement algorithm'
    elif label == 'RobMethod':
        if use_plural:
            str_val = 'robust estimation methods'
        else:
            str_val = 'robust estimation method'
    elif label == 'refineMethod_algorithm':
        if use_plural:
            str_val = 'refinement algorithms'
        else:
            str_val = 'refinement algorithm'
    elif label == 'refineMethod_costFunction':
        if use_plural:
            str_val = 'refinement algorithm cost functions'
        else:
            str_val = 'refinement algorithm cost function'
    elif label == 'kneipInsteadBA':
        str_val = 'Kneip instead BA'
    elif label == 'BART':
        if use_plural:
            str_val = 'BA options'
        else:
            str_val = 'BA option'
    elif label == 'matchesFilter_refineGMS':
        str_val = 'GMS correspondence filter'
    elif label == 'matchesFilter_refineVFC':
        str_val = 'VFC correspondence filter'
    elif label == 'matchesFilter_refineSOF':
        str_val = 'SOF correspondence filter'
    elif label == 'stereoParameters_th_pix_user':
        if use_plural:
            str_val = 'stereo refinement thresholds $\\texttt{th}_{s}$'
        else:
            str_val = 'stereo refinement threshold $\\texttt{th}_{s}$'
    elif label == 'stereoParameters_keypointType':
        if use_plural:
            str_val = 'keypoint types'
        else:
            str_val = 'keypoint type'
    elif label == 'stereoParameters_descriptorType':
        if use_plural:
            str_val = 'descriptor types'
        else:
            str_val = 'descriptor type'
    elif label == 'stereoParameters_RobMethod':
        if use_plural:
            str_val = 'robust estimation methods for stereo refinement'
        else:
            str_val = 'robust estimation method for stereo refinement'
    elif label == 'stereoParameters_refineMethod_algorithm':
        if use_plural:
            str_val = 'refinement algorithms for stereo refinement'
        else:
            str_val = 'refinement algorithm for stereo refinement'
    elif label == 'stereoParameters_refineMethod_costFunction':
        if use_plural:
            str_val = 'refinement algorithm cost functions for stereo refinement'
        else:
            str_val = 'refinement algorithm cost function for stereo refinement'
    elif label == 'stereoParameters_refineMethod_CorrPool_algorithm':
        if use_plural:
            str_val = 'refinement algorithms for correspondence pool'
        else:
            str_val = 'refinement algorithm for correspondence pool'
    elif label == 'stereoParameters_refineMethod_CorrPool_costFunction':
        if use_plural:
            str_val = 'refinement algorithm cost functions for correspondence pool'
        else:
            str_val = 'refinement algorithm cost function for correspondence pool'
    elif label == 'stereoParameters_kneipInsteadBA':
        str_val = 'Kneip instead BA for stereo refinement'
    elif label == 'stereoParameters_kneipInsteadBA_CorrPool':
        str_val = 'Kneip instead BA for correspondence pool'
    elif label == 'stereoParameters_BART':
        if use_plural:
            str_val = 'BA options for stereo refinement'
        else:
            str_val = 'BA option for stereo refinement'
    elif label == 'stereoParameters_BART_CorrPool':
        if use_plural:
            str_val = 'BA options for correspondence pool'
        else:
            str_val = 'BA option for correspondence pool'
    elif label == 'stereoParameters_checkPoolPoseRobust':
        if use_plural:
            str_val = 'iteration parameters for robust correspondence pool check'
        else:
            str_val = 'iteration parameter for robust correspondence pool check'
    elif label == 'stereoParameters_useRANSAC_fewMatches':
        if use_plural:
            str_val = 'options of RANSAC usage for few matches'
        else:
            str_val = 'RANSAC usage for few matches'
    elif label == 'stereoParameters_maxPoolCorrespondences':
        if use_plural:
            str_val = 'maximum correspondence pool sizes $\\hat{n}_{cp}$'
        else:
            str_val = 'maximum correspondence pool size $\\hat{n}_{cp}$'
    elif label == 'stereoParameters_maxDist3DPtsZ':
        if use_plural:
            str_val = 'maximum depths $z_{max}$'
        else:
            str_val = 'maximum depth $z_{max}$'
    elif label == 'stereoParameters_maxRat3DPtsFar':
        if use_plural:
            str_val = 'maximum far correspondence ratios $r_{max}^{z}$'
        else:
            str_val = 'maximum far correspondence ratio $r_{max}^{z}$'
    elif label == 'stereoParameters_minStartAggInlRat':
        if use_plural:
            str_val = 'minimum initial inlier ratios $\\epsilon_{min}^{init}$'
        else:
            str_val = 'minimum initial inlier ratio $\\epsilon_{min}^{init}$'
    elif label == 'stereoParameters_minInlierRatSkip':
        if use_plural:
            str_val = 'minimum inlier ratios $\\epsilon_{min}^{skip}$ for skipping images'
        else:
            str_val = 'minimum inlier ratio $\\epsilon_{min}^{skip}$ for skipping images'
    elif label == 'stereoParameters_relInlRatThLast':
        if use_plural:
            str_val = 'relative inlier ratio thresholds $r_{\\epsilon}^{old}$ on old models'
        else:
            str_val = 'relative inlier ratio threshold $r_{\\epsilon}^{old}$ on old models'
    elif label == 'stereoParameters_relInlRatThNew':
        if use_plural:
            str_val = 'relative inlier ratio thresholds $r_{\\epsilon}^{new}$ on new models'
        else:
            str_val = 'relative inlier ratio threshold $r_{\\epsilon}^{new}$ on new models'
    elif label == 'stereoParameters_relMinInlierRatSkip':
        if use_plural:
            str_val = 'minimum relative inlier ratios $r_{min}^{\\epsilon,skip}$ for skipping images'
        else:
            str_val = 'minimum relative inlier ratio $r_{min}^{\\epsilon,skip}$ for skipping images'
    elif label == 'stereoParameters_minInlierRatioReInit':
        if use_plural:
            str_val = 'minimum inlier ratio reinitialization thresholds $\\epsilon_{min}^{reinit}$'
        else:
            str_val = 'minimum inlier ratio reinitialization threshold $\\epsilon_{min}^{reinit}$'
    elif label == 'stereoParameters_maxSkipPairs':
        if use_plural:
            str_val = 'parameter values for maximum image pairs to skip $n_{max}^{skip}$'
        else:
            str_val = 'maximum image pairs to skip $n_{max}^{skip}$'
    elif label == 'stereoParameters_minNormDistStable':
        if use_plural:
            str_val = 'minimum distances $\\lvert\\breve{d}_{cm}\\rvert$ for stability'
        else:
            str_val = 'minimum distance $\\lvert\\breve{d}_{cm}\\rvert$ for stability'
    elif label == 'stereoParameters_absThRankingStable':
        if use_plural:
            str_val = 'stable ranking thresholds $\\tau$'
        else:
            str_val = 'stable ranking threshold $\\tau$'
    elif label == 'stereoParameters_minContStablePoses':
        if use_plural:
            str_val = 'parameter values for minimum continuous stable poses $n_{min}^{stable}$'
        else:
            str_val = 'minimum continuous stable poses $n_{min}^{stable}$'
    elif label == 'stereoParameters_raiseSkipCnt':
        if use_plural:
            str_val = 'skip count raising factors $a_{skip}$'
        else:
            str_val = 'skip count raising factor $a_{skip}$'
    elif label == 'stereoParameters_minPtsDistance':
        if use_plural:
            str_val = 'minimum correspondence distances $d_{p}$'
        else:
            str_val = 'minimum correspondence distance $d_{p}$'
    else:
        return tex_string_coding_style(label)
    ex = ['and', 'of', 'for', 'to', 'with', 'on']
    if str_capitalize:
        return ' '.join([b.capitalize() if not b.isupper() and
                                           not '$' in b and
                                           not '\\' in b and
                                           not b in ex else b for b in str_val.split(' ')])
    else:
        return str_val


def capitalizeStr(str_val):
    ex = ['and', 'of', 'for', 'to', 'with', 'on']
    return ' '.join([b.capitalize() if not b.isupper() and
                                       not '$' in b and
                                       not '\\' in b and
                                       not len(b) == 1 and
                                       not '{' in b and
                                       not '}' in b and
                                       not '_' in b and
                                       not b in ex else b for b in str_val.split(' ')])


def capitalizeFirstChar(str_val):
    str_l = str_val.split(' ')
    str_l[0] = str_l[0].capitalize()
    return ' '.join(str_l)


def strToLower(str_val):
    return ' '.join([b.lower() if not sum(1 for c in b if c.isupper()) > 1 and
                                  not '$' in b and
                                  not len(b) == 1 and
                                  not '{' in b and
                                  not '}' in b and
                                  not '_' in b and
                                  not '\\' in b else b for b in str_val.split(' ')])

def getOptionDescription(key):
    if key == 'GMS':
        return 'Grid-based Motion Statistics', True
    elif key == 'VFC':
        return 'Vector Field Consensus', True
    elif key == 'SPRT_DEFAULT_INIT':
        return 'Sequential Probability Ratio Test (SPRT) with default values $\\delta_{SPRT} = 0.05$ and ' \
               '$\\epsilon_{SPRT} = 0.15$, where $\\delta_{SPRT}$ corresponds to the propability of a keypoint to be ' \
               'classified as an inlier of an invalid model and $\\epsilon_{SPRT}$ to the probability ' \
               'that a data point is consistent with a good model.', True
    elif key == 'SPRT_DELTA_AUTOM_INIT':
        return 'Sequential Probability Ratio Test (SPRT) with automatic estimation of $\\delta_{SPRT}$ and a default ' \
               'value $\\epsilon_{SPRT} = 0.15$, where $\\delta_{SPRT}$ corresponds to the propability of a keypoint' \
               ' to be classified as an inlier of an invalid model and $\\epsilon_{SPRT}$ to the probability ' \
               'that a data point is consistent with a good model.', True
    elif key == 'SPRT_EPSILON_AUTOM_INIT':
        return 'Sequential Probability Ratio Test (SPRT) with automatic estimation of ' \
               '$\\epsilon_{SPRT}$ and a default value $\\delta_{SPRT} = 0.05$, where $\\delta_{SPRT}$ corresponds ' \
               'to the propability of a keypoint to be classified as an inlier of an invalid model and ' \
               '$\\epsilon_{SPRT}$ to the probability that a data point is consistent with a good model.', True
    elif key == 'SPRT_DELTA_AND_EPSILON_AUTOM_INIT':
        return 'Sequential Probability Ratio Test (SPRT) with automatic estimation of $\\delta_{SPRT}$ and ' \
               '$\\epsilon_{SPRT}$, where $\\delta_{SPRT}$ corresponds to the propability of a keypoint to be ' \
               'classified as an inlier of an invalid model and $\\epsilon_{SPRT}$ to the probability ' \
               'that a data point is consistent with a good model.', True
    elif key == 'POSE_NISTER':
        return 'Pose estimation using Nister\'s 5pt algorithm', True
    elif key == 'POSE_EIG_KNEIP':
        return 'Pose estimation using Kneip\'s Eigen solver', True
    elif key == 'POSE_STEWENIUS':
        return 'Pose estimation using Stewenius\' 5pt algorithm', True
    elif key == 'REF_WEIGHTS':
        return 'Inner refinement algorithm of USAC: 8pt algorithm using Torr weights ' \
               '(Equation 2.25 in Torr dissertation)', True
    elif key == 'REF_8PT_PSEUDOHUBER':
        return 'Inner refinement algorithm of USAC: 8pt algorithm using Pseudo-Huber weights', True
    elif key == 'REF_EIG_KNEIP':
        return 'Inner refinement algorithm of USAC: Kneip\'s Eigen solver using least squares', True
    elif key == 'REF_EIG_KNEIP_WEIGHTS':
        return 'Inner refinement algorithm of USAC: Kneip\'s Eigen solver using Torr weights ' \
               '(Equation 2.25 in Torr dissertation)', True
    elif key == 'REF_STEWENIUS':
        return 'Inner refinement algorithm of USAC: Stewenius\' 5pt algorithm using least squares', True
    elif key == 'REF_STEWENIUS_WEIGHTS':
        return 'Inner refinement algorithm of USAC: Stewenius\' 5pt algorithm using Pseudo-Huber weights', True
    elif key == 'REF_NISTER':
        return 'Inner refinement algorithm of USAC: Nister\'s 5pt algorithm using least squares', True
    elif key == 'REF_NISTER_WEIGHTS':
        return 'Inner refinement algorithm of USAC: Nister\'s 5pt algorithm using Pseudo-Huber weights', True
    elif key == 'extr_only':
        return 'Bundle Adjustment (BA) for extrinsics only including structure', True
    elif key == 'extr_intr':
        return 'Bundle Adjustment (BA) for extrinsics and intrinsics including structure', True
    elif key == 'PR_NO_REFINEMENT':
        return 'Refinement disabled', True
    elif key == 'PR_8PT':
        return 'Refinement using 8pt algorithm', True
    elif key == 'PR_NISTER':
        return 'Refinement using Nister\'s 5pt algorithm', True
    elif key == 'PR_STEWENIUS':
        return 'Refinement using Stewenius\' 5pt algorithm', True
    elif key == 'PR_KNEIP':
        return 'Refinement using Kneip\'s Eigen solver', True
    elif key == 'PR_TORR_WEIGHTS':
        return 'Cost function for refinement: Torr weights (Equation 2.25 in Torr dissertation)', True
    elif key == 'PR_PSEUDOHUBER_WEIGHTS':
        return 'Cost function for refinement: Pseudo-Huber weights', True
    elif key == 'PR_NO_WEIGHTS':
        return 'Cost function for refinement: least squares', True
    elif key == 'N':
        return 'Depth range of visible 3D points in both stereo cameras based on baseline and focal length: near range', True
    elif key == 'M':
        return 'Depth range of visible 3D points in both stereo cameras based on baseline and focal length: mid range', True
    elif key == 'F':
        return 'Depth range of visible 3D points in both stereo cameras based on baseline and focal length: far range', True
    elif key == 'NM':
        return 'Depth range of visible 3D points in both stereo cameras based on baseline and focal length: near and mid range', True
    elif key == 'MF':
        return 'Depth range of visible 3D points in both stereo cameras based on baseline and focal length: mid and far range', True
    elif key == 'NF':
        return 'Depth range of visible 3D points in both stereo cameras based on baseline and focal length: near and far range', True
    elif key == 'NMF':
        return 'Depth range of visible 3D points in both stereo cameras based on baseline and focal length: near, mid, and far (full) range', True
    elif key == '1corn':
        return 'Correspondences are concentrated mainly in one corner of the image', True
    elif key == 'half-img':
        return 'Correspondences are equally distributed over half the image', True
    elif key == 'equ':
        return 'Correspondences are equally distributed over the whole image', True
    else:
        test_glossary = False
        if test_glossary:
            warnings.warn('Glossary test is enabled!', UserWarning)
        return tex_string_coding_style(key), False if not test_glossary else True


def glossary_from_list(entries):
    mylist = list(dict.fromkeys(entries))
    gloss = []
    for elem in mylist:
        des, found = getOptionDescription(elem)
        if found:
            gloss.append({'key': tex_string_coding_style(elem), 'description': des})
    return gloss

def add_to_glossary(entries, gloss):
    mylist = list(dict.fromkeys(entries))
    for elem in mylist:
        elem_tex = tex_string_coding_style(str(elem))
        if gloss:
            found = False
            for entry in gloss:
                if entry['key'] == elem_tex:
                    found = True
                    break
            if found:
                continue
        else:
            gloss = []
        des, found = getOptionDescription(str(elem))
        if found:
            gloss.append({'key': elem_tex, 'description': des})
    return gloss


def build_list(possibilities, multiplier, num_pts):
    tmp = []
    for i in possibilities:
        tmp += [i] * int(num_pts / (len(possibilities) * multiplier))
    tmp *= multiplier
    return tmp


#Only for testing
def main():
    num_pts = int(10000)
    nr_imgs = 150
    pars1_opt = ['first_long_long_opt' + str(i) for i in range(0,3)]
    pars2_opt = ['second_long_opt' + str(i) for i in range(0, 3)]
    pars3_opt = ['third_long_long_opt' + str(i) for i in range(0, 2)]
    pars_kpDistr_opt = ['1corn', 'equ']
    pars_depthDistr_opt = ['NMF', 'NM', 'F']
    pars_nrTP_opt = ['500', '100to1000']
    pars_kpAccSd_opt = ['0.5', '1.0', '1.5']
    inlratMin_opt = list(np.arange(0.45, 0.85, 0.1))
    lin_time_pars = np.array([500, 3, 0.003])
    poolSize = [2000, 10000, 40000]
    min_pts = len(pars_kpAccSd_opt) * len(pars_depthDistr_opt) * len(inlratMin_opt) * \
              nr_imgs * len(poolSize)
    if min_pts < num_pts:
        while min_pts < num_pts:
            pars_kpAccSd_opt += [str(float(pars_kpAccSd_opt[-1]) + 0.5)]
            min_pts = len(pars_kpAccSd_opt) * len(pars_depthDistr_opt) * len(inlratMin_opt) * \
                      nr_imgs * len(poolSize)
        num_pts = min_pts
    else:
        num_pts = int(min_pts)
    kpAccSd_mul = int(len(pars_depthDistr_opt))
    inlratMin_mul = int(kpAccSd_mul * len(pars_kpAccSd_opt))
    USAC_parameters_estimator_mul = int(inlratMin_mul * len(inlratMin_opt))
    poolSize_mul = int(USAC_parameters_estimator_mul * len(pars1_opt))

    data = {#'R_diffAll': 1000 + np.abs(np.random.randn(num_pts) * 10),#[0.3, 0.5, 0.7, 0.4, 0.6] * int(num_pts/5),
            'R_diff_roll_deg': 1000 + np.abs(np.random.randn(num_pts) * 10),
            'R_diff_pitch_deg': 10 + np.random.randn(num_pts) * 5,
            'R_diff_yaw_deg': -1000 + np.abs(np.random.randn(num_pts)),
            # 't_angDiff_deg': [0.3, 0.5, 0.7, 0.4, 0.6] * int(num_pts/5),
            't_distDiff': np.abs(np.random.randn(num_pts) * 100),
            't_diff_tx': -10000 + np.random.randn(num_pts) * 100,
            't_diff_ty': 20000 + np.random.randn(num_pts),
            't_diff_tz': -450 + np.random.randn(num_pts),
            'K1_cxyfxfyNorm': 13 + np.random.randn(num_pts) * 5,
            'K2_cxyfxfyNorm': 14 + np.random.randn(num_pts) * 6,
            'K1_cxyDiffNorm': 15 + np.random.randn(num_pts) * 7,
            'K2_cxyDiffNorm': 16 + np.random.randn(num_pts) * 8,
            'K1_fxyDiffNorm': 17 + np.random.randn(num_pts) * 9,
            'K2_fxyDiffNorm': 18 + np.random.randn(num_pts) * 10,
            'K1_fxDiff': 19 + np.random.randn(num_pts) * 4,
            'K2_fxDiff': 20 + np.random.randn(num_pts) * 3,
            'K1_fyDiff': 21 + np.random.randn(num_pts) * 2,
            'K2_fyDiff': 22 + np.random.randn(num_pts),
            'K1_cxDiff': 23 + np.random.randn(num_pts) * 5,
            'K2_cxDiff': 24 + np.random.randn(num_pts) * 6,
            'K1_cyDiff': 25 + np.random.randn(num_pts) * 7,
            'K2_cyDiff': 26 + np.random.randn(num_pts) * 8,
            # 'USAC_parameters_estimator': np.random.randint(0, 3, num_pts),
            # 'USAC_parameters_refinealg': np.random.randint(0, 7, num_pts),
            # 'USAC_parameters_USACInlratFilt': np.random.randint(8, 10, num_pts),
            'stereoParameters_maxPoolCorrespondences': build_list(poolSize, poolSize_mul, num_pts),
            # 'USAC_parameters_estimator': [pars1_opt[i] for i in np.random.randint(0, len(pars1_opt), num_pts)],
            'USAC_parameters_estimator': build_list(pars1_opt, USAC_parameters_estimator_mul, num_pts),
            'USAC_parameters_refinealg': [pars2_opt[i] for i in np.random.randint(0, len(pars2_opt), num_pts)],
            'kpDistr': [pars_kpDistr_opt[i] for i in np.random.randint(0, len(pars_kpDistr_opt), num_pts)],
            # 'kpDistr': [[i] * (num_pts / len(pars_kpDistr_opt)) for i in pars_kpDistr_opt],
            # 'depthDistr': [pars_depthDistr_opt[i] for i in np.random.randint(0, len(pars_depthDistr_opt), num_pts)],
            'depthDistr': build_list(pars_depthDistr_opt, 1, num_pts),
            'nrTP': [pars_nrTP_opt[i] for i in np.random.randint(0, len(pars_nrTP_opt), num_pts)],
            # 'kpAccSd': [pars_kpAccSd_opt[i] for i in np.random.randint(0, len(pars_kpAccSd_opt), num_pts)],
            'kpAccSd': build_list(pars_kpAccSd_opt, kpAccSd_mul, num_pts),
            'USAC_parameters_USACInlratFilt': [pars3_opt[i] for i in np.random.randint(0, len(pars3_opt), num_pts)],
            'th': np.tile(np.arange(0.4, 0.9, 0.1), int(num_pts/5)),
            # 'inlratMin': np.tile(np.arange(0.05, 0.45, 0.1), int(num_pts/4)),
            'inlratMin': np.array(build_list(inlratMin_opt, inlratMin_mul, num_pts)),
            'useless': [1, 1, 2, 3] * int(num_pts/4),
            'R_out(0,0)': [0] * 10 + [1] * int(num_pts - 10),
            'R_out(0,1)': [0] * 10 + [0] * int(num_pts - 10),
            'R_out(0,2)': [float(0)] * 10 + [0.1] * int(num_pts - 10),
            'R_out(1,0)': [0] * 10 + [1] * int(num_pts - 10),
            'R_out(1,1)': [0] * 10 + [1] * int(num_pts - 10),
            'R_out(1,2)': [0] * 10 + [0] * int(num_pts - 10),
            'R_out(2,0)': [float(0)] * 10 + [0.1] * int(num_pts - 10),
            'R_out(2,1)': [0] * 10 + [0] * int(num_pts - 10),
            'R_out(2,2)': [0] * 10 + [0] * int(num_pts - 10),
            'Nr': list(range(0, nr_imgs)) * int(num_pts / nr_imgs),
            'inlRat_GT': np.tile(np.arange(0.25, 0.72, 0.05), int(num_pts/10))}

    eval_columns = ['K1_cxyfxfyNorm', 'K2_cxyfxfyNorm', 'K1_cxyDiffNorm', 'K2_cxyDiffNorm',
                    'K1_fxyDiffNorm', 'K2_fxyDiffNorm', 'K1_fxDiff', 'K2_fxDiff', 'K1_fyDiff',
                    'K2_fyDiff', 'K1_cxDiff', 'K2_cxDiff', 'K1_cyDiff', 'K2_cyDiff']

    data['poolSize'] = data['inlratMin'] * 500 - np.random.randint(0, 50, num_pts)
    data['poolSize'] *= (np.array(data['Nr']) + 1) * np.abs((0.01 * np.random.randn(num_pts) + 0.95))
    data['poolSize'] = np.round(data['poolSize'], decimals=0)

    min_val = data['poolSize'].min()
    range_val = data['poolSize'].max() - min_val
    tot = range_val / 10
    sigma = range_val / 4
    tau = tot + sigma / 6.67
    exp1 = np.exp(-1 * (data['poolSize'] - min_val) / tau)
    exp2 = np.exp(-1 * np.power(data['poolSize'] - min_val - tot, 2) / (sigma**2))
    exp_sum = 0.34 * exp1 + 0.9 * exp2
    data['R_diffAll'] = 0.5 + exp_sum + 0.15 * np.random.randn(num_pts)
    data['t_angDiff_deg'] = 0.3 + exp_sum + 0.08 * np.random.randn(num_pts)


    data['inlRat_estimated'] = data['inlRat_GT'] + 0.5 * np.random.random_sample(num_pts) - 0.25
    data['nrCorrs_GT'] = [int(a) if a == pars_nrTP_opt[0] else np.random.randint(100, 1000) for a in data['nrTP']]
    # t = np.tile(lin_time_pars[0], num_pts) + \
    #     lin_time_pars[1] * np.array(data['nrCorrs_GT']) + \
    #     lin_time_pars[2] * np.array(data['nrCorrs_GT']) * np.array(data['nrCorrs_GT'])
    t = np.tile(lin_time_pars[0], num_pts) + \
        lin_time_pars[1] * np.array(data['nrCorrs_GT'])
    t *= (data['inlratMin'].max() / data['inlratMin']) ** 2
    t += np.random.randn(num_pts) * 40
    idx1 = np.arange(0, num_pts, dtype=int)
    np.random.shuffle(idx1)
    gross_error_idx = idx1.tolist()[:int(num_pts/5)]
    t[gross_error_idx] += lin_time_pars[0] / 3
    data['robEstimationAndRef_us'] = t
    data['linRefinement_us'] = t
    data['bundleAdjust_us'] = t
    data['filtering_us'] = t
    data['stereoRefine_us'] = t
    data = pd.DataFrame(data)

    test_name = 'correspondence_pool'#'refinement_ba_stereo'#'vfc_gms_sof'#'refinement_ba'#'usac_vs_ransac'#'testing_tests'
    test_nr = 1
    eval_nr = [5]#list(range(5, 8))
    ret = 0
    output_path = '/home/maierj/work/Sequence_Test/py_test'
    # output_path = '/home/maierj/work/Sequence_Test/py_test/refinement_ba/2'
    if test_name == 'testing_tests':#'usac-testing':
        if not test_nr:
            raise ValueError('test_nr is required for usac-testing')
        if test_nr == 1:
            if eval_nr[0] < 0:
                evals = list(range(1, 7))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 1:
                    fig_title_pre_str = 'Statistics on R\\&t differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'USAC_opt_refine_ops_th'}
                    from usac_eval import get_best_comb_and_th_1
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_USAC_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['th'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_and_th_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 2:
                    fig_title_pre_str = 'Statistics on R\\&t differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'USAC_opt_refine_ops_inlrat'}
                    from usac_eval import get_best_comb_inlrat_1
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_USAC_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratMin'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_inlrat_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 3:
                    fig_title_pre_str = 'Values of R\\&t differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'res_par_name': 'USAC_opt_refine_ops_inlrat_th'}
                    from usac_eval import get_best_comb_and_th_for_inlrat_1
                    ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_USAC_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['th', 'inlratMin'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_and_th_for_inlrat_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  fig_type='surface',
                                                  use_marks=True,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 4:
                    fig_title_pre_str = 'Values of R\\&t differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd', 'th']#th must be at the end
                    special_calcs_args = {'build_pdf': (True, True), 'use_marks': True}
                    from usac_eval import get_best_comb_th_scenes_1
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_USAC_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratMin'],
                                                             filter_func=None,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_best_comb_th_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 5:
                    fig_title_pre_str = 'Execution Times for USAC Option Combinations of '
                    eval_columns = ['robEstimationAndRef_us']
                    units = []
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'smooth',
                                          'nr_target_kps': 1000,
                                          't_data_separators': ['inlratMin'],
                                          'res_par_name': 'USAC_opt_refine_min_time'}
                    from usac_eval import filter_nr_kps, calc_Time_Model, estimate_alg_time_fixed_kp
                    ret += calcFromFuncAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_USAC_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='time',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['nrCorrs_GT'],
                                                  filter_func=filter_nr_kps,
                                                  filter_func_args=None,
                                                  special_calcs_func=estimate_alg_time_fixed_kp,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=calc_Time_Model,
                                                  calc_func_args={'data_separators': ['inlratMin', 'th']},
                                                  fig_type='surface',
                                                  use_marks=True,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 6:
                    fig_title_pre_str = 'Execution Times for USAC Option Combinations of '
                    eval_columns = ['robEstimationAndRef_us']
                    units = []
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    from usac_eval import filter_nr_kps, calc_Time_Model
                    ret += calcFromFuncAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_USAC_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='time',
                                                             eval_columns=eval_columns,  # Column names for which statistics are calculated (y-axis)
                                                             units=units,  # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,  # Algorithm parameters to evaluate
                                                             partitions=['th'],  # Data properties to calculate results separately
                                                             x_axis_column=['nrCorrs_GT'],  # x-axis column name
                                                             filter_func=filter_nr_kps,
                                                             filter_func_args=None,
                                                             special_calcs_func=None,
                                                             special_calcs_args=None,
                                                             calc_func=calc_Time_Model,
                                                             calc_func_args={'data_separators': ['inlratMin', 'th']},
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=False)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        elif test_nr == 2:
            if eval_nr[0] < 0:
                evals = list(range(7, 15)) + [36]
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 7:
                    fig_title_pre_str = 'Statistics on R\\&t differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['USAC_parameters_automaticSprtInit',
                    #                  'USAC_parameters_automaticProsacParameters',
                    #                  'USAC_parameters_prevalidateSample',
                    #                  'USAC_parameters_USACInlratFilt']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'USAC_opt_search_ops_th'}
                    from usac_eval import get_best_comb_and_th_1
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_USAC_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['th'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_and_th_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 8:
                    fig_title_pre_str = 'Statistics on R\\&t differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['USAC_parameters_automaticSprtInit',
                    #                  'USAC_parameters_automaticProsacParameters',
                    #                  'USAC_parameters_prevalidateSample',
                    #                  'USAC_parameters_USACInlratFilt']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'USAC_opt_search_ops_inlrat'}
                    from usac_eval import get_best_comb_inlrat_1
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_USAC_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratMin'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_inlrat_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 9:
                    fig_title_pre_str = 'Values of R\\&t differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['USAC_parameters_automaticSprtInit',
                    #                  'USAC_parameters_automaticProsacParameters',
                    #                  'USAC_parameters_prevalidateSample',
                    #                  'USAC_parameters_USACInlratFilt']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'res_par_name': 'USAC_opt_search_ops_kpAccSd_th',
                                          'func_name': 'get_best_comb_and_th_for_kpacc_1'}
                    from usac_eval import get_best_comb_and_th_for_inlrat_1
                    ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_USAC_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['th', 'kpAccSd'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_and_th_for_inlrat_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  fig_type='surface',
                                                  use_marks=True,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 10:
                    fig_title_pre_str = 'Values of R\\&t differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['USAC_parameters_automaticSprtInit',
                    #                  'USAC_parameters_automaticProsacParameters',
                    #                  'USAC_parameters_prevalidateSample',
                    #                  'USAC_parameters_USACInlratFilt']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'res_par_name': 'USAC_opt_search_ops_inlrat_th'}
                    from usac_eval import get_best_comb_and_th_for_inlrat_1
                    ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_USAC_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['th', 'inlratMin'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_and_th_for_inlrat_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  fig_type='surface',
                                                  use_marks=True,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 36:
                    fig_title_pre_str = 'Values of Inlier Ratio Differences for USAC Option Combinations of '
                    eval_columns = ['inlRat_estimated', 'inlRat_GT']
                    units = [('inlRat_diff', '')]
                    # it_parameters = ['USAC_parameters_automaticSprtInit',
                    #                  'USAC_parameters_automaticProsacParameters',
                    #                  'USAC_parameters_prevalidateSample',
                    #                  'USAC_parameters_USACInlratFilt']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'USAC_opt_search_min_inlrat_diff'}
                    from usac_eval import get_inlrat_diff, get_min_inlrat_diff
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_USAC_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='inlRat-diff',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['th'],
                                                             filter_func=None,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_min_inlrat_diff,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=get_inlrat_diff,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 11:
                    fig_title_pre_str = 'Values of R\\&t Differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['USAC_parameters_automaticSprtInit',
                    #                  'USAC_parameters_automaticProsacParameters',
                    #                  'USAC_parameters_prevalidateSample',
                    #                  'USAC_parameters_USACInlratFilt']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpDistr', 'th']  # th must be at the end
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True}
                    from usac_eval import get_best_comb_th_scenes_1
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_USAC_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratMin'],
                                                             filter_func=None,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_best_comb_th_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 12:
                    fig_title_pre_str = 'Execution Times for USAC Option Combinations of '
                    eval_columns = ['robEstimationAndRef_us']
                    units = []
                    # it_parameters = ['USAC_parameters_automaticSprtInit',
                    #                  'USAC_parameters_automaticProsacParameters',
                    #                  'USAC_parameters_prevalidateSample',
                    #                  'USAC_parameters_USACInlratFilt']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'smooth',
                                          'nr_target_kps': 1000,
                                          't_data_separators': ['inlratMin'],
                                          'res_par_name': 'USAC_opt_search_min_time'}
                    from usac_eval import filter_nr_kps, calc_Time_Model, estimate_alg_time_fixed_kp
                    ret += calcFromFuncAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_USAC_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='time',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['nrCorrs_GT'],
                                                  filter_func=filter_nr_kps,
                                                  filter_func_args=None,
                                                  special_calcs_func=estimate_alg_time_fixed_kp,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=calc_Time_Model,
                                                  calc_func_args={'data_separators': ['inlratMin', 'th']},
                                                  fig_type='surface',
                                                  use_marks=True,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 13:
                    fig_title_pre_str = 'Execution Times for USAC Option Combinations of '
                    eval_columns = ['robEstimationAndRef_us']
                    units = []
                    # it_parameters = ['USAC_parameters_automaticSprtInit',
                    #                  'USAC_parameters_automaticProsacParameters',
                    #                  'USAC_parameters_prevalidateSample',
                    #                  'USAC_parameters_USACInlratFilt']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    from usac_eval import filter_nr_kps, calc_Time_Model, estimate_alg_time_fixed_kp_for_props
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'smooth',
                                          'nr_target_kps': 1000,
                                          't_data_separators': ['inlratMin', 'th'],
                                          'res_par_name': 'USAC_opt_search_min_time_inlrat_th'}
                    ret += calcFromFuncAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_USAC_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='time',
                                                             eval_columns=eval_columns,  # Column names for which statistics are calculated (y-axis)
                                                             units=units,  # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,  # Algorithm parameters to evaluate
                                                             partitions=['th'],  # Data properties to calculate results separately
                                                             x_axis_column=['nrCorrs_GT'],  # x-axis column name
                                                             filter_func=filter_nr_kps,
                                                             filter_func_args=None,
                                                             special_calcs_func=estimate_alg_time_fixed_kp_for_props,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_Time_Model,
                                                             calc_func_args={'data_separators': ['inlratMin', 'th']},
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=False)
                elif ev == 14:
                    fig_title_pre_str = 'Execution Times for USAC Option Combinations of '
                    eval_columns = ['robEstimationAndRef_us']
                    units = []
                    # it_parameters = ['USAC_parameters_automaticSprtInit',
                    #                  'USAC_parameters_automaticProsacParameters',
                    #                  'USAC_parameters_prevalidateSample',
                    #                  'USAC_parameters_USACInlratFilt']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    from usac_eval import filter_nr_kps, calc_Time_Model, estimate_alg_time_fixed_kp_for_3_props
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': (True, False),
                                          'fig_type': ('surface', 'xbar'),
                                          'nr_target_kps': 1000,
                                          't_data_separators': ['kpAccSd', 'inlratMin', 'th'],
                                          'accum_step_props': ['inlratMin', 'kpAccSd'],
                                          'eval_minmax_for': 'th',
                                          'res_par_name': 'USAC_opt_search_min_time_kpAccSd_inlrat_th'}
                    ret += calcFromFuncAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_USAC_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='time',
                                                             eval_columns=eval_columns,  # Column names for which statistics are calculated (y-axis)
                                                             units=units,  # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,  # Algorithm parameters to evaluate
                                                             partitions=['th'],  # Data properties to calculate results separately
                                                             xy_axis_columns=['nrCorrs_GT'],  # x-axis column name
                                                             filter_func=filter_nr_kps,
                                                             filter_func_args=None,
                                                             special_calcs_func=estimate_alg_time_fixed_kp_for_3_props,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_Time_Model,
                                                             calc_func_args={'data_separators': ['kpAccSd', 'inlratMin', 'th']},
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        else:
            raise ValueError('Test nr does not exist')
    elif test_name == 'usac_vs_ransac':
        if eval_nr[0] < 0:
            evals = list(range(1, 8))
        else:
            evals = eval_nr
        for ev in evals:
            if ev == 1:
                fig_title_pre_str = 'Statistics on R\\&t Differences for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                # it_parameters = ['RobMethod']
                it_parameters = ['USAC_parameters_estimator']
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': True,
                                      'res_par_name': 'USAC_vs_RANSAC_th'}
                from usac_eval import get_best_comb_and_th_1
                ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_USAC_vs_RANSAC_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='RT-stats',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              x_axis_column=['th'],
                                              pdfsplitentry=['t_distDiff'],
                                              filter_func=None,
                                              filter_func_args=None,
                                              special_calcs_func=get_best_comb_and_th_1,
                                              special_calcs_args=special_calcs_args,
                                              calc_func=None,
                                              calc_func_args=None,
                                              compare_source=None,
                                              fig_type='smooth',
                                              use_marks=True,
                                              ctrl_fig_size=True,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=True)
            elif ev == 2:
                fig_title_pre_str = 'Statistics on R\\&t Differences for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                # it_parameters = ['RobMethod']
                it_parameters = ['USAC_parameters_estimator']
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': True,
                                      'res_par_name': 'USAC_vs_RANSAC_inlrat'}
                from usac_eval import get_best_comb_inlrat_1
                ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_USAC_vs_RANSAC_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='RT-stats',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              x_axis_column=['inlratMin'],
                                              pdfsplitentry=['t_distDiff'],
                                              filter_func=None,
                                              filter_func_args=None,
                                              special_calcs_func=get_best_comb_inlrat_1,
                                              special_calcs_args=special_calcs_args,
                                              calc_func=None,
                                              calc_func_args=None,
                                              compare_source=None,
                                              fig_type='smooth',
                                              use_marks=True,
                                              ctrl_fig_size=True,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=True)
            elif ev == 3:
                fig_title_pre_str = 'Values of R\\&t Differences for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                # it_parameters = ['RobMethod']
                it_parameters = ['USAC_parameters_estimator']
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': True,
                                      'fig_type': 'surface',
                                      'res_par_name': 'USAC_vs_RANSAC_inlrat_th'}
                from usac_eval import get_best_comb_and_th_for_inlrat_1
                ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_USAC_vs_RANSAC_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='RT-stats',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              xy_axis_columns=['th', 'inlratMin'],
                                              filter_func=None,
                                              filter_func_args=None,
                                              special_calcs_func=get_best_comb_and_th_for_inlrat_1,
                                              special_calcs_args=special_calcs_args,
                                              calc_func=None,
                                              calc_func_args=None,
                                              fig_type='surface',
                                              use_marks=True,
                                              ctrl_fig_size=False,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=True)
            elif ev == 7:
                fig_title_pre_str = 'Values of Inlier Ratio Differences for Comparison of '
                eval_columns = ['inlRat_estimated', 'inlRat_GT']
                units = [('inlRat_diff', '')]
                # it_parameters = ['RobMethod']
                it_parameters = ['USAC_parameters_estimator']
                # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                partitions = ['depthDistr', 'kpAccSd']
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': True,
                                      'res_par_name': 'USAC_vs_RANSAC_min_inlrat_diff'}
                from usac_eval import get_inlrat_diff, get_min_inlrat_diff
                ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_USAC_vs_RANSAC_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='inlRat-diff',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         partitions=partitions,
                                                         x_axis_column=['th'],
                                                         filter_func=None,
                                                         filter_func_args=None,
                                                         special_calcs_func=get_min_inlrat_diff,
                                                         special_calcs_args=special_calcs_args,
                                                         calc_func=get_inlrat_diff,
                                                         calc_func_args=None,
                                                         compare_source=None,
                                                         fig_type='smooth',
                                                         use_marks=True,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=True)
            elif ev == 4:
                fig_title_pre_str = 'Values of R\\&t Differences for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                # it_parameters = ['RobMethod']
                it_parameters = ['USAC_parameters_estimator']
                # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                partitions = ['depthDistr', 'kpAccSd', 'th']#th must be at the end
                special_calcs_args = {'build_pdf': (True, True), 'use_marks': True}
                from usac_eval import get_best_comb_th_scenes_1
                ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_USAC_vs_RANSAC_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='RT-stats',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         partitions=partitions,
                                                         x_axis_column=['inlratMin'],
                                                         filter_func=None,
                                                         filter_func_args=None,
                                                         special_calcs_func=get_best_comb_th_scenes_1,
                                                         special_calcs_args=special_calcs_args,
                                                         calc_func=None,
                                                         calc_func_args=None,
                                                         compare_source=None,
                                                         fig_type='smooth',
                                                         use_marks=True,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=True)
            elif ev == 5:
                fig_title_pre_str = 'Execution Times for Comparison of '
                eval_columns = ['robEstimationAndRef_us']
                units = []
                # it_parameters = ['RobMethod']
                it_parameters = ['USAC_parameters_estimator']
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': True,
                                      'fig_type': 'smooth',
                                      'nr_target_kps': 1000,
                                      't_data_separators': ['inlratMin'],
                                      'res_par_name': 'USAC_vs_RANSAC_min_time'}
                from usac_eval import filter_nr_kps, calc_Time_Model, estimate_alg_time_fixed_kp
                ret += calcFromFuncAndPlot_3D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_USAC_vs_RANSAC_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='time',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              xy_axis_columns=['nrCorrs_GT'],
                                              filter_func=filter_nr_kps,
                                              filter_func_args=None,
                                              special_calcs_func=estimate_alg_time_fixed_kp,
                                              special_calcs_args=special_calcs_args,
                                              calc_func=calc_Time_Model,
                                              calc_func_args={'data_separators': ['inlratMin', 'th']},
                                              fig_type='surface',
                                              use_marks=True,
                                              ctrl_fig_size=False,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=True)
            elif ev == 6:
                fig_title_pre_str = 'Execution Times for Comparison of '
                eval_columns = ['robEstimationAndRef_us']
                units = []
                # it_parameters = ['RobMethod']
                it_parameters = ['USAC_parameters_estimator']
                from usac_eval import filter_nr_kps, calc_Time_Model
                ret += calcFromFuncAndPlot_2D_partitions(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_USAC_vs_RANSAC_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='time',
                                                         eval_columns=eval_columns,  # Column names for which statistics are calculated (y-axis)
                                                         units=units,  # Units in string format for every entry of eval_columns
                                                         it_parameters=it_parameters,  # Algorithm parameters to evaluate
                                                         partitions=['th'],  # Data properties to calculate results separately
                                                         x_axis_column=['nrCorrs_GT'],  # x-axis column name
                                                         filter_func=filter_nr_kps,
                                                         filter_func_args=None,
                                                         special_calcs_func=None,
                                                         special_calcs_args=None,
                                                         calc_func=calc_Time_Model,
                                                         calc_func_args={'data_separators': ['inlratMin', 'th']},
                                                         compare_source=None,
                                                         fig_type='smooth',
                                                         use_marks=True,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=False)
            else:
                raise ValueError('Eval nr ' + ev + ' does not exist')
    elif test_name == 'refinement_ba':
        if not test_nr:
            raise ValueError('test_nr is required refinement_ba')
        if test_nr == 1:
            if eval_nr[0] < 0:
                evals = list(range(1, 4))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 1:
                    fig_title_pre_str = 'Statistics on R\\&t Differences for Different  '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['refineMethod_algorithm',
                    #                  'refineMethod_costFunction',
                    #                  'BART']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refineRT_BA_opts_inlrat'}
                    from usac_eval import get_best_comb_inlrat_1
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_refineRT_BA_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratMin'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_inlrat_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 2:
                    fig_title_pre_str = 'Values of R\\&t Differences for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['refineMethod_algorithm',
                    #                  'refineMethod_costFunction',
                    #                  'BART']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refinement_ba_best_comb_scenes'}
                    from refinement_eval import get_best_comb_scenes_1
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_refineRT_BA_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratMin'],
                                                             filter_func=None,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_best_comb_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 3:
                    fig_title_pre_str = 'Execution Times for Different '
                    eval_columns = ['linRef_BA_us']
                    units = []
                    # it_parameters = ['refineMethod_algorithm',
                    #                  'refineMethod_costFunction',
                    #                  'BART']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'nr_target_kps': 1000,
                                          'res_par_name': 'refineRT_BA_min_time'}
                    from usac_eval import calc_Time_Model
                    from refinement_eval import filter_nr_kps_calc_t, estimate_alg_time_fixed_kp_agg
                    ret += calcFromFuncAndPlot_aggregate(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_refineRT_BA_opts_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='time',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         x_axis_column=['nrCorrs_GT'],
                                                         filter_func=filter_nr_kps_calc_t,
                                                         filter_func_args=None,
                                                         special_calcs_func=estimate_alg_time_fixed_kp_agg,
                                                         special_calcs_args=special_calcs_args,
                                                         calc_func=calc_Time_Model,
                                                         calc_func_args={'data_separators': []},
                                                         compare_source=None,
                                                         fig_type='ybar',
                                                         use_marks=True,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=False)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        elif test_nr == 2:
            if eval_nr[0] < 0:
                evals = list(range(1, 5))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 1:
                    fig_title_pre_str = 'Statistics on R\\&t Differences After Bundle Adjustment (BA) Including ' \
                                        'Intrinsics and Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['refineMethod_algorithm',
                    #                  'refineMethod_costFunction']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refineRT_opts_for_BA2_inlrat'}
                    from usac_eval import get_best_comb_inlrat_1
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_refineRT_BA_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratMin'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_inlrat_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 2:
                    fig_title_pre_str = 'Values of R\\&t Differences After Bundle Adjustment (BA) Including ' \
                                        'Intrinsics and Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['refineMethod_algorithm',
                    #                  'refineMethod_costFunction']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refinement_best_comb_for_BA2_scenes'}
                    from refinement_eval import get_best_comb_scenes_1
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_refineRT_BA_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratMin'],
                                                             filter_func=None,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_best_comb_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 3:
                    fig_title_pre_str = 'Statistics on Focal Length and Principal Point Differences ' \
                                        'after Bundle Adjustment (BA) Including Intrinsics and ' \
                                        'Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['K1_cxyfxfyNorm', 'K2_cxyfxfyNorm', 'K1_cxyDiffNorm', 'K2_cxyDiffNorm',
                                    'K1_fxyDiffNorm', 'K2_fxyDiffNorm', 'K1_fxDiff', 'K2_fxDiff', 'K1_fyDiff',
                                    'K2_fyDiff', 'K1_cxDiff', 'K2_cxDiff', 'K1_cyDiff', 'K2_cyDiff']
                    units = [('K1_cxyfxfyNorm', '/pixel'), ('K2_cxyfxfyNorm', '/pixel'),
                             ('K1_cxyDiffNorm', '/pixel'), ('K2_cxyDiffNorm', '/pixel'),
                             ('K1_fxyDiffNorm', '/pixel'), ('K2_fxyDiffNorm', '/pixel'), ('K1_fxDiff', '/pixel'),
                             ('K2_fxDiff', '/pixel'), ('K1_fyDiff', '/pixel'), ('K2_fyDiff', '/pixel'),
                             ('K1_cxDiff', '/pixel'), ('K2_cxDiff', '/pixel'), ('K1_cyDiff', '/pixel'),
                             ('K2_cyDiff', '/pixel')]
                    # it_parameters = ['refineMethod_algorithm',
                    #                  'refineMethod_costFunction']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refineRT_opts_for_BA2_K_inlrat'}
                    from refinement_eval import get_best_comb_inlrat_k
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_refineRT_BA_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='K-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratMin'],
                                                  pdfsplitentry=['K1_fxyDiffNorm', 'K1_fyDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_inlrat_k,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 4:
                    fig_title_pre_str = 'Values on Focal Length and Principal Point Differences ' \
                                        'after Bundle Adjustment (BA) Including Intrinsics and ' \
                                        'Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['K1_cxyfxfyNorm', 'K2_cxyfxfyNorm', 'K1_cxyDiffNorm', 'K2_cxyDiffNorm',
                                    'K1_fxyDiffNorm', 'K2_fxyDiffNorm', 'K1_fxDiff', 'K2_fxDiff', 'K1_fyDiff',
                                    'K2_fyDiff', 'K1_cxDiff', 'K2_cxDiff', 'K1_cyDiff', 'K2_cyDiff']
                    units = [('K1_cxyfxfyNorm', '/pixel'), ('K2_cxyfxfyNorm', '/pixel'),
                             ('K1_cxyDiffNorm', '/pixel'), ('K2_cxyDiffNorm', '/pixel'),
                             ('K1_fxyDiffNorm', '/pixel'), ('K2_fxyDiffNorm', '/pixel'), ('K1_fxDiff', '/pixel'),
                             ('K2_fxDiff', '/pixel'), ('K1_fyDiff', '/pixel'), ('K2_fyDiff', '/pixel'),
                             ('K1_cxDiff', '/pixel'), ('K2_cxDiff', '/pixel'), ('K1_cyDiff', '/pixel'),
                             ('K2_cyDiff', '/pixel')]
                    # it_parameters = ['refineMethod_algorithm',
                    #                  'refineMethod_costFunction']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    from refinement_eval import combineK
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'error_function': combineK,
                                          'error_type_text': 'Combined Camera Matrix Errors '
                                                             '$e_{\\mli{K1,2}}$',
                                          'file_name_err_part': 'Kerror',
                                          'error_col_name': 'ke',
                                          'res_par_name': 'refinement_best_comb_for_BA2_K_scenes'}
                    from refinement_eval import get_best_comb_scenes_1
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_refineRT_BA_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='K-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratMin'],
                                                             filter_func=None,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_best_comb_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
    elif test_name == 'vfc_gms_sof':
        if eval_nr[0] < 0:
            evals = list(range(1, 8))
        else:
            evals = eval_nr
        for ev in evals:
            if ev == 1:
                fig_title_pre_str = 'Statistics on R\\&t Differences for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                # it_parameters = ['matchesFilter_refineGMS',
                #                  'matchesFilter_refineVFC',
                #                  'matchesFilter_refineSOF']
                it_parameters = ['USAC_parameters_estimator',
                                 'USAC_parameters_refinealg']
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': True,
                                      'res_par_name': 'vfc_gms_sof_inlrat'}
                from usac_eval import get_best_comb_inlrat_1
                ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_vfc_gms_sof_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='RT-stats',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              x_axis_column=['inlratMin'],
                                              pdfsplitentry=['t_distDiff'],
                                              filter_func=None,
                                              filter_func_args=None,
                                              special_calcs_func=get_best_comb_inlrat_1,
                                              special_calcs_args=special_calcs_args,
                                              calc_func=None,
                                              calc_func_args=None,
                                              compare_source=None,
                                              fig_type='smooth',
                                              use_marks=True,
                                              ctrl_fig_size=True,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=True)
            elif ev == 2:
                fig_title_pre_str = 'Statistics on Inlier Ratio Differences for Comparison of '
                eval_columns = ['inlRat_estimated', 'inlRat_GT']
                units = [('inlRat_diff', '')]
                # it_parameters = ['matchesFilter_refineGMS',
                #                  'matchesFilter_refineVFC',
                #                  'matchesFilter_refineSOF']
                # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                it_parameters = ['USAC_parameters_estimator',
                                 'USAC_parameters_refinealg']
                from usac_eval import get_inlrat_diff#, get_min_inlrat_diff
                ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_vfc_gms_sof_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='inlRat-diff',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              x_axis_column=['kpAccSd'],
                                              pdfsplitentry=None,
                                              filter_func=None,
                                              filter_func_args=None,
                                              special_calcs_func=None,
                                              special_calcs_args=None,
                                              calc_func=get_inlrat_diff,
                                              calc_func_args=None,
                                              compare_source=None,
                                              fig_type='smooth',
                                              use_marks=True,
                                              ctrl_fig_size=True,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=False)
            elif ev == 5:
                fig_title_pre_str = 'Statistics on Inlier Ratio Differences for Comparison of '
                eval_columns = ['inlRat_estimated', 'inlRat_GT']
                units = [('inlRat_diff', '')]
                # it_parameters = ['matchesFilter_refineGMS',
                #                  'matchesFilter_refineVFC',
                #                  'matchesFilter_refineSOF']
                it_parameters = ['USAC_parameters_estimator',
                                 'USAC_parameters_refinealg']
                from usac_eval import get_inlrat_diff  # , get_min_inlrat_diff
                ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_vfc_gms_sof_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='inlRat-diff',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              x_axis_column=['kpDistr'],
                                              pdfsplitentry=None,
                                              filter_func=None,
                                              filter_func_args=None,
                                              special_calcs_func=None,
                                              special_calcs_args=None,
                                              calc_func=get_inlrat_diff,
                                              calc_func_args=None,
                                              compare_source=None,
                                              fig_type='ybar',
                                              use_marks=True,
                                              ctrl_fig_size=True,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=False)
            elif ev == 6:
                fig_title_pre_str = 'Statistics on Inlier Ratio Differences for Comparison of '
                eval_columns = ['inlRat_estimated', 'inlRat_GT']
                units = [('inlRat_diff', '')]
                # it_parameters = ['matchesFilter_refineGMS',
                #                  'matchesFilter_refineVFC',
                #                  'matchesFilter_refineSOF']
                it_parameters = ['USAC_parameters_estimator',
                                 'USAC_parameters_refinealg']
                from usac_eval import get_inlrat_diff  # , get_min_inlrat_diff
                ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_vfc_gms_sof_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='inlRat-diff',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              x_axis_column=['depthDistr'],
                                              pdfsplitentry=None,
                                              filter_func=None,
                                              filter_func_args=None,
                                              special_calcs_func=None,
                                              special_calcs_args=None,
                                              calc_func=get_inlrat_diff,
                                              calc_func_args=None,
                                              compare_source=None,
                                              fig_type='ybar',
                                              use_marks=True,
                                              ctrl_fig_size=True,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=False)
            elif ev == 7:
                fig_title_pre_str = 'Statistics on Inlier Ratio Differences for Comparison of '
                eval_columns = ['inlRat_estimated', 'inlRat_GT']
                units = [('inlRat_diff', '')]
                # it_parameters = ['matchesFilter_refineGMS',
                #                  'matchesFilter_refineVFC',
                #                  'matchesFilter_refineSOF']
                it_parameters = ['USAC_parameters_estimator',
                                 'USAC_parameters_refinealg']
                special_calcs_args = {'res_par_name': 'vfc_gms_sof_min_inlrat_diff',
                                      'err_type': 'inlRatDiff',
                                      'mk_no_folder': True}
                from usac_eval import get_inlrat_diff
                from vfc_gms_sof_eval import get_min_inlrat_diff_no_fig
                ret += calcSatisticAndPlot_aggregate(data=data.copy(deep=True),
                                                     store_path=output_path,
                                                     tex_file_pre_str='plots_vfc_gms_sof_',
                                                     fig_title_pre_str=fig_title_pre_str,
                                                     eval_description_path='inlRat-diff',
                                                     eval_columns=eval_columns,
                                                     units=units,
                                                     it_parameters=it_parameters,
                                                     pdfsplitentry=None,
                                                     filter_func=None,
                                                     filter_func_args=None,
                                                     special_calcs_func=get_min_inlrat_diff_no_fig,
                                                     special_calcs_args=special_calcs_args,
                                                     calc_func=get_inlrat_diff,
                                                     calc_func_args=None,
                                                     compare_source=None,
                                                     fig_type='ybar',
                                                     use_marks=False,
                                                     ctrl_fig_size=True,
                                                     make_fig_index=True,
                                                     build_pdf=True,
                                                     figs_externalize=False)
            elif ev == 3:
                fig_title_pre_str = 'Values of R\\&t Differences for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                # it_parameters = ['matchesFilter_refineGMS',
                #                  'matchesFilter_refineVFC',
                #                  'matchesFilter_refineSOF']
                it_parameters = ['USAC_parameters_estimator',
                                 'USAC_parameters_refinealg']
                # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                partitions = ['kpDistr', 'depthDistr', 'kpAccSd']
                special_calcs_args = {'build_pdf': (True, True, True),
                                      'use_marks': True,
                                      'res_par_name': 'vfc_gms_sof_best_comb_for_scenes'}
                from refinement_eval import get_best_comb_scenes_1
                ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_vfc_gms_sof_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='RT-stats',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         partitions=partitions,
                                                         x_axis_column=['inlratMin'],
                                                         filter_func=None,
                                                         filter_func_args=None,
                                                         special_calcs_func=get_best_comb_scenes_1,
                                                         special_calcs_args=special_calcs_args,
                                                         calc_func=None,
                                                         calc_func_args=None,
                                                         compare_source=None,
                                                         fig_type='smooth',
                                                         use_marks=True,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=True)
            elif ev == 4:
                fig_title_pre_str = 'Execution Times for Comparison of '
                eval_columns = ['filtering_us']
                units = []
                # it_parameters = ['matchesFilter_refineGMS',
                #                  'matchesFilter_refineVFC',
                #                  'matchesFilter_refineSOF']
                it_parameters = ['USAC_parameters_estimator',
                                 'USAC_parameters_refinealg']
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': False,
                                      'nr_target_kps': 1000,
                                      'res_par_name': 'vfc_gms_sof_min_time'}
                from usac_eval import calc_Time_Model, filter_nr_kps
                from refinement_eval import estimate_alg_time_fixed_kp_agg
                ret += calcFromFuncAndPlot_aggregate(data=data.copy(deep=True),
                                                     store_path=output_path,
                                                     tex_file_pre_str='plots_vfc_gms_sof_',
                                                     fig_title_pre_str=fig_title_pre_str,
                                                     eval_description_path='time',
                                                     eval_columns=eval_columns,
                                                     units=units,
                                                     it_parameters=it_parameters,
                                                     x_axis_column=['nrCorrs_GT'],
                                                     filter_func=filter_nr_kps,
                                                     filter_func_args=None,
                                                     special_calcs_func=estimate_alg_time_fixed_kp_agg,
                                                     special_calcs_args=special_calcs_args,
                                                     calc_func=calc_Time_Model,
                                                     calc_func_args={'data_separators': []},
                                                     compare_source=None,
                                                     fig_type='ybar',
                                                     use_marks=True,
                                                     ctrl_fig_size=True,
                                                     make_fig_index=True,
                                                     build_pdf=True,
                                                     figs_externalize=False)
            else:
                raise ValueError('Eval nr ' + ev + ' does not exist')
    elif test_name == 'refinement_ba_stereo':
        if not test_nr:
            raise ValueError('test_nr is required refinement_ba_stereo')
        from eval_tests_main import get_compare_info
        if test_nr == 1:
            if eval_nr[0] < 0:
                evals = list(range(1, 4))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 1:
                    fig_title_pre_str = 'Statistics on R\\&t Differences for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_refineMethod_algorithm',
                    #                  'stereoParameters_refineMethod_costFunction',
                    #                  'stereoParameters_BART']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg',
                                     'USAC_parameters_USACInlratFilt']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refRT_stereo_BA_opts_inlrat'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
                    from usac_eval import get_best_comb_inlrat_1
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_refRT_BA_stereo_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratMin'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_inlrat_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=compare_source,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 2:
                    fig_title_pre_str = 'Values of R\\&t Differences for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_refineMethod_algorithm',
                    #                  'stereoParameters_refineMethod_costFunction',
                    #                  'stereoParameters_BART']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg',
                                     'USAC_parameters_USACInlratFilt']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'ref_stereo_ba_best_comb_scenes'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
                    from refinement_eval import get_best_comb_scenes_1
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_refRT_BA_stereo_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratMin'],
                                                             filter_func=None,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_best_comb_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=compare_source,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 3:
                    fig_title_pre_str = 'Statistics on Execution Times for Comparison of '
                    eval_columns = ['stereoRefine_us']
                    units = [('stereoRefine_us', '/$\\mu s$')]
                    # it_parameters = ['stereoParameters_matchesFilter_refineGMS',
                    #                  'stereoParameters_matchesFilter_refineVFC',
                    #                  'stereoParameters_matchesFilter_refineSOF']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg',
                                     'USAC_parameters_USACInlratFilt']
                    special_calcs_args = {'res_par_name': 'refRT_BA_stereo_min_time',
                                          'err_type': 'min_mean_time',
                                          'mk_no_folder': True}
                    from vfc_gms_sof_eval import get_min_inlrat_diff_no_fig
                    ret += calcSatisticAndPlot_aggregate(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_refRT_BA_stereo_opts_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='time',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         pdfsplitentry=None,
                                                         filter_func=None,
                                                         filter_func_args=None,
                                                         special_calcs_func=get_min_inlrat_diff_no_fig,
                                                         special_calcs_args=special_calcs_args,
                                                         calc_func=None,
                                                         calc_func_args=None,
                                                         compare_source=None,
                                                         fig_type='ybar',
                                                         use_marks=False,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=False)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        elif test_nr == 2:
            if eval_nr[0] < 0:
                evals = list(range(1, 5))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 1:
                    fig_title_pre_str = 'Statistics on R\\&t Differences After Bundle Adjustment (BA) Including ' \
                                        'Intrinsics and Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_refineMethod_algorithm',
                    #                  'stereoParameters_refineMethod_costFunction']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg',
                                     'USAC_parameters_USACInlratFilt']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refRT_stereo_opts_for_BA2_inlrat'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 2, 'RT-stats', descr)
                    from usac_eval import get_best_comb_inlrat_1
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_refRT_BA_stereo_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratMin'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_inlrat_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=compare_source,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 2:
                    fig_title_pre_str = 'Values of R\\&t Differences After Bundle Adjustment (BA) Including ' \
                                        'Intrinsics and Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_refineMethod_algorithm',
                    #                  'stereoParameters_refineMethod_costFunction']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg',
                                     'USAC_parameters_USACInlratFilt']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'ref_stereo_best_comb_for_BA2_scenes'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 2, 'RT-stats', descr)
                    from refinement_eval import get_best_comb_scenes_1
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_refRT_BA_stereo_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratMin'],
                                                             filter_func=None,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_best_comb_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=compare_source,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 3:
                    fig_title_pre_str = 'Statistics on Focal Length and Principal Point Differences ' \
                                        'after Bundle Adjustment (BA) Including Intrinsics and ' \
                                        'Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['K1_cxyfxfyNorm', 'K2_cxyfxfyNorm', 'K1_cxyDiffNorm', 'K2_cxyDiffNorm',
                                    'K1_fxyDiffNorm', 'K2_fxyDiffNorm', 'K1_fxDiff', 'K2_fxDiff', 'K1_fyDiff',
                                    'K2_fyDiff', 'K1_cxDiff', 'K2_cxDiff', 'K1_cyDiff', 'K2_cyDiff']
                    units = [('K1_cxyfxfyNorm', '/pixel'), ('K2_cxyfxfyNorm', '/pixel'),
                             ('K1_cxyDiffNorm', '/pixel'), ('K2_cxyDiffNorm', '/pixel'),
                             ('K1_fxyDiffNorm', '/pixel'), ('K2_fxyDiffNorm', '/pixel'), ('K1_fxDiff', '/pixel'),
                             ('K2_fxDiff', '/pixel'), ('K1_fyDiff', '/pixel'), ('K2_fyDiff', '/pixel'),
                             ('K1_cxDiff', '/pixel'), ('K2_cxDiff', '/pixel'), ('K1_cyDiff', '/pixel'),
                             ('K2_cyDiff', '/pixel')]
                    # it_parameters = ['stereoParameters_refineMethod_algorithm',
                    #                  'stereoParameters_refineMethod_costFunction']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg',
                                     'USAC_parameters_USACInlratFilt']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refRT_stereo_opts_for_BA2_K_inlrat'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 2, 'K-stats', descr)
                    from refinement_eval import get_best_comb_inlrat_k
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_refRT_BA_stereo_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='K-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratMin'],
                                                  pdfsplitentry=['K1_fxyDiffNorm', 'K1_fyDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_inlrat_k,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=compare_source,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 4:
                    fig_title_pre_str = 'Values on Focal Length and Principal Point Differences ' \
                                        'after Bundle Adjustment (BA) Including Intrinsics and ' \
                                        'Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['K1_cxyfxfyNorm', 'K2_cxyfxfyNorm', 'K1_cxyDiffNorm', 'K2_cxyDiffNorm',
                                    'K1_fxyDiffNorm', 'K2_fxyDiffNorm', 'K1_fxDiff', 'K2_fxDiff', 'K1_fyDiff',
                                    'K2_fyDiff', 'K1_cxDiff', 'K2_cxDiff', 'K1_cyDiff', 'K2_cyDiff']
                    units = [('K1_cxyfxfyNorm', '/pixel'), ('K2_cxyfxfyNorm', '/pixel'),
                             ('K1_cxyDiffNorm', '/pixel'), ('K2_cxyDiffNorm', '/pixel'),
                             ('K1_fxyDiffNorm', '/pixel'), ('K2_fxyDiffNorm', '/pixel'), ('K1_fxDiff', '/pixel'),
                             ('K2_fxDiff', '/pixel'), ('K1_fyDiff', '/pixel'), ('K2_fyDiff', '/pixel'),
                             ('K1_cxDiff', '/pixel'), ('K2_cxDiff', '/pixel'), ('K1_cyDiff', '/pixel'),
                             ('K2_cyDiff', '/pixel')]
                    # it_parameters = ['stereoParameters_refineMethod_algorithm',
                    #                  'stereoParameters_refineMethod_costFunction']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg',
                                     'USAC_parameters_USACInlratFilt']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    from refinement_eval import combineK
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'error_function': combineK,
                                          'error_type_text': 'Combined Camera Matrix Errors '
                                                             '$e_{\\mli{K1,2}}$',
                                          'file_name_err_part': 'Kerror',
                                          'error_col_name': 'ke',
                                          'res_par_name': 'ref_stereo_best_comb_for_BA2_K_scenes'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 2, 'K-stats', descr)
                    from refinement_eval import get_best_comb_scenes_1
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_refRT_BA_stereo_opts_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='K-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratMin'],
                                                             filter_func=None,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_best_comb_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=compare_source,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
    elif test_name == 'correspondence_pool':
        if not test_nr:
            raise ValueError('test_nr is required correspondence_pool')
        from eval_tests_main import get_compare_info
        if test_nr == 1:
            if eval_nr[0] < 0:
                evals = list(range(1, 11))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 1:
                    fig_title_pre_str = 'Statistics on R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_maxPoolCorrespondences',
                    #                  'stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_size_pts_dist_inlrat'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
                    from usac_eval import get_best_comb_inlrat_1
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_corrPool_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratMin'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=get_best_comb_inlrat_1,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=compare_source,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 2:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_maxPoolCorrespondences',
                    #                  'stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_size_pts_dist_best_comb_scenes'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
                    from refinement_eval import get_best_comb_scenes_1
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_corrPool_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratMin'],
                                                             filter_func=None,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_best_comb_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=compare_source,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 3:
                    fig_title_pre_str = 'Values of R\\&t Differences over the last 30 out of 150 frames ' \
                                        'for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_maxPoolCorrespondences',
                    #                  'stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_size_pts_dist_end_frames_best_comb_scenes'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
                    from refinement_eval import get_best_comb_scenes_1
                    from corr_pool_eval import filter_take_end_frames
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_corrPool_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats-last-frames',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratMin'],
                                                             filter_func=filter_take_end_frames,
                                                             filter_func_args=None,
                                                             special_calcs_func=get_best_comb_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=compare_source,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 4:
                    fig_title_pre_str = 'R\\&t Differences from Frame to Frame with a maximum correspondence pool ' \
                                        'size of $\\hat{n}_{cp}=40000$ features for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator']
                    calc_func_args = {'data_separators': ['Nr', 'depthDistr', 'kpAccSd', 'inlratMin']}
                    from corr_pool_eval import filter_max_pool_size, calc_rt_diff_frame_to_frame
                    ret += calcFromFuncAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_corrPool_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-diff',
                                                             eval_columns=eval_columns,  # Column names for which statistics are calculated (y-axis)
                                                             units=units,  # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,  # Algorithm parameters to evaluate
                                                             partitions=['depthDistr', 'kpAccSd'],  # Data properties to calculate results separately
                                                             xy_axis_columns=[],  # x-axis column name
                                                             filter_func=filter_max_pool_size,
                                                             filter_func_args=None,
                                                             special_calcs_func=None,
                                                             special_calcs_args=None,
                                                             calc_func=calc_rt_diff_frame_to_frame,
                                                             calc_func_args=calc_func_args,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 5:
                    fig_title_pre_str = 'Differences of R\\&t Differences from Frame to Frame with a maximum ' \
                                        'correspondence pool size of $\\hat{n}_{cp}=40000$ features for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'poolSize']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''), ('poolSize', '')]
                    # it_parameters = ['stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator']
                    calc_func_args = {'data_separators': ['Nr', 'depthDistr', 'kpAccSd', 'inlratMin'],
                                      'keepEval': ['poolSize', 'R_diffAll', 't_angDiff_deg']}
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_size_converge'}
                    from corr_pool_eval import filter_max_pool_size, \
                        calc_rt_diff2_frame_to_frame, \
                        eval_corr_pool_converge
                    ret += calcFromFuncAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_corrPool_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-diff',
                                                             eval_columns=eval_columns,  # Column names for which statistics are calculated (y-axis)
                                                             units=units,  # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,  # Algorithm parameters to evaluate
                                                             partitions=['depthDistr', 'kpAccSd'],  # Data properties to calculate results separately
                                                             xy_axis_columns=[],  # x-axis column name
                                                             filter_func=filter_max_pool_size,
                                                             filter_func_args=None,
                                                             special_calcs_func=eval_corr_pool_converge,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_rt_diff2_frame_to_frame,
                                                             calc_func_args=calc_func_args,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)


    return ret


if __name__ == "__main__":
    main()