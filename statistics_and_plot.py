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
                figs_externalize=False):
    texdf = os.path.join(out_tex_dir, out_tex_file)
    with open(texdf, 'w') as outfile:
        outfile.write(rendered_tex)
    if out_pdf_filen is not None:
        rep_make = 1
        if make_fig_index or figs_externalize:
            rep_make = 2
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
                    print("Child returned", retcode)
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
                           eval_columns,
                           units,
                           it_parameters,
                           x_axis_column,
                           pdfsplitentry,
                           special_calcs_func = None,
                           special_calcs_args = None,
                           calc_func = None,
                           calc_func_args = None,
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
    store_path_sub = os.path.join(store_path, '-'.join(map(str, it_parameters)) + '_vs_' +
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

    if data.empty:
        raise ValueError('No data left after filtering unsuccessful estimations')
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
    #Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            raise ValueError('Expected some arguments')
        calc_func_args['data'] = data
        df = calc_func(**calc_func_args)
    else:
        needed_columns = eval_columns + it_parameters + x_axis_column
        df = data[needed_columns]
    #Group by USAC parameters 5&6 and calculate the statistic
    stats = df.groupby(it_parameters + x_axis_column).describe()
    if special_calcs_func is not None and special_calcs_args is not None:
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
                 'fill_bar': True}
    pdf_nr = 0
    for it in errvalnames:
        if it[-1] != 'count':
            tmp = stats[it[0]].unstack()
            tmp = tmp[it[1]]
            tmp = tmp.T
            #tmp.columns = ['%s%s' % (str(a), '-%s' % str(b) if b is not None else '') for a, b in tmp.columns]
            tmp.columns = ['-'.join(map(str, a)) for a in tmp.columns]
            tmp.columns.name = '-'.join(grp_names[0:-1])
            dataf_name = 'data_' + '_'.join(map(str, it)) + '_vs_' + \
                       str(grp_names[-1]) + '.csv'
            dataf_name = dataf_name.replace('%', 'perc')
            fdataf_name = os.path.join(tdata_folder, dataf_name)
            with open(fdataf_name, 'a') as f:
                f.write('# ' + str(it[-1]) + ' values for ' + str(it[0]) + '\n')
                f.write('# Column parameters: ' + '-'.join(grp_names[0:-1]) + '\n')
                tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

            #Construct tex-file
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
            if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
                use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
            if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
                use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
            reltex_name = os.path.join(rel_data_path, dataf_name)
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': replace_stat_names(it[-1]) + ' values for ' +
                                                  replaceCSVLabels(str(it[0]), True) +
                                                  ' compared to ' + replaceCSVLabels(str(grp_names[-1]), True),
                                          'fig_type': fig_type,
                                          'plots': list(tmp.columns.values),
                                          'label_y': replace_stat_names(it[-1]) + findUnit(str(it[0]), units),
                                          'plot_x': str(grp_names[-1]),
                                          'label_x': replaceCSVLabels(str(grp_names[-1])),
                                          'limits': use_limits,
                                          'legend': [tex_string_coding_style(a) for a in list(tmp.columns.values)],
                                          'legend_cols': None,
                                          'use_marks': use_marks,
                                          'pdf_nr': pdf_nr
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    template = ji_env.get_template('usac-testing_2D_plots.tex')
    #Get number of pdfs to generate
    pdf_nr = tex_infos['sections'][-1]['pdf_nr']
    if pdf_nr == 0:
        rendered_tex = template.render(title=tex_infos['title'],
                                       make_index=tex_infos['make_index'],
                                       ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                       figs_externalize=tex_infos['figs_externalize'],
                                       fill_bar=tex_infos['fill_bar'],
                                       sections=tex_infos['sections'])
        texf_name = base_out_name + '.tex'
        if build_pdf:
            pdf_name = base_out_name + '.pdf'
            res = compile_tex(rendered_tex,
                              tex_folder,
                              texf_name,
                              make_fig_index,
                              os.path.join(pdf_folder, pdf_name),
                              tex_infos['figs_externalize'])
        else:
            res = compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)
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
                                           sections=it)
            texf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.tex'
            if build_pdf:
                pdf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.pdf'
                res = compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index,
                                  os.path.join(pdf_folder, pdf_name), tex_infos['figs_externalize'])
            else:
                res = compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)
    return res


def calcSatisticAndPlot_2D_partitions(data,
                                      store_path,
                                      tex_file_pre_str,
                                      fig_title_pre_str,
                                      eval_columns,#Column names for which statistics are calculated (y-axis)
                                      units,# Units in string format for every entry of eval_columns
                                      it_parameters,# Algorithm parameters to evaluate
                                      partitions,# Data properties to calculate statistics seperately
                                      x_axis_column,# x-axis column name
                                      pdfsplitentry,# One or more column names present in eval_columns for splitting pdf
                                      special_calcs_func = None,
                                      special_calcs_args = None,
                                      calc_func = None,
                                      calc_func_args = None,
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
    store_path_sub = os.path.join(store_path, '-'.join(map(str, it_parameters)) + '_vs_' +
                                              '-'.join(map(str, x_axis_column)) + '_for_' +
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

    if data.empty:
        raise ValueError('No data left after filtering unsuccessful estimations')
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
    #Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            raise ValueError('Expected some arguments')
        calc_func_args['data'] = data
        df = calc_func(**calc_func_args)
    else:
        needed_columns = eval_columns + it_parameters + x_axis_column + partitions
        df = data[needed_columns]
    #Group by USAC parameters 5&6 and calculate the statistic
    stats = df.groupby(partitions + it_parameters + x_axis_column).describe()
    if special_calcs_func is not None and special_calcs_args is not None:
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
    base_out_name += '_combs_vs_' + grp_names[-1] + '_for_' + \
                     '-'.join([a[:min(3, len(a))] for a in map(str, partitions)])
    title_name += ' Compared to ' + replaceCSVLabels(grp_names[-1], False, True) + ' Values separately for '
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
                 'fill_bar': True}
    pdf_nr = 0
    for it in errvalnames:
        if it[-1] != 'count':
            tmp = stats[it[0]].unstack()
            tmp = tmp[it[1]]
            tmp1 = tmp.reset_index().set_index(partitions)
            for p in tmp1.index:
                tmp2 = tmp1.loc[p]
                part_name = '_'.join([str(ni) + '-' + str(vi) for ni, vi in zip(tmp2.index.names, tmp2.index[0])])
                tmp2 = tmp2.reset_index().drop(partitions, axis=1)
                tmp2 = tmp2.set_index(it_parameters).T
                tmp2.columns = ['-'.join(map(str, a)) for a in tmp2.columns]
                tmp2.columns.name = '-'.join(it_parameters)
                dataf_name = 'data_' + '_'.join(map(str, it)) + '_vs_' + \
                             str(grp_names[-1]) + '_for_' + part_name + '.csv'
                dataf_name = dataf_name.replace('%', 'perc')
                fdataf_name = os.path.join(tdata_folder, dataf_name)
                with open(fdataf_name, 'a') as f:
                    f.write('# ' + str(it[-1]) + ' values for ' + str(it[0]) + ' and scene ' + part_name + '\n')
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
                if stats_all['min'][0] < (stats_all['mean'][0] - stats_all['std'][0] * 2.576):
                    use_limits['miny'] = round(stats_all['mean'][0] - stats_all['std'][0] * 2.576, 6)
                if stats_all['max'][0] > (stats_all['mean'][0] + stats_all['std'][0] * 2.576):
                    use_limits['maxy'] = round(stats_all['mean'][0] + stats_all['std'][0] * 2.576, 6)
                reltex_name = os.path.join(rel_data_path, dataf_name)
                tex_infos['sections'].append({'file': reltex_name,
                                              'name': replace_stat_names(it[-1]) + ' values for ' +
                                                      replaceCSVLabels(str(it[0]), True) +
                                                      ' compared to ' + replaceCSVLabels(str(grp_names[-1]), True) +
                                                      ' for different ',
                                              'fig_type': fig_type,
                                              'plots': list(tmp2.columns.values),
                                              'label_y': replace_stat_names(it[-1]) + findUnit(str(it[0]), units),
                                              'plot_x': str(grp_names[-1]),
                                              'label_x': replaceCSVLabels(str(grp_names[-1])),
                                              'limits': use_limits,
                                              'legend': [tex_string_coding_style(a) for a in list(tmp2.columns.values)],
                                              'legend_cols': None,
                                              'use_marks': use_marks
                                              })
                tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    template = ji_env.get_template('usac-testing_2D_plots.tex')
    #Get number of pdfs to generate
    pdf_nr = tex_infos['sections'][-1]['pdf_nr']
    if pdf_nr == 0:
        rendered_tex = template.render(title=tex_infos['title'],
                                       make_index=tex_infos['make_index'],
                                       ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                       figs_externalize=tex_infos['figs_externalize'],
                                       fill_bar=tex_infos['fill_bar'],
                                       sections=tex_infos['sections'])
        texf_name = base_out_name + '.tex'
        if build_pdf:
            pdf_name = base_out_name + '.pdf'
            res = compile_tex(rendered_tex,
                              tex_folder,
                              texf_name,
                              make_fig_index,
                              os.path.join(pdf_folder, pdf_name),
                              tex_infos['figs_externalize'])
        else:
            res = compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)
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
                                           sections=it)
            texf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.tex'
            if build_pdf:
                pdf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.pdf'
                res = compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index,
                                  os.path.join(pdf_folder, pdf_name), tex_infos['figs_externalize'])
            else:
                res = compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)
    return res


def calcSatisticAndPlot_3D(data,
                           store_path,
                           tex_file_pre_str,
                           fig_title_pre_str,
                           eval_columns,
                           units,
                           it_parameters,
                           xy_axis_columns,
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
    store_path_sub = os.path.join(store_path, '-'.join(map(str, it_parameters)) + '_vs_' +
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

    if data.empty:
        raise ValueError('No data left after filtering unsuccessful estimations')
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
    #Select columns we need
    if calc_func is not None:
        if calc_func_args is None:
            raise ValueError('Expected some arguments')
        calc_func_args['data'] = data
        df = calc_func(**calc_func_args)
    else:
        needed_columns = eval_columns + it_parameters + xy_axis_columns
        df = data[needed_columns]
    #Group by USAC parameters 5&6 and calculate the statistic
    stats = df.groupby(it_parameters + xy_axis_columns).describe()
    if special_calcs_func is not None and special_calcs_args is not None:
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
    title_name += ' compared to ' + replaceCSVLabels(grp_names[-2], False, True) + ' and ' + \
                  replaceCSVLabels(grp_names[-1], False, True) + ' Values'
    tex_infos = {'title': title_name,
                 'sections': [],
                 'make_index': make_fig_index,#Builds an index with hyperrefs on the beginning of the pdf
                 'ctrl_fig_size': ctrl_fig_size}#If True, the figures are adapted to the page height if they are too big
    # Get names of statistics
    stat_names = list(dict.fromkeys([i[-1] for i in errvalnames if i[-1] != 'count']))
    for it in errvalnames:
        if it[-1] != 'count':
            tmp = stats[it[0]].unstack()
            tmp = tmp[it[1]]
            tmp = tmp.unstack()
            tmp = tmp.T
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
                f.write('# Column parameters: ' + '-'.join(grp_names[0:-2]) + '\n')
                tmp.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

            #Construct tex-file information
            stats_all = tmp.stack().reset_index()
            stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).describe().T
            if (np.isclose(stats_all['min'][0], 0, atol=1e-06) and
                np.isclose(stats_all['max'][0], 0, atol=1e-06)) or \
                    np.isclose(stats_all['min'][0], stats_all['max'][0]):
                continue
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
                                          'label_z': replace_stat_names(it[-1]) + findUnit(str(it[0]), units),
                                          'plot_x': str(tmp.columns.values[1]),
                                          'label_x': replace_stat_names(str(tmp.columns.values[1])) +
                                                     findUnit(str(tmp.columns.values[1]), units),
                                          'plot_y': str(tmp.columns.values[0]),
                                          'label_y': replace_stat_names(str(tmp.columns.values[0])) +
                                                     findUnit(str(tmp.columns.values[0]), units),
                                          'legend': [tex_string_coding_style(a) for a in list(tmp.columns.values)[2:]],
                                          'use_marks': use_marks,
                                          'mesh_cols': nr_equal_ss
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
                              'ctrl_fig_size': tex_infos['ctrl_fig_size']})

    # endt = time.time()
    # print(endt - startt)

    template = ji_env.get_template('usac-testing_3D_plots.tex')
    res = 0
    for it in pdfs_info:
        rendered_tex = template.render(title=it['title'],
                                       make_index=it['make_index'],
                                       ctrl_fig_size=it['ctrl_fig_size'],
                                       figs_externalize=it['figs_externalize'],
                                       sections=it['sections'])
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
    elif name == '75%':
        if for_tex:
            return r'75\% percentile'
        else:
            return r'75perc percentile'
    else:
        if for_tex:
            return str(name).replace('%', '\%').capitalize()
        else:
            return str(name).replace('%', 'perc').capitalize()


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
        return '$\\angle{\\Delta t}$'
    elif label == 't_distDiff':
        return '$\\|\\Delta t\\|$'
    elif label == 't_diff_tx':
        return '$\\Delta t_{x}$'
    elif label == 't_diff_ty':
        return '$\\Delta t_{y}$'
    elif label == 't_diff_tz':
        return '$\\Delta t_{z}$'
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
        return '$\\angle{\\Delta \\hat{t}}$'
    elif label == 't_mostLikely_distDiff':
        return '$\\|\\Delta \\hat{t}\\|$'
    elif label == 't_mostLikely_diff_tx':
        return '$\\Delta \\hat{t}_{x}$'
    elif label == 't_mostLikely_diff_ty':
        return '$\\Delta \\hat{t}_{y}$'
    elif label == 't_mostLikely_diff_tz':
        return '$\\Delta \\hat{t}_{z}$'
    elif label == 'K1_fxDiff':
        return '$\\Delta f_{x}^{K1}$'
    elif label == 'K1_fyDiff':
        return '$\\Delta f_{y}^{K1}$'
    elif label == 'K1_fxyDiffNorm':
        return '$\\|\\Delta f_{x,y}^{K1}\\|$'
    elif label == 'K1_cxDiff':
        return '$\\Delta c_{x}^{K1}$'
    elif label == 'K1_cyDiff':
        return '$\\Delta c_{y}^{K1}$'
    elif label == 'K1_cxyDiffNorm':
        return '$\\|\\Delta c_{x,y}^{K1}\\|$'
    elif label == 'K1_cxyfxfyNorm':
        return '$\\|\\Delta c_{x,y}^{K1}\\, f_{x,y}^{K1}\\|$'
    elif label == 'K2_fxDiff':
        return '$\\Delta f_{x}^{K2}$'
    elif label == 'K2_fyDiff':
        return '$\\Delta f_{y}^{K2}$'
    elif label == 'K2_fxyDiffNorm':
        return '$\\|\\Delta f_{x,y}^{K2}\\|$'
    elif label == 'K2_cxDiff':
        return '$\\Delta c_{x}^{K2}$'
    elif label == 'K2_cyDiff':
        return '$\\Delta c_{y}^{K2}$'
    elif label == 'K2_cxyDiffNorm':
        return '$\\|\\Delta c_{x,y}^{K2}\\|$'
    elif label == 'K2_cxyfxfyNorm':
        return '$\\|\\Delta c_{x,y}^{K2}\\, f_{x,y}^{K2}\\|$'
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
    elif label == 'USAC_parameters_noAutomaticProsacParamters':
        str_val = 'disabled automatic PROSAC parameter estimation'
    elif label == 'USAC_parameters_prevalidateSample':
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
            str_val = 'use RANSAC for few matches'
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
            str_val = 'minimum distances $\\|\\breve{d}_{cm}\\|$ for stability'
        else:
            str_val = 'minimum distance $\\|\\breve{d}_{cm}\\|$ for stability'
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
    ex = ['and', 'of', 'for', 'to', 'with']
    if str_capitalize:
        return ' '.join([b.capitalize() if not b.isupper() and
                                           not '$' in b and
                                           not '\\' in b and
                                           not b in ex else b for b in str_val.split(' ')])
    else:
        return str_val

#Only for testing
def main():
    num_pts = int(8000)
    pars1_opt = ['first_long_long_opt' + str(i) for i in range(0,3)]
    pars2_opt = ['second_long_opt' + str(i) for i in range(0, 3)]
    pars3_opt = ['third_long_long_opt' + str(i) for i in range(0, 2)]
    pars_kpDistr_opt = ['1corn', 'equ']
    pars_depthDistr_opt = ['NMF', 'NM', 'F']
    pars_nrTP_opt = ['500', '100to1000']
    pars_kpAccSd_opt = ['0.5', '1.0', '1.5']
    data = {'R_diffAll': 1000 + np.abs(np.random.randn(num_pts) * 10),#[0.3, 0.5, 0.7, 0.4, 0.6] * int(num_pts/5),
            'R_diff_roll_deg': 1000 + np.abs(np.random.randn(num_pts) * 10),
            'R_diff_pitch_deg': 10 + np.random.randn(num_pts) * 5,
            'R_diff_yaw_deg': -1000 + np.abs(np.random.randn(num_pts)),
            't_angDiff_deg': [0.3, 0.5, 0.7, 0.4, 0.6] * int(num_pts/5),
            't_distDiff': np.abs(np.random.randn(num_pts) * 100),
            't_diff_tx': -10000 + np.random.randn(num_pts) * 100,
            't_diff_ty': 20000 + np.random.randn(num_pts),
            't_diff_tz': -450 + np.random.randn(num_pts),
            # 'USAC_parameters_estimator': np.random.randint(0, 3, num_pts),
            # 'USAC_parameters_refinealg': np.random.randint(0, 7, num_pts),
            # 'USAC_parameters_USACInlratFilt': np.random.randint(8, 10, num_pts),
            'USAC_parameters_estimator': [pars1_opt[i] for i in np.random.randint(0, len(pars1_opt), num_pts)],
            'USAC_parameters_refinealg': [pars2_opt[i] for i in np.random.randint(0, len(pars2_opt), num_pts)],
            #'kpDistr': [pars_kpDistr_opt[i] for i in np.random.randint(0, len(pars_kpDistr_opt), num_pts)],
            'depthDistr': [pars_depthDistr_opt[i] for i in np.random.randint(0, len(pars_depthDistr_opt), num_pts)],
            #'nrTP': [pars_nrTP_opt[i] for i in np.random.randint(0, len(pars_nrTP_opt), num_pts)],
            'kpAccSd': [pars_kpAccSd_opt[i] for i in np.random.randint(0, len(pars_kpAccSd_opt), num_pts)],
            #'USAC_parameters_USACInlratFilt': [pars3_opt[i] for i in np.random.randint(0, len(pars3_opt), num_pts)],
            'th': np.tile(np.arange(0.4, 0.9, 0.1), int(num_pts/5)),
            'inlrat': np.tile(np.arange(0.05, 0.45, 0.1), int(num_pts/4)),
            'useless': [1, 1, 2, 3] * int(num_pts/4),
            'R_out(0,0)': [0] * 10 + [1] * int(num_pts - 10),
            'R_out(0,1)': [0] * 10 + [0] * int(num_pts - 10),
            'R_out(0,2)': [float(0)] * 10 + [0.1] * int(num_pts - 10),
            'R_out(1,0)': [0] * 10 + [1] * int(num_pts - 10),
            'R_out(1,1)': [0] * 10 + [1] * int(num_pts - 10),
            'R_out(1,2)': [0] * 10 + [0] * int(num_pts - 10),
            'R_out(2,0)': [float(0)] * 10 + [0.1] * int(num_pts - 10),
            'R_out(2,1)': [0] * 10 + [0] * int(num_pts - 10),
            'R_out(2,2)': [0] * 10 + [0] * int(num_pts - 10)}
    data = pd.DataFrame(data)

    tex_file_pre_str = 'plots_USAC_opts_'
    output_dir = '/home/maierj/work/Sequence_Test/py_test'
    fig_title_pre_str = 'Statistics for USAC Option Combinations of '
    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
             ('t_diff_ty', ''), ('t_diff_tz', '')]
    it_parameters = ['USAC_parameters_estimator', 'USAC_parameters_refinealg']#, 'USAC_parameters_USACInlratFilt']
    x_axis_column = ['th']
    pdfsplitentry = ['t_distDiff']
    from usac_eval import get_best_comb_and_th_1, get_best_comb_inlrat_1, get_best_comb_and_th_for_kpacc_1
    special_calcs_func = None#get_best_comb_and_th_1#get_best_comb_inlrat_1
    special_calcs_args = None#{'build_pdf': (True, True), 'use_marks': True}
    # figure types: sharp plot, smooth, const plot, ybar, xbar
    calc_func = None
    calc_func_args = None
    fig_type = 'ybar'
    use_marks = True
    ctrl_fig_size = True
    make_fig_index = True
    build_pdf = True
    figs_externalize = True
    # calcSatisticAndPlot_2D(data,
    #                        output_dir,
    #                        tex_file_pre_str,
    #                        fig_title_pre_str,
    #                        eval_columns,
    #                        units,
    #                        it_parameters,
    #                        x_axis_column,
    #                        pdfsplitentry,
    #                        special_calcs_func,
    #                        special_calcs_args,
    #                        calc_func,
    #                        calc_func_args,
    #                        fig_type,
    #                        use_marks,
    #                        ctrl_fig_size,
    #                        make_fig_index,
    #                        build_pdf,
    #                        figs_externalize)
    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd']
    partitions = ['depthDistr', 'kpAccSd']
    calcSatisticAndPlot_2D_partitions(data,
                                      output_dir,
                                      tex_file_pre_str,
                                      fig_title_pre_str,
                                      eval_columns,  # Column names for which statistics are calculated (y-axis)
                                      units,  # Units in string format for every entry of eval_columns
                                      it_parameters,  # Algorithm parameters to evaluate
                                      partitions,  # Data properties to calculate statistics seperately
                                      x_axis_column,  # x-axis column name
                                      pdfsplitentry,
                                      # One or more column names present in eval_columns for splitting pdf
                                      special_calcs_func,
                                      special_calcs_args,
                                      calc_func,
                                      calc_func_args,
                                      fig_type,
                                      use_marks,
                                      ctrl_fig_size,
                                      make_fig_index,
                                      build_pdf,
                                      figs_externalize)
    # x_axis_column = ['th', 'inlrat']
    # fig_type = 'surface'
    # fig_title_pre_str = 'Values for USAC Option Combinations of '
    # special_calcs_func = get_best_comb_and_th_for_kpacc_1
    # special_calcs_args = {'build_pdf': (True, True), 'use_marks': True, 'fig_type': 'surface'}
    # calcSatisticAndPlot_3D(data,
    #                        output_dir,
    #                        tex_file_pre_str,
    #                        fig_title_pre_str,
    #                        eval_columns,
    #                        units,
    #                        it_parameters,
    #                        x_axis_column,
    #                        special_calcs_func,
    #                        special_calcs_args,
    #                        calc_func,
    #                        calc_func_args,
    #                        fig_type,
    #                        use_marks,
    #                        ctrl_fig_size,
    #                        make_fig_index,
    #                        build_pdf,
    #                        figs_externalize)


if __name__ == "__main__":
    main()