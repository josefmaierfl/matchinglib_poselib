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
import multiprocessing
from difflib import SequenceMatcher

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
                build_glossary=False,
                nr_cpus=-1):
    mult_proc = False
    if isinstance(out_tex_dir, list):
        raise ValueError('Only a single output directory is supported')
    if isinstance(rendered_tex, list):
        if not isinstance(out_tex_file, list):
            raise ValueError('A list of tex texts is provided but no list of file names.')
        if len(rendered_tex) != len(out_tex_file):
            raise ValueError('Number of tex texts and file names must be equal.')
        if out_pdf_filen is not None and not isinstance(out_pdf_filen, list):
            raise ValueError('A list of tex texts is provided but no list of pdf file names.')
        if len(rendered_tex) > 1 and out_pdf_filen is not None and nr_cpus != 1:
            mult_proc = True
    else:
        rendered_tex = [rendered_tex]
        out_tex_file = [out_tex_file]
        if out_pdf_filen is not None:
            out_pdf_filen = [out_pdf_filen]
    texdf = []
    for tf, rt in zip(out_tex_file, rendered_tex):
        texdf.append(os.path.join(out_tex_dir, tf))
        with open(texdf[-1], 'w') as outfile:
            outfile.write(rt)
    if out_pdf_filen is not None:
        av_cpus = os.cpu_count()
        if av_cpus:
            if nr_cpus < 1:
                cpu_use = av_cpus
            elif nr_cpus > av_cpus:
                print('Demanded ' + str(nr_cpus) + ' but only ' + str(av_cpus) + ' CPUs are available. Using '
                      + str(av_cpus) + ' CPUs.')
                cpu_use = av_cpus
            else:
                cpu_use = nr_cpus
        rep_make = 1
        if make_fig_index or figs_externalize:
            rep_make = 2
        if build_glossary:
            rep_make = 3
        dir_created = False
        pdf_info = {'pdfpath': [],
                    'pdfname': [],
                    'stdoutf': [],
                    'erroutf': [],
                    'cmdline': []}
        for of, td in zip(out_pdf_filen, texdf):
            pdfpath, pdfname = os.path.split(of)
            pdf_info['pdfpath'].append(pdfpath)
            pdfname = os.path.splitext(pdfname)[0]
            pdf_info['pdfname'].append(pdfname)
            stdoutf = os.path.join(pdfpath, 'stdout_' + pdfname + '.txt')
            erroutf = os.path.join(pdfpath, 'error_' + pdfname + '.txt')
            cmdline = ['pdflatex',
                       '--jobname=' + pdfname,
                       '--output-directory=' + pdfpath,
                       '-synctex=1',
                       '--interaction=nonstopmode']
            if figs_externalize:
                cmdline += ['--shell-escape', td]
                if not dir_created:
                    figs = os.path.join(pdfpath, 'figures')
                    try:
                        os.mkdir(figs)
                    except FileExistsError:
                        print('Folder', figs, 'for storing temp images already exists')
                    except:
                        print("Unexpected error (Unable to create directory for storing temp images):",
                              sys.exc_info()[0])
                    try:
                        os.symlink(figs, os.path.join(out_tex_dir, 'figures'))
                    except OSError:
                        print('Unable to create a symlink to the stored images')
                    except:
                        print("Unexpected error (Unable to create directory for storing temp images):",
                              sys.exc_info()[0])
                    dir_created = True
            else:
                cmdline += [td]
            pdf_info['cmdline'].append(cmdline)
            pdf_info['stdoutf'].append(stdoutf)
            pdf_info['erroutf'].append(erroutf)

        if mult_proc:
            tasks = [(pdf_info['pdfpath'][it],
                      pdf_info['pdfname'][it],
                      pdf_info['cmdline'][it],
                      pdf_info['stdoutf'][it],
                      pdf_info['erroutf'][it],
                      rep_make,
                      out_tex_dir)
                     for it in range(0, len(pdf_info['pdfpath']))]
            retcode_mp = []
            with multiprocessing.Pool(processes=cpu_use) as pool:
                results = [pool.apply_async(compile_pdf_base, t) for t in tasks]
                for r in results:
                    cnt_dot = 0
                    while 1:
                        sys.stdout.flush()
                        try:
                            res = r.get(2.0)
                            break
                        except multiprocessing.TimeoutError:
                            if cnt_dot >= 90:
                                print()
                                cnt_dot = 0
                            sys.stdout.write('.')
                            cnt_dot = cnt_dot + 1
                        except:
                            res = 1
                            break
                    retcode_mp.append(res)
            retcode = sum(retcode_mp)
        else:
            retcode = 0
            for it in range(0, len(pdf_info['pdfpath'])):
                retcode += compile_pdf_base(pdf_info['pdfpath'][it],
                      pdf_info['pdfname'][it],
                      pdf_info['cmdline'][it],
                      pdf_info['stdoutf'][it],
                      pdf_info['erroutf'][it],
                      rep_make,
                      out_tex_dir)
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
        return retcode
    return 0


def compile_pdf_base(pdfpath, pdfname, cmdline, stdoutf, erroutf, rep_make_in, out_tex_dir):
    stdoutfh = open(stdoutf, 'w')
    erroutfh = open(erroutf, 'w')
    retcode = 0
    rep_make = rep_make_in
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
                           figs_externalize=True,
                           no_tex=False):
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

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + short_concat_str(it_parameters) + '_vs_' +
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
                                                   short_concat_str(compare_source['it_parameters']) + '_vs_' +
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
    base_out_name += concat_combs(grp_names, nr_it_parameters)
    for i in range(0, nr_it_parameters):
        title_name += replaceCSVLabels(grp_names[i], True, True, True)
        if(nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_name += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_name += ', '
            elif i < nr_it_parameters - 1:
                title_name += ', and '
    base_out_name += '_combs_vs_' + grp_names[-1]
    title_name += ' Compared to ' + replaceCSVLabels(grp_names[-1], False, True, True) + ' Values'
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
            it_tmp = list(it)
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
                if compare_source['replace_evals']:
                    succe, dst_eval, new_name = get_replace_eval(compare_source, str(it[0]))
                    if succe:
                        datafc_name = dataf_name.replace(str(it[0]), dst_eval)
                        dataf_name = dataf_name.replace(str(it[0]), new_name)
                        tmp.rename(columns={it[0]: new_name}, inplace=True)
                        it_tmp[0] = new_name
                    else:
                        datafc_name = dataf_name
                    _, dst_eval, _ = get_replace_eval(compare_source, grp_names[-1], True)
                    datafc_name = datafc_name.replace(str(grp_names[-1]), str(dst_eval))
                else:
                    datafc_name = dataf_name
                tmp, succ = add_comparison_column(compare_source, datafc_name, tmp)
            fdataf_name = os.path.join(tdata_folder, dataf_name)
            with open(fdataf_name, 'a') as f:
                f.write('# ' + str(it_tmp[-1]) + ' values for ' + str(it_tmp[0]) + '\n')
                f.write('# Column parameters: ' + '-'.join(it_parameters) + '\n')
                tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

            if no_tex:
                continue

            #Construct tex-file
            if pdfsplitentry:
                if pdf_nr < len(pdfsplitentry):
                    if pdfsplitentry[pdf_nr] == str(it_tmp[0]):
                        pdf_nr += 1
            useless, use_limits, use_log, exp_value = get_limits_log_exp(tmp, False, False, True)
            if useless:
                continue

            is_numeric = pd.to_numeric(tmp.reset_index()[grp_names[-1]], errors='coerce').notnull().all()
            enlarge_lbl_dist = check_legend_enlarge(tmp, grp_names[-1], len(list(tmp.columns.values)), fig_type)
            section_name = replace_stat_names(it_tmp[-1]) + ' values for ' +\
                           replaceCSVLabels(str(it_tmp[0]), True, False, True) +\
                           ' compared to ' + replaceCSVLabels(str(grp_names[-1]), True, False, True)
            exp_value = enl_space_title(exp_value, section_name, tmp, grp_names[-1],
                                        len(list(tmp.columns.values)), fig_type)
            reltex_name = os.path.join(rel_data_path, dataf_name)
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': section_name,
                                          # If caption is None, the field name is used
                                          'caption': None,
                                          'fig_type': fig_type,
                                          'plots': list(tmp.columns.values),
                                          'label_y': replace_stat_names(it_tmp[-1]) + findUnit(str(it_tmp[0]), units),
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
                                          'enlarge_lbl_dist': enlarge_lbl_dist,
                                          'pdf_nr': pdf_nr
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    if no_tex:
        return 0
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
        pdf_l_info = {'rendered_tex': [], 'texf_name': [], 'pdf_name': [] if build_pdf else None}
        for it in diff_pdfs:
            rendered_tex = template.render(title=tex_infos['title'],
                                           make_index=tex_infos['make_index'],
                                           ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                           figs_externalize=tex_infos['figs_externalize'],
                                           fill_bar=tex_infos['fill_bar'],
                                           sections=it,
                                           abbreviations=tex_infos['abbreviations'])
            pdf_l_info['rendered_tex'].append(rendered_tex)
            texf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.tex'
            pdf_l_info['texf_name'].append(texf_name)
            if build_pdf:
                pdf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.pdf'
                pdf_l_info['pdf_name'].append(os.path.join(pdf_folder, pdf_name))
        res = compile_tex(pdf_l_info['rendered_tex'], tex_folder, pdf_l_info['texf_name'], make_fig_index,
                          pdf_l_info['pdf_name'], tex_infos['figs_externalize'])
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
                                      figs_externalize=True,
                                      no_tex=False):
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
        store_path_sub = os.path.join(store_path, eval_description_path + '_' + short_concat_str(it_parameters) +
                                                  '_vs_' + '-'.join(map(str, x_axis_column)) + '_for_' +
                                                  '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]))
    else:
        store_path_sub = os.path.join(store_path, eval_description_path + '_' + short_concat_str(it_parameters) +
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
                                                       short_concat_str(compare_source['it_parameters']) +
                                                       '_vs_' + '-'.join(map(str, x_axis_column)) + '_for_' +
                                                       '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]))
        else:
            compare_source['full_path'] = os.path.join(compare_source['store_path'],
                                                       compare_source['eval_description_path'] + '_' +
                                                       short_concat_str(compare_source['it_parameters']) +
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
    base_out_name += concat_combs(grp_names, nr_it_parameters, nr_partitions)
    for i in range(0, nr_it_parameters):
        title_name += replaceCSVLabels(grp_names[nr_partitions + i], True, True, True)
        if(nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_name += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_name += ', '
            elif i < nr_it_parameters - 1:
                title_name += ', and '
    if len(partitions) > 1:
        base_out_name += '_combs_vs_' + grp_names[-1] + '_for_' + \
                         '-'.join([a[:min(3, len(a))] for a in map(str, partitions)])
    else:
        base_out_name += '_combs_vs_' + grp_names[-1] + '_for_' + str(partitions)
    title_name += ' Compared to ' + replaceCSVLabels(grp_names[-1], False, True, True) + ' Values Separately for '
    for i in range(0, nr_partitions):
        title_name += replaceCSVLabels(grp_names[i], True, True, True)
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
            it_tmp = list(it)
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
                part_name_l = [replaceCSVLabels(str(ni), False, False, True) + ' = ' +
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
                    if compare_source['replace_evals']:
                        succe, dst_eval, new_name = get_replace_eval(compare_source, str(it[0]))
                        if succe:
                            datafc_name = dataf_name.replace(str(it[0]), dst_eval)
                            dataf_name = dataf_name.replace(str(it[0]), new_name)
                            tmp2.rename(columns={it[0]: new_name}, inplace=True)
                            it_tmp[0] = new_name
                        else:
                            datafc_name = dataf_name
                        _, dst_eval, _ = get_replace_eval(compare_source, grp_names[-1], True)
                        datafc_name = datafc_name.replace(str(grp_names[-1]), str(dst_eval))
                    else:
                        datafc_name = dataf_name
                    tmp2, succ = add_comparison_column(compare_source, datafc_name, tmp2)
                fdataf_name = os.path.join(tdata_folder, dataf_name)
                with open(fdataf_name, 'a') as f:
                    f.write('# ' + str(it_tmp[-1]) + ' values for ' + str(it_tmp[0]) +
                            ' and properties ' + part_name + '\n')
                    f.write('# Column parameters: ' + '-'.join(it_parameters) + '\n')
                    tmp2.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

                if no_tex:
                    continue

                #Construct tex-file
                useless, use_limits, use_log, exp_value = get_limits_log_exp(tmp2, False, False, True)
                if useless:
                    continue
                is_numeric = pd.to_numeric(tmp2.reset_index()[grp_names[-1]], errors='coerce').notnull().all()
                enlarge_lbl_dist = check_legend_enlarge(tmp2, grp_names[-1], len(list(tmp2.columns.values)), fig_type)
                section_name = replace_stat_names(it_tmp[-1]) + ' values for ' +\
                               replaceCSVLabels(str(it_tmp[0]), True, False, True) +\
                               ' compared to ' + replaceCSVLabels(str(grp_names[-1]), True, False, True) +\
                               '\\\\for properties ' + part_name.replace('_', '\\_')
                section_name = split_large_titles(section_name)
                exp_value = enl_space_title(exp_value, section_name, tmp2, grp_names[-1],
                                            len(list(tmp2.columns.values)), fig_type)
                reltex_name = os.path.join(rel_data_path, dataf_name)
                tex_infos['sections'].append({'file': reltex_name,
                                              'name': section_name,
                                              # If caption is None, the field name is used
                                              'caption': replace_stat_names(it_tmp[-1]) + ' values for ' +
                                                      replaceCSVLabels(str(it_tmp[0]), True) +
                                                      ' compared to ' + replaceCSVLabels(str(grp_names[-1]), True) +
                                                      ' for properties ' + part_name_title,
                                              'fig_type': fig_type,
                                              'plots': list(tmp2.columns.values),
                                              'label_y': replace_stat_names(it_tmp[-1]) +
                                                         findUnit(str(it_tmp[0]), units),
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
                                              'enlarge_lbl_dist': enlarge_lbl_dist,
                                              'stat_name': it_tmp[-1],
                                              })
                tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    if no_tex:
        return 0

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
    pdf_l_info = {'rendered_tex': [], 'texf_name': [], 'pdf_name': [] if build_pdf else None}
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
            pdf_l_info['pdf_name'].append(os.path.join(pdf_folder, pdf_name))

        pdf_l_info['rendered_tex'].append(rendered_tex)
        pdf_l_info['texf_name'].append(texf_name)
    res = compile_tex(pdf_l_info['rendered_tex'], tex_folder, pdf_l_info['texf_name'], make_fig_index,
                      pdf_l_info['pdf_name'], figs_externalize)

    return res


def calcFromFuncAndPlot_2D(data,
                           store_path,
                           tex_file_pre_str,
                           fig_title_pre_str,
                           eval_description_path,
                           eval_columns,
                           units,
                           it_parameters,
                           x_axis_column,
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
                           figs_externalize=True,
                           no_tex=False):
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
    evals_for_gloss = []
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
        x_axis_column = ret['x_axis_column']
        if 'evals_for_gloss' in ret:
            evals_for_gloss = ret['evals_for_gloss']
    else:
        raise ValueError('No function for calculating results provided')

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + short_concat_str(it_parameters) + '_vs_' +
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
                                                   short_concat_str(compare_source['it_parameters']) + '_vs_' +
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
            special_calcs_args['it_parameters'] = it_parameters
            special_calcs_args['res_folder'] = special_path_sub
            if evals_for_gloss:
                special_calcs_args['evals_for_gloss'] = evals_for_gloss
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Calculation of specific results failed!', UserWarning)

    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    it_title_part = ''
    it_pars_short = short_concat_str(it_parameters)
    base_out_name += it_pars_short
    for i, val in enumerate(it_parameters):
        it_title_part += replaceCSVLabels(val, True, True, True)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                it_title_part += ' and '
        else:
            if i < nr_it_parameters - 2:
                it_title_part += ', '
            elif i < nr_it_parameters - 1:
                it_title_part += ', and '
    title_name += it_title_part

    if eval_init_input:
        init_pars_title = ''
        init_pars_out_name = ''
        nr_eval_init_input = len(eval_init_input)
        if nr_eval_init_input > 1:
            for i, val in enumerate(eval_init_input):
                init_pars_out_name += val
                init_pars_title += replaceCSVLabels(val, True, True, True)
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
            init_pars_title = replaceCSVLabels(eval_init_input[0], True, True, True)
        base_out_name += '_combs_vs_' + x_axis_column[0] + '_based_on_' + init_pars_out_name
        title_name += ' Compared to ' + replaceCSVLabels(x_axis_column[0], True, True, True) + \
                      ' Based On ' + init_pars_title
    else:
        base_out_name += '_combs_vs_' + x_axis_column[0]
        title_name += ' Compared to ' + replaceCSVLabels(x_axis_column[0], True, True, True)
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

    df.set_index(it_parameters, inplace=True)
    df = df.T
    if len(it_parameters) > 1:
        gloss = glossary_from_list([str(b) for a in df.columns for b in a])
    else:
        gloss = glossary_from_list([str(a) for a in df.columns])
    if compare_source:
        add_to_glossary(compare_source['it_par_select'], gloss)
        gloss.append({'key': 'cmp', 'description': compare_source['cmp']})
    if gloss:
        if evals_for_gloss:
            gloss = add_to_glossary_eval(evals_for_gloss + x_axis_column, gloss)
        else:
            gloss = add_to_glossary_eval(eval_columns + x_axis_column, gloss)
        tex_infos['abbreviations'] = gloss
    else:
        if evals_for_gloss:
            gloss = add_to_glossary_eval(evals_for_gloss + x_axis_column)
        else:
            gloss = add_to_glossary_eval(eval_columns + x_axis_column)
        if gloss:
            tex_infos['abbreviations'] = gloss
    if len(it_parameters) > 1:
        par_cols = ['-'.join(map(str, a)) for a in df.columns]
        df.columns = par_cols
        it_pars_cols_name = '-'.join(map(str, it_parameters))
        df.columns.name = it_pars_cols_name
    else:
        par_cols = [str(a) for a in df.columns]
        it_pars_cols_name = it_parameters[0]
    tmp = df.T.reset_index().set_index(x_axis_column + [it_pars_cols_name]).unstack()
    par_cols1 = ['-'.join(map(str, a)) for a in tmp.columns]
    tmp.columns = par_cols1
    tmp.columns.name = 'eval-' + it_pars_cols_name
    if compare_source:
        if eval_init_input:
            if compare_source['replace_evals']:
                eval_init_input_old = []
                for ie in eval_init_input:
                    _, dst_eval, _ = get_replace_eval(compare_source, str(ie), True)
                    eval_init_input_old.append(dst_eval)
                init_pars_out_name_c = (init_pars_out_name + '.')[:-1]
                for act, rep in zip(eval_init_input, eval_init_input_old):
                    init_pars_out_name_c.replace(act, rep)
            else:
                init_pars_out_name_c = init_pars_out_name
            cmp_fname = 'data_evals_' + init_pars_out_name_c + '_for_pars_'
        else:
            cmp_fname = 'data_evals_for_pars_'
        cmp_fname += short_concat_str(compare_source['it_parameters'])
    if eval_init_input:
        dataf_name = 'data_evals_' + init_pars_out_name + '_for_pars_' + it_pars_short
    else:
        dataf_name = 'data_evals_for_pars_' + it_pars_short
    dataf_name += '_vs_' + x_axis_column[0] + '.csv'
    if compare_source:
        if compare_source['replace_evals']:
            _, dst_eval, _ = get_replace_eval(compare_source, x_axis_column[0], True)
            cmp_fname += '_vs_' + dst_eval + '.csv'
        else:
            cmp_fname += '_vs_' + x_axis_column[0] + '.csv'
        cmp_its = '-'.join(map(str, compare_source['it_par_select']))
        mult_cols = [a.replace(par_cols[0], cmp_its) for a in par_cols1 if par_cols[0] in a]
        tmp, succ = add_comparison_column(compare_source, cmp_fname, tmp, mult_cols)
        if succ:
            par_cols1 += mult_cols

    fdataf_name = os.path.join(tdata_folder, dataf_name)
    with open(fdataf_name, 'a') as f:
        if eval_init_input:
            f.write('# Evaluations on ' + init_pars_out_name + ' for parameter variations of ' +
                    it_pars_cols_name + '\n')
        else:
            f.write('# Evaluations for parameter variations of ' +
                    it_pars_cols_name + '\n')
        f.write('# Column parameters: ' + ', '.join(eval_cols_lname) + '\n')
        tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    if no_tex:
        return 0

    for i, ev in enumerate(eval_columns):
        # sel_cols = [a for a in par_cols1 if ev in a]
        sel_cols = []
        for a in par_cols1:
            if ev in a:
                not_found = True
                for b in a.split('-'):
                    if b == ev:
                        not_found = False
                        break
                if not_found:
                    continue
                else:
                    sel_cols.append(a)

        legend = ['-'.join([b for b in a.split('-') if ev not in b]) for a in sel_cols]

        # Construct tex-file
        useless, stats_all, use_limits = calc_limits(tmp, True, False, None, sel_cols, 3.291)
        if useless:
            continue
        if use_limits['miny'] and use_limits['maxy']:
            exp_value = is_exp_used(use_limits['miny'], use_limits['maxy'], eval_cols_log_scaling[i])
        elif use_limits['miny']:
            exp_value = is_exp_used(use_limits['miny'], stats_all['max'][0], eval_cols_log_scaling[i])
        elif use_limits['maxy']:
            exp_value = is_exp_used(stats_all['min'][0], use_limits['maxy'], eval_cols_log_scaling[i])
        else:
            exp_value = is_exp_used(stats_all['min'][0], stats_all['max'][0], eval_cols_log_scaling[i])
        is_numeric = pd.to_numeric(tmp.reset_index()[x_axis_column[0]], errors='coerce').notnull().all()
        enlarge_lbl_dist = check_legend_enlarge(tmp, x_axis_column[0], len(sel_cols), fig_type)
        reltex_name = os.path.join(rel_data_path, dataf_name)
        if eval_init_input:
            fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' based on ' + strToLower(init_pars_title)
            fig_name += '\\\\for parameter variations of ' + strToLower(it_title_part)
        else:
            fig_name = capitalizeFirstChar(eval_cols_lname[i])
            fig_name += ' for parameter variations of\\\\' + strToLower(it_title_part)
        fig_name += '\\\\compared to ' + replaceCSVLabels(x_axis_column[0], True, False, True)
        fig_name = split_large_titles(fig_name)
        exp_value = enl_space_title(exp_value, fig_name, tmp, x_axis_column[0],
                                    len(sel_cols), fig_type)
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
                                      'enlarge_lbl_dist': enlarge_lbl_dist,
                                      'stat_name': ev,
                                      })
        tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

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
                          'figs_externalize': figs_externalize,
                          'sections': it['figs'],
                          'make_index': tex_infos['make_index'],
                          'ctrl_fig_size': tex_infos['ctrl_fig_size'],
                          'fill_bar': tex_infos['fill_bar'],
                          'abbreviations': tex_infos['abbreviations']})

    template = ji_env.get_template('usac-testing_2D_plots.tex')
    pdf_l_info = {'rendered_tex': [], 'texf_name': [], 'pdf_name': [] if build_pdf else None}
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
            pdf_l_info['pdf_name'].append(os.path.join(pdf_folder, pdf_name))

        pdf_l_info['rendered_tex'].append(rendered_tex)
        pdf_l_info['texf_name'].append(texf_name)
    res = compile_tex(pdf_l_info['rendered_tex'], tex_folder, pdf_l_info['texf_name'], make_fig_index,
                      pdf_l_info['pdf_name'], figs_externalize)

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
                                      figs_externalize=True,
                                      no_tex=False):
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
    evals_for_gloss = []
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
        if 'evals_for_gloss' in ret:
            evals_for_gloss = ret['evals_for_gloss']
    else:
        raise ValueError('No function for calculating results provided')

    if len(partitions) > 1:
        store_path_sub = os.path.join(store_path, eval_description_path + '_' +
                                      short_concat_str(it_parameters) + '_vs_' +
                                      '-'.join(map(str, x_axis_column)) + '_for_' +
                                      '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]))
    else:
        store_path_sub = os.path.join(store_path, eval_description_path + '_' +
                                      short_concat_str(it_parameters) +
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
                                                       short_concat_str(compare_source['it_parameters']) +
                                                       '_vs_' + '-'.join(map(str, x_axis_column)) + '_for_' +
                                                       '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]))
        else:
            compare_source['full_path'] = os.path.join(compare_source['store_path'],
                                                       compare_source['eval_description_path'] + '_' +
                                                       short_concat_str(compare_source['it_parameters']) +
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
            if evals_for_gloss:
                special_calcs_args['evals_for_gloss'] = evals_for_gloss
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Calculation of specific results failed!', UserWarning)

    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    nr_partitions = len(partitions)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    it_title_part = ''
    it_pars_short = short_concat_str(it_parameters)
    base_out_name += it_pars_short
    for i, val in enumerate(it_parameters):
        it_title_part += replaceCSVLabels(val, True, True, True)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                it_title_part += ' and '
        else:
            if i < nr_it_parameters - 2:
                it_title_part += ', '
            elif i < nr_it_parameters - 1:
                it_title_part += ', and '
    title_name += it_title_part

    if eval_init_input:
        init_pars_title = ''
        init_pars_out_name = ''
        nr_eval_init_input = len(eval_init_input)
        if nr_eval_init_input > 1:
            for i, val in enumerate(eval_init_input):
                init_pars_out_name += val
                init_pars_title += replaceCSVLabels(val, True, True, True)
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
            init_pars_title = replaceCSVLabels(eval_init_input[0], True, True, True)
        base_out_name += '_combs_vs_' + x_axis_column[0] + '_based_on_' + init_pars_out_name + \
                         '_for_' + '-'.join([a[:min(3, len(a))] for a in map(str, partitions)])
        title_name += ' Compared to ' + replaceCSVLabels(x_axis_column[0], True, True, True) + \
                      ' Based On ' + init_pars_title + ' Separately for '
    else:
        base_out_name += '_combs_vs_' + x_axis_column[0] + \
                         '_for_' + '-'.join([a[:min(3, len(a))] for a in map(str, partitions)])
        title_name += ' Compared to ' + replaceCSVLabels(x_axis_column[0], True, True, True) + ' Separately for '
    partition_text = ''
    for i, val in enumerate(partitions):
        partition_text += replaceCSVLabels(val, True, True, True)
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
        partition_text_val.append([replaceCSVLabels(val, False, False, True)])
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
                    partition_text_val_tmp[i][0] = partition_text_val_tmp[i][0][:-1] + '=' + str(grp[i]) + '$'
                elif '}' == ptv[0][-1]:
                    partition_text_val_tmp[i][0] += '=' + str(grp[i])
                else:
                    partition_text_val_tmp[i][0] += ' equal to ' + str(grp[i])
            partition_text_val1 = ''.join([''.join(a) for a in partition_text_val_tmp])
        else:
            if '$' == partition_text_val_tmp[0][0][-1]:
                partition_text_val_tmp[0][0] = partition_text_val_tmp[0][0][:-1] + '=' + str(grp) + '$'
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
                if evals_for_gloss:
                    gloss = add_to_glossary_eval(evals_for_gloss + x_axis_column + partitions, gloss)
                else:
                    gloss = add_to_glossary_eval(eval_columns + x_axis_column + partitions, gloss)
                tex_infos['abbreviations'] = gloss
            else:
                if evals_for_gloss:
                    gloss = add_to_glossary_eval(evals_for_gloss + x_axis_column + partitions)
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
            if eval_init_input:
                if compare_source['replace_evals']:
                    eval_init_input_old = []
                    for ie in eval_init_input:
                        _, dst_eval, _ = get_replace_eval(compare_source, str(ie), True)
                        eval_init_input_old.append(dst_eval)
                    init_pars_out_name_c = (init_pars_out_name + '.')[:-1]
                    for act, rep in zip(eval_init_input, eval_init_input_old):
                        init_pars_out_name_c.replace(act, rep)
                else:
                    init_pars_out_name_c = init_pars_out_name
                cmp_fname = 'data_evals_' + init_pars_out_name_c + '_for_pars_'
            else:
                cmp_fname = 'data_evals_for_pars_'
            cmp_fname += short_concat_str(compare_source['it_parameters'])
            cmp_fname += '_on_partition_'
            if compare_source['replace_evals']:
                partitions_old = []
                for ie in partitions:
                    _, dst_eval, _ = get_replace_eval(compare_source, ie, True)
                    partitions_old.append(dst_eval)
            else:
                partitions_old = partitions
            cmp_fname += '-'.join([a[:min(3, len(a))] for a in map(str, partitions_old)]) + '_'
        if eval_init_input:
            dataf_name = 'data_evals_' + init_pars_out_name + '_for_pars_' + it_pars_short + '_on_partition_'
        else:
            dataf_name = 'data_evals_for_pars_' + it_pars_short + '_on_partition_'
        dataf_name += '-'.join([a[:min(3, len(a))] for a in map(str, partitions)]) + '_'
        grp_name = '-'.join([a[:min(4, len(a))] for a in map(str, grp)]) if len(partitions) > 1 else str(grp)
        dataf_name += grp_name.replace('.', 'd')
        dataf_name += '_vs_' + x_axis_column[0] + '.csv'
        if compare_source:
            if compare_source['replace_evals']:
                grp_old = []
                for ie in grp:
                    _, dst_eval, _ = get_replace_eval(compare_source, ie, True)
                    grp_old.append(dst_eval)
                grp_name_old = '-'.join([a[:min(4, len(a))] for a in map(str, grp_old)]) \
                    if len(partitions) > 1 else str(grp_old)
                cmp_fname += grp_name_old.replace('.', 'd')
                _, dst_eval, _ = get_replace_eval(compare_source, x_axis_column[0], True)
                cmp_fname += '_vs_' + dst_eval + '.csv'
            else:
                cmp_fname += grp_name.replace('.', 'd')
                cmp_fname += '_vs_' + x_axis_column[0] + '.csv'
            cmp_its = '-'.join(map(str, compare_source['it_par_select']))
            mult_cols = [a.replace(par_cols[0], cmp_its) for a in par_cols1 if par_cols[0] in a]
            tmp, succ = add_comparison_column(compare_source, cmp_fname, tmp, mult_cols)
            if succ:
                par_cols1 += mult_cols

        fdataf_name = os.path.join(tdata_folder, dataf_name)
        with open(fdataf_name, 'a') as f:
            if eval_init_input:
                f.write('# Evaluations on ' + init_pars_out_name + ' for parameter variations of ' +
                        it_pars_cols_name + '\n')
            else:
                f.write('# Evaluations for parameter variations of ' + it_pars_cols_name + '\n')
            f.write('# Used data part of ' + '-'.join(map(str, partitions)) + ': ' + grp_name + '\n')
            f.write('# Column parameters: ' + ', '.join(eval_cols_lname) + '\n')
            tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

        if no_tex:
            continue

        for i, ev in enumerate(eval_columns):
            # sel_cols = [a for a in par_cols1 if ev in a]
            sel_cols = []
            for a in par_cols1:
                if ev in a:
                    not_found = True
                    for b in a.split('-'):
                        if b == ev:
                            not_found = False
                            break
                    if not_found:
                        continue
                    else:
                        sel_cols.append(a)
            legend = ['-'.join([b for b in a.split('-') if ev not in b]) for a in sel_cols]

            # Construct tex-file
            useless, stats_all, use_limits = calc_limits(tmp, True, False, None, sel_cols, 3.291)
            if useless:
                continue
            if use_limits['miny'] and use_limits['maxy']:
                exp_value = is_exp_used(use_limits['miny'], use_limits['maxy'], eval_cols_log_scaling[i])
            elif use_limits['miny']:
                exp_value = is_exp_used(use_limits['miny'], stats_all['max'][0], eval_cols_log_scaling[i])
            elif use_limits['maxy']:
                exp_value = is_exp_used(stats_all['min'][0], use_limits['maxy'], eval_cols_log_scaling[i])
            else:
                exp_value = is_exp_used(stats_all['min'][0], stats_all['max'][0], eval_cols_log_scaling[i])
            is_numeric = pd.to_numeric(tmp.reset_index()[x_axis_column[0]], errors='coerce').notnull().all()
            enlarge_lbl_dist = check_legend_enlarge(tmp, x_axis_column[0], len(sel_cols), fig_type)
            reltex_name = os.path.join(rel_data_path, dataf_name)
            if eval_init_input:
                fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' based on ' + strToLower(init_pars_title)
                fig_name += '\\\\for '
            else:
                fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' for '
            fig_name += partition_text_val1 + ' in addition to ' + \
                        '\\\\parameter variations of ' + strToLower(it_title_part) + \
                        '\\\\compared to ' + \
                        replaceCSVLabels(x_axis_column[0], True, False, True)
            fig_name = split_large_titles(fig_name)
            exp_value = enl_space_title(exp_value, fig_name, tmp, x_axis_column[0],
                                        len(sel_cols), fig_type)
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
                                          'enlarge_lbl_dist': enlarge_lbl_dist,
                                          'stat_name': ev,
                                          })
            tex_infos['sections'][-1]['legend_cols'] = calcNrLegendCols(tex_infos['sections'][-1])

    if no_tex:
        return 0

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
    pdf_l_info = {'rendered_tex': [], 'texf_name': [], 'pdf_name': [] if build_pdf else None}
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
            pdf_l_info['pdf_name'].append(os.path.join(pdf_folder, pdf_name))

        pdf_l_info['rendered_tex'].append(rendered_tex)
        pdf_l_info['texf_name'].append(texf_name)
    res = compile_tex(pdf_l_info['rendered_tex'], tex_folder, pdf_l_info['texf_name'], make_fig_index,
                      pdf_l_info['pdf_name'], figs_externalize)

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
                           figs_externalize=True,
                           no_tex=False):
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

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + short_concat_str(it_parameters) + '_vs_' +
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
    base_out_name += concat_combs(grp_names, nr_it_parameters)
    for i in range(0, nr_it_parameters):
        title_name += replaceCSVLabels(grp_names[i], True, True, True)
        if(nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_name += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_name += ', '
            elif i < nr_it_parameters - 1:
                title_name += ', and '
    base_out_name += '_combs_vs_' + grp_names[-2] + '_and_' + grp_names[-1]
    title_name += ' Compared to ' + replaceCSVLabels(grp_names[-2], False, True, True) + ' and ' + \
                  replaceCSVLabels(grp_names[-1], False, True, True) + ' Values'
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

            if no_tex:
                continue

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
                                                  replaceCSVLabels(str(it[0]), True, False, True) +
                                                  ' compared to ' +
                                                  replaceCSVLabels(str(grp_names[-2]), True, False, True) +
                                                  ' and ' + replaceCSVLabels(str(grp_names[-1]), True, False, True),
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

    if no_tex:
        return 0

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
    pdf_l_info = {'rendered_tex': [], 'texf_name': [], 'pdf_name': [] if build_pdf else None}
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
            pdf_l_info['pdf_name'].append(os.path.join(pdf_folder, pdf_name))

        pdf_l_info['rendered_tex'].append(rendered_tex)
        pdf_l_info['texf_name'].append(texf_name)
    res = compile_tex(pdf_l_info['rendered_tex'], tex_folder, pdf_l_info['texf_name'], make_fig_index,
                      pdf_l_info['pdf_name'], figs_externalize)

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
                           figs_externalize=True,
                           no_tex=False):
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
    evals_for_gloss = []
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
        if 'evals_for_gloss' in ret:
            evals_for_gloss = ret['evals_for_gloss']
    else:
        raise ValueError('No function for calculating results provided')

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + short_concat_str(it_parameters) + '_vs_' +
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
            if evals_for_gloss:
                special_calcs_args['evals_for_gloss'] = evals_for_gloss
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Calculation of specific results failed!', UserWarning)
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    base_out_name += short_concat_str(it_parameters)
    for i, val in enumerate(it_parameters):
        title_name += replaceCSVLabels(val, True, True, True)
        if(nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_name += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_name += ', '
            elif i < nr_it_parameters - 1:
                title_name += ', and '
    if eval_init_input:
        init_pars_title = ''
        init_pars_out_name = ''
        nr_eval_init_input = len(eval_init_input)
        if nr_eval_init_input > 1:
            for i, val in enumerate(eval_init_input):
                init_pars_out_name += val
                init_pars_title += replaceCSVLabels(val, True, True, True)
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
            init_pars_title = replaceCSVLabels(eval_init_input[0], True, True, True)
        base_out_name += '_combs_vs_' + xy_axis_columns[0] + '_and_' + xy_axis_columns[1] + '_based_on_' + \
                         init_pars_out_name
        title_name += ' Compared to ' + replaceCSVLabels(xy_axis_columns[0], True, True, True) + ' and ' + \
                      replaceCSVLabels(xy_axis_columns[1], True, True, True) + ' Based On ' + init_pars_title
    else:
        base_out_name += '_combs_vs_' + xy_axis_columns[0] + '_and_' + xy_axis_columns[1]
        title_name += ' Compared to ' + replaceCSVLabels(xy_axis_columns[0], True, True, True) + ' and ' + \
                      replaceCSVLabels(xy_axis_columns[1], True, True, True)
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
        if evals_for_gloss:
            gloss = add_to_glossary_eval(evals_for_gloss + xy_axis_columns, gloss)
        else:
            gloss = add_to_glossary_eval(eval_columns + xy_axis_columns, gloss)
        tex_infos['abbreviations'] = gloss
    else:
        if evals_for_gloss:
            gloss = add_to_glossary_eval(evals_for_gloss + xy_axis_columns)
        else:
            gloss = add_to_glossary_eval(eval_columns + xy_axis_columns)
        if gloss:
            tex_infos['abbreviations'] = gloss
    for grp in grp_keys:
        if eval_init_input:
            dataf_name = 'data_evals_' + init_pars_out_name + '_for_pars_' + \
                         short_concat_str(grp) + '_vs_' + xy_axis_columns[0] + \
                         '_and_' + xy_axis_columns[1] + '.csv'
        else:
            dataf_name = 'data_evals_for_pars_' + \
                         short_concat_str(grp) + '_vs_' + xy_axis_columns[0] + \
                         '_and_' + xy_axis_columns[1] + '.csv'
        fdataf_name = os.path.join(tdata_folder, dataf_name)
        tmp = df.get_group(grp)
        tmp = tmp.drop(it_parameters, axis=1)
        # nr_equal_ss = int(tmp.groupby(xy_axis_columns[0]).size().array[0])
        nr_equal_ss = get_block_length_3D(tmp, xy_axis_columns)
        with open(fdataf_name, 'a') as f:
            if eval_init_input:
                f.write('# Evaluations on ' + init_pars_out_name + ' for parameter variations of ' +
                        '-'.join(map(str, it_parameters)) + '\n')
            else:
                f.write('# Evaluations for parameter variations of ' +
                        '-'.join(map(str, it_parameters)) + '\n')
            if len(it_parameters) > 1:
                f.write('# Used parameter values: ' + '-'.join(map(str, grp)) + '\n')
            else:
                f.write('# Used parameter values: ' + str(grp) + '\n')
            f.write('# Column parameters: ' + ', '.join(eval_cols_lname) + '\n')
            tmp.to_csv(index=False, sep=';', path_or_buf=f, header=True, na_rep='nan')

        if no_tex:
            continue

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
                if eval_init_input:
                    fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' based on ' + strToLower(init_pars_title)
                else:
                    fig_name = capitalizeFirstChar(eval_cols_lname[i])
                fig_name += ' for parameters ' + tex_string_coding_style('-'.join(map(str, grp))) + ' compared to ' + \
                           replaceCSVLabels(xy_axis_columns[0], True, False, True) + ' and ' + \
                           replaceCSVLabels(xy_axis_columns[1], True, False, True)
                legends = [eval_cols_lname[i] + ' for ' + tex_string_coding_style('-'.join(map(str, grp)))]
            else:
                if eval_init_input:
                    fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' based on ' + strToLower(init_pars_title)
                else:
                    fig_name = capitalizeFirstChar(eval_cols_lname[i])
                fig_name += ' for parameters ' + tex_string_coding_style(str(grp)) + ' compared to ' + \
                           replaceCSVLabels(xy_axis_columns[0], True, False, True) + ' and ' + \
                           replaceCSVLabels(xy_axis_columns[1], True, False, True)
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

    if no_tex:
        return 0

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
    pdf_l_info = {'rendered_tex': [], 'texf_name': [], 'pdf_name': [] if build_pdf else None}
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
            pdf_l_info['pdf_name'].append(os.path.join(pdf_folder, pdf_name))

        pdf_l_info['rendered_tex'].append(rendered_tex)
        pdf_l_info['texf_name'].append(texf_name)
    res = compile_tex(pdf_l_info['rendered_tex'], tex_folder, pdf_l_info['texf_name'], make_fig_index,
                      pdf_l_info['pdf_name'], figs_externalize)

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
                                      figs_externalize=True,
                                      no_tex=False):
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
    evals_for_gloss = []
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
        if 'evals_for_gloss' in ret:
            evals_for_gloss = ret['evals_for_gloss']
    else:
        raise ValueError('No function for calculating results provided')

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + short_concat_str(it_parameters) + '_vs_' +
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
            if evals_for_gloss:
                special_calcs_args['evals_for_gloss'] = evals_for_gloss
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Calculation of specific results failed!', UserWarning)

    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    nr_partitions = len(partitions)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    it_title_part = ''
    base_out_name += short_concat_str(it_parameters)
    for i, val in enumerate(it_parameters):
        it_title_part += replaceCSVLabels(val, True, True, True)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                it_title_part += ' and '
        else:
            if i < nr_it_parameters - 2:
                it_title_part += ', '
            elif i < nr_it_parameters - 1:
                it_title_part += ', and '
    title_name += it_title_part

    if eval_init_input:
        init_pars_title = ''
        init_pars_out_name = ''
        nr_eval_init_input = len(eval_init_input)
        if nr_eval_init_input > 1:
            for i, val in enumerate(eval_init_input):
                init_pars_out_name += val
                init_pars_title += replaceCSVLabels(val, True, True, True)
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
            init_pars_title = replaceCSVLabels(eval_init_input[0], True, True, True)
        base_out_name += '_combs_vs_' + xy_axis_columns[0] + '_and_' + xy_axis_columns[1] + '_based_on_' + \
                         init_pars_out_name + '_for_' + '-'.join([a[:min(3, len(a))] for a in map(str, partitions)])
        title_name += ' Compared to ' + replaceCSVLabels(xy_axis_columns[0], True, True, True) + ' and ' + \
                      replaceCSVLabels(xy_axis_columns[1], True, True, True) + ' Based On ' + init_pars_title + \
                      ' Separately for '
    else:
        base_out_name += '_combs_vs_' + xy_axis_columns[0] + '_and_' + xy_axis_columns[1] + \
                         '_for_' + '-'.join([a[:min(3, len(a))] for a in map(str, partitions)])
        title_name += ' Compared to ' + replaceCSVLabels(xy_axis_columns[0], True, True, True) + ' and ' + \
                      replaceCSVLabels(xy_axis_columns[1], True, True, True) + ' Separately for '

    partition_text = ''
    for i, val in enumerate(partitions):
        partition_text += replaceCSVLabels(val, True, True, True)
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
        partition_text_val.append([replaceCSVLabels(val, False, False, True)])
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
                    partition_text_val_tmp[i][0] = partition_text_val_tmp[i][0][:-1] + '=' + str(grp[i]) + '$'
                elif '}' == ptv[0][-1]:
                    partition_text_val_tmp[i][0] += '=' + str(grp[i])
                else:
                    partition_text_val_tmp[i][0] += ' equal to ' + str(grp[i])
            partition_text_val1 = ''.join([''.join(a) for a in partition_text_val_tmp])
        else:
            if '$' == partition_text_val_tmp[0][0][-1]:
                partition_text_val_tmp[0][0] = partition_text_val_tmp[0][0][:-1] + '=' + str(grp) + '$'
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
                if evals_for_gloss:
                    gloss = add_to_glossary_eval(evals_for_gloss + xy_axis_columns + partitions, gloss)
                else:
                    gloss = add_to_glossary_eval(eval_columns + xy_axis_columns + partitions, gloss)
                tex_infos['abbreviations'] = gloss
            else:
                if evals_for_gloss:
                    gloss = add_to_glossary_eval(evals_for_gloss + xy_axis_columns + partitions)
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
            # nr_equal_ss = int(tmp1.groupby(xy_axis_columns[0]).size().array[0])
            nr_equal_ss = get_block_length_3D(tmp1, xy_axis_columns)

            if eval_init_input:
                dataf_name = 'data_evals_' + init_pars_out_name + '_for_pars_' + short_concat_str(grp_it.split('-')) + \
                             '_on_partition_'
            else:
                dataf_name = 'data_evals_for_pars_' + short_concat_str(grp_it.split('-')) + '_on_partition_'
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

            if no_tex:
                continue

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
                            replaceCSVLabels(xy_axis_columns[0], True, False, True) + ' and ' + \
                            replaceCSVLabels(xy_axis_columns[1], True, False, True)
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

    if no_tex:
        return 0

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
    pdf_l_info = {'rendered_tex': [], 'texf_name': [], 'pdf_name': [] if build_pdf else None}
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
            pdf_l_info['pdf_name'].append(os.path.join(pdf_folder, pdf_name))

        pdf_l_info['rendered_tex'].append(rendered_tex)
        pdf_l_info['texf_name'].append(texf_name)
    res = compile_tex(pdf_l_info['rendered_tex'], tex_folder, pdf_l_info['texf_name'], make_fig_index,
                      pdf_l_info['pdf_name'], figs_externalize)

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
                                  figs_externalize=False,
                                  no_tex=False):
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
    evals_for_gloss = []
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
        if 'evals_for_gloss' in ret:
            evals_for_gloss = ret['evals_for_gloss']
    else:
        raise ValueError('No function for calculating results provided')

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + short_concat_str(it_parameters))
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
                                                   short_concat_str(compare_source['it_parameters']))
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
            if evals_for_gloss:
                special_calcs_args['evals_for_gloss'] = evals_for_gloss
            res = special_calcs_func(**special_calcs_args)
            if res != 0:
                warnings.warn('Calculation of specific results failed!', UserWarning)
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    nr_it_parameters = len(it_parameters)
    base_out_name = tex_file_pre_str
    title_name = fig_title_pre_str
    title_it_pars = ''
    it_pars_short = short_concat_str(it_parameters)
    base_out_name += it_pars_short
    for i, val in enumerate(it_parameters):
        title_it_pars += replaceCSVLabels(val, True, True, True)
        if(nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_it_pars += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_it_pars += ', '
            elif i < nr_it_parameters - 1:
                title_it_pars += ', and '
    title_name += title_it_pars
    if eval_init_input:
        init_pars_title = ''
        init_pars_out_name = ''
        nr_eval_init_input = len(eval_init_input)
        if nr_eval_init_input > 1:
            for i, val in enumerate(eval_init_input):
                init_pars_out_name += val
                init_pars_title += replaceCSVLabels(val, True, True, True)
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
            init_pars_title = replaceCSVLabels(eval_init_input[0], True, True, True)
        base_out_name += '_combs_based_on_' + init_pars_out_name
        title_name += ' Based On ' + init_pars_title
    else:
        base_out_name += '_combs'
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
        if evals_for_gloss:
            gloss = add_to_glossary_eval(evals_for_gloss, gloss)
        else:
            gloss = add_to_glossary_eval(eval_columns, gloss)
    else:
        if evals_for_gloss:
            gloss = add_to_glossary_eval(evals_for_gloss)
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
        if eval_init_input:
            if compare_source['replace_evals']:
                eval_init_input_old = []
                for ie in eval_init_input:
                    _, dst_eval, _ = get_replace_eval(compare_source, str(ie), True)
                    eval_init_input_old.append(dst_eval)
                init_pars_out_name_c = (init_pars_out_name + '.')[:-1]
                for act, rep in zip(eval_init_input, eval_init_input_old):
                    init_pars_out_name_c.replace(act, rep)
            else:
                init_pars_out_name_c = init_pars_out_name
            comp_fname = 'data_evals_' + init_pars_out_name_c + '_for_pars_' + \
                         short_concat_str(compare_source['it_parameters']) + '.csv'
        else:
            comp_fname = 'data_evals_for_pars_' + short_concat_str(compare_source['it_parameters']) + '.csv'
        df, succ = add_comparison_column(compare_source, comp_fname, df, None, cmp_col_name)
    if eval_init_input:
        dataf_name = 'data_evals_' + init_pars_out_name + '_for_pars_' + it_pars_short + '.csv'
    else:
        dataf_name = 'data_evals_for_pars_' + it_pars_short + '.csv'
    fdataf_name = os.path.join(tdata_folder, dataf_name)
    with open(fdataf_name, 'a') as f:
        if eval_init_input:
            f.write('# Evaluations on ' + init_pars_out_name + ' for parameter variations of ' +
                    it_pars_name + '\n')
        else:
            f.write('# Evaluations for parameter variations of ' + it_pars_name + '\n')
        f.write('# Column parameters: ' + ', '.join(eval_cols_lname) + '\n')
        df.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

    if no_tex:
        return 0

    for i, it in enumerate(eval_columns):
        # Construct tex-file information
        useless, stats_all, use_limits = calc_limits(df, True, False, None, it, 3.291)
        if useless:
            continue
        if use_limits['miny'] and use_limits['maxy']:
            exp_value = is_exp_used(use_limits['miny'], use_limits['maxy'], eval_cols_log_scaling[i])
        elif use_limits['miny']:
            exp_value = is_exp_used(use_limits['miny'], stats_all['max'], eval_cols_log_scaling[i])
        elif use_limits['maxy']:
            exp_value = is_exp_used(stats_all['min'], use_limits['maxy'], eval_cols_log_scaling[i])
        else:
            exp_value = is_exp_used(stats_all['min'], stats_all['max'], eval_cols_log_scaling[i])
        reltex_name = os.path.join(rel_data_path, dataf_name)
        if eval_init_input:
            fig_name = capitalizeFirstChar(eval_cols_lname[i]) + ' based on ' + strToLower(init_pars_title) + \
                       ' for parameter variations of\\\\' + strToLower(title_it_pars)
        else:
            fig_name = capitalizeFirstChar(eval_cols_lname[i]) + \
                       ' for parameter variations of\\\\' + strToLower(title_it_pars)
        fig_name = split_large_titles(fig_name)
        exp_value = enl_space_title(exp_value, fig_name, df, 'tex_it_pars',
                                    1, fig_type)
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
                                      'enlarge_lbl_dist': None,
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
                                  figs_externalize=True,
                                  no_tex=False):
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
        if 'units' in ret:
            units = ret['units']
        it_parameters = ret['it_parameters']
    else:
        needed_columns = eval_columns + it_parameters
        df = data[needed_columns]

    store_path_sub = os.path.join(store_path, eval_description_path + '_' + short_concat_str(it_parameters))
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
                                                   short_concat_str(compare_source['it_parameters']))
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
            special_calcs_args['units'] = units
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
    base_out_name += concat_combs(grp_names, nr_it_parameters)
    for i in range(0, nr_it_parameters):
        title_it_pars += replaceCSVLabels(grp_names[i], True, True, True)
        if (nr_it_parameters <= 2):
            if i < nr_it_parameters - 1:
                title_it_pars += ' and '
        else:
            if i < nr_it_parameters - 2:
                title_it_pars += ', '
            elif i < nr_it_parameters - 1:
                title_it_pars += ', and '
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
            it_tmp = list(it)
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
                if compare_source['replace_evals']:
                    succe, dst_eval, new_name = get_replace_eval(compare_source, str(it[0]))
                    if succe:
                        datafc_name = dataf_name.replace(str(it[0]), dst_eval)
                        dataf_name = dataf_name.replace(str(it[0]), new_name)
                        tmp.rename(columns={it[0]: new_name}, inplace=True)
                        it_tmp[0] = new_name
                    else:
                        datafc_name = dataf_name
                else:
                    datafc_name = dataf_name
                cmp_col_name = '-'.join(compare_source['it_parameters'])
                tmp, succ = add_comparison_column(compare_source, datafc_name, tmp, None, cmp_col_name)
            fdataf_name = os.path.join(tdata_folder, dataf_name)
            with open(fdataf_name, 'a') as f:
                f.write('# ' + str(it_tmp[-1]) + ' values for ' + str(it_tmp[0]) + '\n')
                f.write('# Parameters: ' + '-'.join(it_parameters) + '\n')
                tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')

            if no_tex:
                continue

            # Construct tex-file
            if pdfsplitentry:
                if pdf_nr < len(pdfsplitentry):
                    if pdfsplitentry[pdf_nr] == str(it_tmp[0]):
                        pdf_nr += 1
            useless, use_limits, use_log, exp_value = get_limits_log_exp(tmp, True, True, True, 'tex_it_pars')
            if useless:
                continue

            fig_name = replace_stat_names(it_tmp[-1]) + ' values for ' +\
                       replaceCSVLabels(str(it_tmp[0]), True, False, True) + ' comparing parameter variations of\\\\' + \
                       strToLower(title_it_pars)
            fig_name = split_large_titles(fig_name)
            exp_value = enl_space_title(exp_value, fig_name, tmp, 'tex_it_pars',
                                        1, fig_type)
            reltex_name = os.path.join(rel_data_path, dataf_name)
            tex_infos['sections'].append({'file': reltex_name,
                                          'name': fig_name.replace('\\\\', ' '),
                                          'title': fig_name,
                                          'title_rows': fig_name.count('\\\\'),
                                          'fig_type': fig_type,
                                          'plots': [col_name],
                                          'label_y': replace_stat_names(it_tmp[-1]) + findUnit(str(it_tmp[0]), units),
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
                                          'enlarge_lbl_dist': None,
                                          'enlarge_title_space': exp_value,
                                          'large_meta_space_needed': False,
                                          'caption': fig_name.replace('\\\\', ' '),
                                          'pdf_nr': pdf_nr
                                          })

    if no_tex:
        return 0

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
        pdf_l_info = {'rendered_tex': [], 'texf_name': [], 'pdf_name': [] if build_pdf else None}
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
                pdf_l_info['pdf_name'].append(os.path.join(pdf_folder, pdf_name))

            pdf_l_info['rendered_tex'].append(rendered_tex)
            pdf_l_info['texf_name'].append(texf_name)
        res = compile_tex(pdf_l_info['rendered_tex'], tex_folder, pdf_l_info['texf_name'], make_fig_index,
                          pdf_l_info['pdf_name'], tex_infos['figs_externalize'])
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
            if compare_source['replace_evals']:
                dnew_cols = []
                for a in data_tmp.columns:
                    if isinstance(a, str) and '-' in str(a):
                        found_i = False
                        for b in a.split('-'):
                            for i, c in enumerate(compare_source['replace_evals']['actual']):
                                if b == c:
                                    if c != compare_source['replace_evals']['new'][i]:
                                        dnew_cols.append(a.replace(b, compare_source['replace_evals']['new'][i]))
                                    else:
                                        dnew_cols.append(a)
                                    found_i = True
                                    break
                            if found_i:
                                break
                        if not found_i:
                            dnew_cols.append(a)
                    else:
                        found_i = False
                        for i, c in enumerate(compare_source['replace_evals']['actual']):
                            if a == c:
                                if c != compare_source['replace_evals']['new'][i]:
                                    dnew_cols.append(compare_source['replace_evals']['new'][i])
                                else:
                                    dnew_cols.append(a)
                                found_i = True
                                break
                        if not found_i:
                            dnew_cols.append(a)
                data_tmp.columns = dnew_cols
                dold_cols = []
                for a in comp_data.columns:
                    if isinstance(a, str) and '-' in str(a):
                        found_i = False
                        for b in a.split('-'):
                            for i, c in enumerate(compare_source['replace_evals']['old']):
                                if b == c:
                                    if c != compare_source['replace_evals']['new'][i]:
                                        dold_cols.append(a.replace(b, compare_source['replace_evals']['new'][i]))
                                    else:
                                        dold_cols.append(a)
                                    found_i = True
                                    break
                            if found_i:
                                break
                        if not found_i:
                            dold_cols.append(a)
                    else:
                        found_i = False
                        for i, c in enumerate(compare_source['replace_evals']['old']):
                            if a == c:
                                if c != compare_source['replace_evals']['new'][i]:
                                    dold_cols.append(compare_source['replace_evals']['new'][i])
                                else:
                                    dold_cols.append(a)
                                found_i = True
                                break
                        if not found_i:
                            dold_cols.append(a)
                comp_data.columns = dold_cols
                dmult_cols = []
                for a in mult_cols:
                    if isinstance(a, str) and '-' in str(a):
                        found_i = False
                        for b in a.split('-'):
                            for i, c in enumerate(compare_source['replace_evals']['old']):
                                if b == c:
                                    if c != compare_source['replace_evals']['new'][i]:
                                        dmult_cols.append(a.replace(b, compare_source['replace_evals']['new'][i]))
                                    else:
                                        dmult_cols.append(a)
                                    found_i = True
                                    break
                            if found_i:
                                break
                        if not found_i:
                            dmult_cols.append(a)
                    else:
                        found_i = False
                        for i, c in enumerate(compare_source['replace_evals']['old']):
                            if a == c:
                                if c != compare_source['replace_evals']['new'][i]:
                                    dmult_cols.append(compare_source['replace_evals']['new'][i])
                                else:
                                    dmult_cols.append(a)
                                found_i = True
                                break
                        if not found_i:
                            dmult_cols.append(a)
                mult_cols = dmult_cols

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
                data_tmp = data_new.copy(deep=True)
                if compare_source['replace_evals']:
                    dnew_cols = []
                    for a in data_tmp.columns:
                        if isinstance(a, str) and '-' in str(a):
                            found_i = False
                            for b in a.split('-'):
                                for i, c in enumerate(compare_source['replace_evals']['actual']):
                                    if b == c:
                                        if c != compare_source['replace_evals']['new'][i]:
                                            dnew_cols.append(a.replace(b, compare_source['replace_evals']['new'][i]))
                                        else:
                                            dnew_cols.append(a)
                                        found_i = True
                                        break
                                if found_i:
                                    break
                            if not found_i:
                                dnew_cols.append(a)
                        else:
                            found_i = False
                            for i, c in enumerate(compare_source['replace_evals']['actual']):
                                if a == c:
                                    if c != compare_source['replace_evals']['new'][i]:
                                        dnew_cols.append(compare_source['replace_evals']['new'][i])
                                    else:
                                        dnew_cols.append(a)
                                    found_i = True
                                    break
                            if not found_i:
                                dnew_cols.append(a)
                    data_tmp.columns = dnew_cols
                    dold_cols = []
                    for a in comp_data.columns:
                        if isinstance(a, str) and '-' in str(a):
                            found_i = False
                            for b in a.split('-'):
                                for i, c in enumerate(compare_source['replace_evals']['old']):
                                    if b == c:
                                        if c != compare_source['replace_evals']['new'][i]:
                                            dold_cols.append(a.replace(b, compare_source['replace_evals']['new'][i]))
                                        else:
                                            dold_cols.append(a)
                                        found_i = True
                                        break
                                if found_i:
                                    break
                            if not found_i:
                                dold_cols.append(a)
                        else:
                            found_i = False
                            for i, c in enumerate(compare_source['replace_evals']['old']):
                                if a == c:
                                    if c != compare_source['replace_evals']['new'][i]:
                                        dold_cols.append(compare_source['replace_evals']['new'][i])
                                    else:
                                        dold_cols.append(a)
                                    found_i = True
                                    break
                            if not found_i:
                                dold_cols.append(a)
                    comp_data.columns = dold_cols

                for col1, col2 in zip(data_tmp.columns, comp_data.columns):
                    if col1 != col2:
                        succ = False
                        break
                if succ:
                    if comp_col_name not in comp_data.index:
                        warnings.warn('Parameter ' + comp_col_name + ' not found in csv file ' + comp_fdataf_name +
                                      ' for comparing results not found. Skipping comparison.', UserWarning)
                        succ = False
                    else:
                        line = comp_data.loc[comp_col_name, :].copy(deep=True)
                        line.name = 'cmp-' + line.name
                        if 'tex_it_pars' in line.index:
                            line.loc['tex_it_pars'] = 'cmp-' + str(line['tex_it_pars'])
                        data_tmp = data_tmp.append(line, ignore_index=False)
                        data_new = data_tmp
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


def replace_bm_in_headings(str_in):
    if '\\bm{' in str_in:
        return str_in.replace('\\bm{', '\\vect{')
    return str_in

def get_replace_eval(compare_source, act_eval, is_not_eval=False):
    if not compare_source['replace_evals']:
        return False, act_eval, act_eval
    if act_eval not in compare_source['replace_evals']['actual']:
        return False, act_eval, act_eval
    dest_eval = None
    i_f = 0
    for i, a in enumerate(compare_source['replace_evals']['actual']):
        if act_eval == a:
            dest_eval = compare_source['replace_evals']['old'][i]
            i_f = i
    if dest_eval is None:
        return False, act_eval, act_eval

    if compare_source['replace_evals']['new'][i_f] is None or is_not_eval:
        compare_source['replace_evals']['new'][i_f] = act_eval
    return True, dest_eval, compare_source['replace_evals']['new'][i_f]


def concat_combs(str_list, nr_elems, start=0, join_char='-'):
    return short_concat_str(str_list[start: start + nr_elems], join_char)


def short_concat_str(str_list, join_char='-'):
    if len(str_list) == 1 and len(str(str_list[0])) < 30:
        return str(str_list[0])
    elif len(str_list) < 4 and sum([len(str(a)) for a in str_list]) < 40:
        return join_char.join(map(str, str_list))

    #Get longest string sequence first which can be found in every element
    str_list_tmp = list(map(str, str_list))
    l1 = len(str_list_tmp[0])
    matches = []
    for i in range(1, len(str_list_tmp)):
        seqMatch = SequenceMatcher(None, str_list_tmp[0], str_list_tmp[i])
        match = seqMatch.find_longest_match(0, l1, 0, len(str_list_tmp[i]))
        if (match.size > 3):
            matches.append((match.a, match.size, str_list_tmp[i][match.a: match.a + match.size]))
    if matches:
        start_pos_l = []
        same_matches = [True] * len(matches)
        for i in range(1, len(matches)):
            if matches[i][2] != matches[0][2]:
                same_matches[i] = False
        if all(same_matches):
            shorty = ''
            if '_' == matches[0][2][0]:
                shorty += matches[0][2][0:3]
            else:
                shorty += matches[0][2][0:2]
            shorty += '_'
            shorty = shorty.upper()
            l2 = min(6, matches[0][1])
            # str_list_tmp[0] = str_list_tmp[0].replace(matches[0][2][l2:], '')
            start_pos_l.append([(str_list_tmp[0].find(matches[0][2]), l2, matches[0][1])])
            for i in range(1, len(str_list_tmp)):
                start_pos_l.append([(str_list_tmp[i].find(matches[0][2]), len(shorty), matches[0][1])])
                str_list_tmp[i] = str_list_tmp[i].replace(matches[0][2], shorty)
        else:
            matches2 = []
            start_pos_l = [[] for i in range(0, len(str_list_tmp))]
            for i in range(0, len(str_list_tmp) - 1):
                matches = []
                for j in range(i + 1, len(str_list_tmp)):
                    seqMatch = SequenceMatcher(None, str_list_tmp[i], str_list_tmp[j])
                    match = seqMatch.find_longest_match(0, len(str_list_tmp[i]), 0, len(str_list_tmp[j]))
                    if (match.size > 3):
                        matches.append((i, j, str_list_tmp[i][match.a: match.a + match.size]))
                matches2.append(matches)
            del_list2 = []
            for elm in matches2:
                del_list = []
                if elm:
                    for j in range(0, len(elm) - 1):
                        for j1 in range(j + 1, len(elm)):
                            if elm[j][2] == elm[j1][2]:
                                del_list.append(j1)
                if del_list:
                    del_list = list(dict.fromkeys(del_list))
                del_list2.append(del_list)
            for i, elm in enumerate(del_list2):
                if elm:
                    for j in range(len(elm) - 1, -1, -1):
                        del matches2[i][j]
            found_matches = {'keys': [], 'hit_poses': [], 'first_elem': []}
            for elmb in matches2:
                if elmb:
                    for j, elm in enumerate(elmb):
                        if elm[2] in found_matches['keys']:
                            for k, k1 in enumerate(found_matches['keys']):
                                if elm[2] == k1:
                                    found_matches['hit_poses'][k] += [elm[0], elm[1]]
                                    break
                        elif found_matches['keys']:
                            nfound = True
                            for k, k1 in enumerate(found_matches['keys']):
                                if elm[2] in k1:
                                    found_matches['keys'][k] = elm[2]
                                    found_matches['hit_poses'][k] += [elm[0], elm[1]]
                                    nfound = False
                                    break
                                elif k1 in elm[2]:
                                    found_matches['hit_poses'][k] += [elm[0], elm[1]]
                                    nfound = False
                                    break
                            if nfound:
                                found_matches['keys'].append(elm[2])
                                found_matches['hit_poses'].append([elm[0], elm[1]])
                        else:
                            found_matches['keys'].append(elm[2])
                            found_matches['hit_poses'].append([elm[0], elm[1]])
            del_list2 = []
            for i in range(0, len(found_matches['keys'])):
                found_matches['hit_poses'][i] = list(dict.fromkeys(found_matches['hit_poses'][i]))
                found_matches['hit_poses'][i].sort()
                found_matches['first_elem'].append(found_matches['hit_poses'][i][0])
                # del_list = []
                # for j in range(1, len(found_matches['hit_poses'][i])):
                #     if found_matches['hit_poses'][i][j] - found_matches['hit_poses'][i][j - 1] > 1:
                #         del_list.append(j)
                # if del_list:
                #     if 0 not in del_list:
                #         del_list.append(0)
                #     del_list.sort(reverse=True)
                #     for j in del_list:
                #         del found_matches['hit_poses'][i][j]
                del found_matches['hit_poses'][i][0]
                if not found_matches['hit_poses'][i]:
                    del_list2.append(i)
            if del_list2:
                for i in range(len(del_list2) - 1, -1, -1):
                    del found_matches['hit_poses'][del_list2[i]]
                    del found_matches['keys'][del_list2[i]]
                    del found_matches['first_elem'][del_list2[i]]
            # del_list = []
            # for i in range(0, len(found_matches['keys']) - 1):
            #     for j in range(i + 1, len(found_matches['keys'])):
            #         if found_matches['keys'][i] in found_matches['keys'][j]:
            #             if len(found_matches['hit_poses'][i]) >= len(found_matches['hit_poses'][j]):
            #                 del_list.append(j)
            #             else:
            #                 del_list.append(i)
            #         elif found_matches['keys'][j] in found_matches['keys'][i]:
            #             if len(found_matches['hit_poses'][j]) >= len(found_matches['hit_poses'][i]):
            #                 del_list.append(i)
            #             else:
            #                 del_list.append(j)
            # if del_list:
            #     del_list = list(dict.fromkeys(del_list))
            #     del_list.sort(reverse=True)
            #     for i in del_list:
            #         del found_matches['hit_poses'][i]
            #         del found_matches['keys'][i]
            #         del found_matches['first_elem'][i]
            if not found_matches['keys']:
                return join_char.join([a[:min(8, len(a))] for a in str_list_tmp])
            for i, it in enumerate(found_matches['keys']):
                shorty = ''
                if '_' == it[0]:
                    shorty += it[0:3]
                else:
                    shorty += it[0:2]
                shorty += '_'
                shorty = shorty.upper()
                for j in found_matches['hit_poses'][i]:
                    str_list_tmp[j] = str_list_tmp[j].replace(it, shorty)
            for i, it in enumerate(found_matches['keys']):
                l2 = min(6, len(it))
                start_pos_l[found_matches['first_elem'][i]].append((str_list_tmp[found_matches['first_elem'][i]].find(it), l2, len(it)))
                # str_list_tmp[found_matches['first_elem'][i]].replace(it[l2:], '')
        elems_new = []
        for i, it in enumerate(str_list_tmp):
            if start_pos_l[i]:
                #For later: if this whole procedure should be called in an iterated fashion, the last list index must
                #be iterated instaed of taking pos 0 (but track must be taken of the positions and lengths of found
                #matches in the original strings
                if start_pos_l[i][0][0] > 0:
                    tmp = it[:start_pos_l[i][0][0]]
                    tmp = tmp[:min(8, len(tmp))] + ('_' if tmp[-1] == '_' else '')
                    tmp += it[start_pos_l[i][0][0]: start_pos_l[i][0][0] + start_pos_l[i][0][1]].capitalize()
                else:
                    tmp = it[:start_pos_l[i][0][1]] + ('_' if it[start_pos_l[i][0][2] - 1] == '_' else '')
                tmp2 = it[start_pos_l[i][0][0] + start_pos_l[i][0][2]:].capitalize()
                tmp2 = tmp2[:min(8, len(tmp2))]
                elems_new.append(tmp + tmp2)
            else:
                elems_new.append(it[:min(8, len(it))])
        return join_char.join(elems_new)
    return join_char.join([a[:min(8, len(a))] for a in str_list_tmp])


def get_block_length_3D(df, xy_axis_columns):
    x_diff = float(df[xy_axis_columns[0]].iloc[0]) - float(df[xy_axis_columns[0]].iloc[1])
    y_diff = float(df[xy_axis_columns[1]].iloc[0]) - float(df[xy_axis_columns[1]].iloc[1])
    if np.isclose(x_diff, 0, atol=1e-06) and not np.isclose(y_diff, 0, atol=1e-06):
        nr_equal_ss = int(df.groupby(xy_axis_columns[0]).size().array[0])
    elif not np.isclose(x_diff, 0, atol=1e-06) and np.isclose(y_diff, 0, atol=1e-06):
        nr_equal_ss = int(df.groupby(xy_axis_columns[1]).size().array[0])
    else:
        nr_equal_ss1 = int(df.groupby(xy_axis_columns[0]).size().array[0])
        nr_equal_ss2 = int(df.groupby(xy_axis_columns[1]).size().array[0])
        nr_equal_ss = max(nr_equal_ss1, nr_equal_ss2)
    return nr_equal_ss


def is_exp_used(min_val, max_val, use_log=False):
    if use_log:
        return False
    if np.isclose(min_val, 0, atol=1e-06) and np.isclose(max_val, 0, atol=1e-06):
        return False
    elif np.isclose(min_val, 0, atol=1e-06):
        m_val = min(0, float(np.log10(np.abs(max_val))))
        m_val2 = max(0, float(np.log10(np.abs(max_val))))
    elif np.isclose(max_val, 0, atol=1e-06):
        m_val = min(float(np.log10(np.abs(min_val))), 0)
        m_val2 = max(float(np.log10(np.abs(min_val))), 0)
    else:
        m_val = min(float(np.log10(np.abs(min_val))), float(np.log10(np.abs(max_val))))
        m_val2 = max(float(np.log10(np.abs(min_val))), float(np.log10(np.abs(max_val))))
    if m_val < 0 and abs(m_val) > 1.01:
        return True
    elif m_val2 >= 4:
        return True
    return False


def enl_space_title(exp_value, title, df, x_axis_column, nr_plots, fig_type):
    if not exp_value:
        return False
    data_points = get_fig_x_size(df, x_axis_column, nr_plots, fig_type)
    text_l = len(title.split('\\\\')[-1])
    if fig_type == 'ybar' and data_points * 3 < text_l:
        return True
    elif text_l < 70:
        return False
    return True


def use_log_axis_and_exp_val(min_val, max_val, limit_min=None, limit_max=None):
    if limit_min and limit_max:
        use_log = use_log_axis(limit_min, limit_max)
        exp_value = is_exp_used(limit_min, limit_max, use_log)
    elif limit_min:
        use_log = use_log_axis(limit_min, max_val)
        exp_value = is_exp_used(limit_min, max_val, use_log)
    elif limit_max:
        use_log = use_log_axis(min_val, limit_max)
        exp_value = is_exp_used(min_val, limit_max, use_log)
    else:
        use_log = use_log_axis(min_val, max_val)
        exp_value = is_exp_used(min_val, max_val, use_log)
    return use_log, exp_value


def use_log_axis(min_val, max_val):
    if min_val < 0 or max_val < 0:
        use_log = False
    elif np.isclose(min_val, 0, atol=1e-06) and np.isclose(max_val, 0, atol=1e-06):
        use_log = False
    elif np.isclose(min_val, 0, atol=1e-06):
        use_log = True if np.abs(np.log10(np.abs(max_val))) > 1 else False
    elif np.isclose(max_val, 0, atol=1e-06):
        use_log = True if np.abs(np.log10(np.abs(min_val))) > 1 else False
    else:
        use_log = True if np.abs(np.log10(np.abs(min_val)) -
                                 np.log10(np.abs(max_val))) > 1 else False
    return use_log


def check_if_series(df):
    if 'modin.pandas' not in sys.modules:
        if isinstance(df, pd.DataFrame):
            return False
        else:
            return True
    else:
        try:
            if isinstance(df, pd.dataframe.DataFrame):
                return False
            else:
                return True
        except:
            if isinstance(df, pd.DataFrame):
                return False
            else:
                return True


def calc_limits(df, check_useless_data=False, no_big_limit=False, drop_cols = None, filter_pars=None, std_mult=None):
    use_limits = {'miny': None, 'maxy': None}
    is_series = False
    is_series = check_if_series(df)
    is_series_filter = False
    if filter_pars:
        if isinstance(filter_pars, str):
            is_series_filter = True
        else:
            try:
                oit = iter(filter_pars)
            except TypeError as te:
                is_series_filter = True
    mult_drops = False
    if drop_cols:
        if isinstance(drop_cols, str):
            mult_drops = True
        else:
            try:
                oit = iter(drop_cols)
            except TypeError as te:
                mult_drops = True
    if filter_pars and drop_cols:
        if is_series:
            if is_series_filter:
                warnings.warn('Unable to calculate limits including filtering with a '
                              'single label on a series', UserWarning)
                return False, None, use_limits
            else:
                for lbl in filter_pars:
                    if lbl not in df.index:
                        warnings.warn('Unable to calculate limits including '
                                      'filtering with unknown labels on a series', UserWarning)
                        return False, None, use_limits
            if mult_drops:
                for lbl in drop_cols:
                    if lbl not in df.index:
                        warnings.warn('Unable to calculate limits and '
                                      'dropping values with unknown labels on a series', UserWarning)
                        return False, None, use_limits
            elif drop_cols not in df.index:
                warnings.warn('Unable to calculate limits and '
                              'dropping values with unknown labels on a series', UserWarning)
                return False, None, use_limits
            stats_all = df.drop(drop_cols)[filter_pars].to_frame().stack().reset_index()
        elif is_series_filter:
            stats_all = df.drop(drop_cols)[filter_pars].to_frame().stack().reset_index()
            is_series_filter = False
        else:
            stats_all = df.drop(drop_cols, axis=1)[filter_pars].stack().reset_index()
    elif filter_pars:
        if is_series:
            if is_series_filter:
                warnings.warn('Unable to calculate limits including '
                              'filtering with a single label on a series', UserWarning)
                return False, None, use_limits
            else:
                for lbl in filter_pars:
                    if lbl not in df.index:
                        warnings.warn('Unable to calculate limits including '
                                      'filtering with unknown labels on a series', UserWarning)
                        return False, None, use_limits
            stats_all = df[filter_pars].to_frame().stack().reset_index()
        elif is_series_filter:
            stats_all = df[filter_pars].to_frame().stack().reset_index()
            is_series_filter = False
        else:
            stats_all = df[filter_pars].stack().reset_index()
    elif drop_cols:
        if is_series:
            if mult_drops:
                for lbl in drop_cols:
                    if lbl not in df.index:
                        warnings.warn('Unable to calculate limits and '
                                      'dropping values with unknown labels on a series', UserWarning)
                        return False, None, use_limits
            elif drop_cols not in df.index:
                warnings.warn('Unable to calculate limits and '
                              'dropping values with unknown labels on a series', UserWarning)
                return False, None, use_limits
            stats_all = df.drop(drop_cols).to_frame().stack().reset_index()
            is_series_filter = False
        else:
            stats_all = df.drop(drop_cols, axis=1).stack().reset_index()
    elif is_series:
        stats_all = df.to_frame().stack().reset_index()
        is_series_filter = False
    else:
        stats_all = df.stack().reset_index()
    stats_all = stats_all.drop(stats_all.columns[0:-1], axis=1).astype(float).describe().T
    if is_series_filter:
        min_val = stats_all['min']
        max_val = stats_all['max']
        mean_val = stats_all['mean']
        std_val = stats_all['std']
    else:
        min_val = stats_all['min'][0]
        max_val = stats_all['max'][0]
        mean_val = stats_all['mean'][0]
        std_val = stats_all['std'][0]

    if check_useless_data:
        if (np.isclose(min_val, 0, atol=1e-06) and
            np.isclose(max_val, 0, atol=1e-06)) or \
                np.isclose(min_val, max_val):
            return True, stats_all, use_limits
    if np.abs(max_val - min_val) < np.abs(max_val / 200):
        if min_val < 0:
            use_limits['miny'] = round(1.01 * min_val, 6)
        else:
            use_limits['miny'] = round(0.99 * min_val, 6)
        if max_val < 0:
            use_limits['maxy'] = round(0.99 * max_val, 6)
        else:
            use_limits['maxy'] = round(1.01 * max_val, 6)
    elif not no_big_limit and not np.isnan(std_val):
        mult = 2.576
        if std_mult:
            mult = std_mult
        if min_val < (mean_val - std_val * mult):
            use_limits['miny'] = round(mean_val - std_val * mult, 6)
        if max_val > (mean_val + std_val * mult):
            use_limits['maxy'] = round(mean_val + std_val * mult, 6)
    return False, stats_all, use_limits


def get_limits_log_exp(df,
                       no_big_limit=False,
                       no_limit_log=False,
                       check_useless_data=False,
                       drop_cols=None,
                       filter_pars=None,
                       std_mult=None):
    useless, stats_all, use_limits = calc_limits(df, check_useless_data, no_big_limit, drop_cols, filter_pars, std_mult)
    if useless:
        return True, use_limits, False, False
    is_series = False
    if stats_all is None:
        return False, use_limits, False, False
    if filter_pars:
        if isinstance(filter_pars, str):
            is_series = True
        else:
            try:
                oit = iter(filter_pars)
            except TypeError as te:
                is_series = True
    # if is_series:
    #     min_val = stats_all['min']
    #     max_val = stats_all['max']
    # else:
    min_val = stats_all['min'][0]
    max_val = stats_all['max'][0]
    if no_limit_log:
        use_log, exp_value = use_log_axis_and_exp_val(min_val,
                                                      max_val)
    else:
        use_log, exp_value = use_log_axis_and_exp_val(min_val,
                                                      max_val,
                                                      use_limits['miny'],
                                                      use_limits['maxy'])
    return False, use_limits, use_log, exp_value


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
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z}\\right]^{T}$', True)
    elif label == 't_diff_ty':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{y}=\\tilde{t}_{y}/\\lvert\\tilde{\\bm{t}}\\rvert '
                '-t_{y}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized y-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z}\\right]^{T}$', True)
    elif label == 't_diff_tz':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{z}=\\tilde{t}_{z}/\\lvert\\tilde{\\bm{t}}\\rvert '
                '-t_{z}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized z-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z}\\right]^{T}$', True)
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
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z}\\right]^{T}$', True)
    elif label == 't_diff_ty_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of differences '
                '$\\Delta t_{y}=\\tilde{t}_{y}/\\lvert\\tilde{\\bm{t}}\\rvert -t_{y}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized y-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z}\\right]^{T}$', True)
    elif label == 't_diff_tz_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of differences '
                '$\\Delta t_{z}=\\tilde{t}_{z}/\\lvert\\tilde{\\bm{t}}\\rvert -t_{z}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized z-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\tilde{\\bm{t}}=\\left[\\tilde{t}_{x},\\;\\tilde{t}_{y},\\;\\tilde{t}_{z}\\right]^{T}$', True)
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
                'Difference $\\Delta \\hat{t}_{x}=\\hat{t}_{x}/\\lvert\\hat{\\bm{t}}\\rvert '
                '-t_{x}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized x-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\hat{\\bm{t}}=\\left[\\hat{t}_{x},\\;\\hat{t}_{y},\\;\\hat{t}_{z}\\right]^{T}$. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_diff_ty':
        return (replaceCSVLabels(label),
                'Difference $\\Delta \\hat{t}_{y}=\\hat{t}_{y}/\\lvert\\hat{\\bm{t}}\\rvert '
                '-t_{y}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized y-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\hat{\\bm{t}}=\\left[\\hat{t}_{x},\\;\\hat{t}_{y},\\;\\hat{t}_{z}\\right]^{T}$. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_diff_tz':
        return (replaceCSVLabels(label),
                'Difference $\\Delta \\hat{t}_{z}=\\hat{t}_{z}/\\lvert\\hat{\\bm{t}}\\rvert '
                '-t_{z}^{GT}/\\lvert\\bm{t}^{GT}\\rvert$ '
                'between normalized z-components of ground truth and estimated relative stereo camera translation '
                'vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\hat{\\bm{t}}=\\left[\\hat{t}_{x},\\;\\hat{t}_{y},\\;\\hat{t}_{z}\\right]^{T}$. '
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
                '$\\Delta \\hat{t}_{x}=\\hat{t}_{x}/\\lvert\\hat{\\bm{t}}\\rvert -t_{x}^{GT}/\\lvert\\bm{t}^{GT}'
                '\\rvert$ between normalized x-components of ground truth and estimated relative stereo camera '
                'translation vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\hat{\\bm{t}}=\\left[\\hat{t}_{x},\\;\\hat{t}_{y},\\;\\hat{t}_{z}\\right]^{T}$. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_diff_ty_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of differences '
                '$\\Delta \\hat{t}_{y}=\\hat{t}_{y}/\\lvert\\hat{\\bm{t}}\\rvert -t_{y}^{GT}/\\lvert\\bm{t}^{GT}'
                '\\rvert$ between normalized y-components of ground truth and estimated relative stereo camera '
                'translation vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\hat{\\bm{t}}=\\left[\\hat{t}_{x},\\;\\hat{t}_{y},\\;\\hat{t}_{z}\\right]^{T}$. '
                'The latter was chosen as most accurate in a Monte Carlo similar fashion among a few '
                'estimated rotation matrices over the last pose estimations.', True
                )
    elif label == 't_mostLikely_diff_tz_diff':
        return (replaceCSVLabels(label),
                'Difference from frame to frame of differences '
                '$\\Delta \\hat{t}_{z}=\\hat{t}_{z}/\\lvert\\hat{\\bm{t}}\\rvert -t_{z}^{GT}/\\lvert\\bm{t}^{GT}'
                '\\rvert$ between normalized z-components of ground truth and estimated relative stereo camera '
                'translation vectors $\\bm{t}^{GT}=\\left[t^{GT}_{x},\\;t^{GT}_{y},\\;t^{GT}_{z}\\right]^{T}$ '
                'and $\\hat{\\bm{t}}=\\left[\\hat{t}_{x},\\;\\hat{t}_{y},\\;\\hat{t}_{z}\\right]^{T}$. '
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
                '$\\Delta^{2}R_{i}=\\Delta R_{\\Sigma ,i}-'
                '\\Delta R_{\\Sigma ,i-1}\\; \\forall i \\in \\left[ 1,\\; n_{I}\\; \\right]$ and '
                '$\\angle{\\Delta^{2} \\bm{t}_{i}}=\\angle{\\Delta \\bm{t}_{i}}-\\angle{\\Delta \\bm{t}_{i-1}}\\; '
                '\\forall i \\in \\left[ 1,\\; n_{I}\\; \\right]$ with '
                '$\\Delta e_{\\Sigma ,i}=\\text{sgn}\\left( \\Delta R_{\\Sigma ,i}\\right)\\Delta^{2}R_{i}+'
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
                'Difference from frame to frame on the number of matches $n_{pool}$ '
                'within the correspondence pool', True)
    elif label == 'R_GT_n_diffAll':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth (GT) relative stereo camera rotation matrices from '
                'frame to frame (calculated using quaternion notations)', True)
    elif label == 'R_GT_n_diff_roll_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth (GT) relative stereo camera rotation from frame to frame '
                'about the x-axis', True)
    elif label == 'R_GT_n_diff_pitch_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth (GT) relative stereo camera rotation from frame to frame '
                'about the y-axis', True)
    elif label == 'R_GT_n_diff_yaw_deg':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth (GT) relative stereo camera rotation from frame to frame '
                'about the z-axis', True)
    elif label == 't_GT_n_angDiff':
        return (replaceCSVLabels(label),
                'Angular difference between ground truth (GT) relative stereo camera translation vectors from '
                'frame to frame', True)
    elif label == 't_GT_n_elemDiff_tx':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{x,i}^{GT}=t_{x,i}^{GT}/\\lvert\\bm{t}^{GT}_{i}\\rvert '
                '-t_{x,i-1}^{GT}/\\lvert\\bm{t}^{GT}_{i-1}\\rvert$ '
                'between normalized x-components of ground truth relative stereo camera translation '
                'vectors $\\bm{t}^{GT}_{i}=\\left[t^{GT}_{x,i},\\;t^{GT}_{y,i},\\;t^{GT}_{z,i}\\right]^{T}$ '
                'for frame numbers $i$', True)
    elif label == 't_GT_n_elemDiff_ty':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{y,i}^{GT}=t_{y,i}^{GT}/\\lvert\\bm{t}^{GT}_{i}\\rvert '
                '-t_{y,i-1}^{GT}/\\lvert\\bm{t}^{GT}_{i-1}\\rvert$ '
                'between normalized y-components of ground truth relative stereo camera translation '
                'vectors $\\bm{t}^{GT}_{i}=\\left[t^{GT}_{x,i},\\;t^{GT}_{y,i},\\;t^{GT}_{z,i}\\right]^{T}$ '
                'for frame numbers $i$', True)
    elif label == 't_GT_n_elemDiff_tz':
        return (replaceCSVLabels(label),
                'Difference $\\Delta t_{z,i}^{GT}=t_{z,i}^{GT}/\\lvert\\bm{t}^{GT}_{i}\\rvert '
                '-t_{z,i-1}^{GT}/\\lvert\\bm{t}^{GT}_{i-1}\\rvert$ '
                'between normalized z-components of ground truth relative stereo camera translation '
                'vectors $\\bm{t}^{GT}_{i}=\\left[t^{GT}_{x,i},\\;t^{GT}_{y,i},\\;t^{GT}_{z,i}\\right]^{T}$ '
                'for frame numbers $i$', True)
    else:
        return (replaceCSVLabels(label), replaceCSVLabels(label), False)


def replaceCSVLabels(label, use_plural=False, str_capitalize=False, in_heading=False):
    if label == 'R_diffAll':
        return '$\\Delta R_{\\Sigma}$'
    elif label == 'R_diff_roll_deg':
        return '$\\Delta R_{x}$'
    elif label == 'R_diff_pitch_deg':
        return '$\\Delta R_{y}$'
    elif label == 'R_diff_yaw_deg':
        return '$\\Delta R_{z}$'
    elif label == 't_angDiff_deg':
        str_val = '$\\angle{\\Delta \\bm{t}}$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 't_distDiff':
        str_val = '$\\lvert\\Delta \\bm{t}\\rvert$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 't_diff_tx':
        return '$\\Delta t_{x}$'
    elif label == 't_diff_ty':
        return '$\\Delta t_{y}$'
    elif label == 't_diff_tz':
        return '$\\Delta t_{z}$'
    elif label == 'Rt_diff':
        str_val = '$e_{f\\left( R\\bm{t}\\right) }$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 'R_diffAll_diff':
        return '$\\Delta^{2} R_{\\Sigma}$'
    elif label == 'R_diff_roll_deg_diff':
        return '$\\Delta^{2} R_{x}$'
    elif label == 'R_diff_pitch_deg_diff':
        return '$\\Delta^{2} R_{y}$'
    elif label == 'R_diff_yaw_deg_diff':
        return '$\\Delta^{2} R_{z}$'
    elif label == 't_angDiff_deg_diff':
        str_val = '$\\angle{\\Delta^{2} \\bm{t}}$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 't_distDiff_diff':
        str_val = '$\\Delta\\lvert\\Delta \\bm{t}\\rvert$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 't_diff_tx_diff':
        return '$\\Delta^{2} t_{x}$'
    elif label == 't_diff_ty_diff':
        return '$\\Delta^{2} t_{y}$'
    elif label == 't_diff_tz_diff':
        return '$\\Delta^{2} t_{z}$'
    elif label == 'Rt_diff_diff':
        str_val = '$\\Delta e_{f\\left( R\\bm{t}\\right) }$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 'Rt_diff2':
        str_val = '$\\Delta e_{R\\bm{t}}$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
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
        str_val = '$\\angle{\\Delta \\hat{\\bm{t}}}$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 't_mostLikely_distDiff':
        str_val = '$\\lvert\\Delta \\hat{\\bm{t}}\\rvert$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 't_mostLikely_diff_tx':
        return '$\\Delta \\hat{t}_{x}$'
    elif label == 't_mostLikely_diff_ty':
        return '$\\Delta \\hat{t}_{y}$'
    elif label == 't_mostLikely_diff_tz':
        return '$\\Delta \\hat{t}_{z}$'
    elif label == 'Rt_mostLikely_diff':
        str_val = '$\\hat{e}_{f\\left( \\hat{R}\\hat{\\bm{t}}\\right) }$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 'R_mostLikely_diffAll_diff':
        return '$\\Delta^{2} \\hat{R}_{\\Sigma}$'
    elif label == 'R_mostLikely_diff_roll_deg_diff':
        return '$\\Delta^{2} \\hat{R}_{x}$'
    elif label == 'R_mostLikely_diff_pitch_deg_diff':
        return '$\\Delta^{2} \\hat{R}_{y}$'
    elif label == 'R_mostLikely_diff_yaw_deg_diff':
        return '$\\Delta^{2} \\hat{R}_{z}$'
    elif label == 't_mostLikely_angDiff_deg_diff':
        str_val = '$\\angle{\\Delta^{2} \\hat{\\bm{t}}}$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 't_mostLikely_distDiff_diff':
        str_val = '$\\Delta\\lvert\\Delta \\hat{\\bm{t}}\\rvert$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 't_mostLikely_diff_tx_diff':
        return '$\\Delta^{2} \\hat{t}_{x}$'
    elif label == 't_mostLikely_diff_ty_diff':
        return '$\\Delta^{2} \\hat{t}_{y}$'
    elif label == 't_mostLikely_diff_tz_diff':
        return '$\\Delta^{2} \\hat{t}_{z}$'
    elif label == 'Rt_mostLikely_diff_diff':
        str_val = '$\\Delta\\hat{e}_{f\\left( \\hat{R}\\hat{\\bm{t}}\\right) }$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 'R_GT_n_diffAll':
        return '$\\Delta R_{\\Sigma}^{GT}$'
    elif label == 't_GT_n_angDiff':
        str_val = '$\\angle{\\Delta \\bm{t}^{GT}}$'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
        return str_val
    elif label == 'R_GT_n_diff_roll_deg':
        return '$\\Delta R_{x}^{GT}$'
    elif label == 'R_GT_n_diff_pitch_deg':
        return '$\\Delta R_{y}^{GT}$'
    elif label == 'R_GT_n_diff_yaw_deg':
        return '$\\Delta R_{z}^{GT}$'
    elif label == 't_GT_n_elemDiff_tx':
        return '$\\Delta t_{x}^{GT}$'
    elif label == 't_GT_n_elemDiff_ty':
        return '$\\Delta t_{y}^{GT}$'
    elif label == 't_GT_n_elemDiff_tz':
        return '$\\Delta t_{z}^{GT}$'
    elif label == 'rt_change_type':
        if use_plural:
            str_val = 'types of $R^{GT}$ \\& $\\bm{t}^{GT}$ changes on a frame to frame basis'
        else:
            str_val = 'type of $R^{GT}$ \\& $\\bm{t}^{GT}$ change on a frame to frame basis'
        if in_heading:
            str_val = replace_bm_in_headings(str_val)
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
            str_val = '\\# filtered correspondences $n_{fc}$'
    elif label == 'nrCorrs_estimated':
        if use_plural:
            str_val = 'numbers of estimated correspondences $\\tilde{n}_{c}$'
        else:
            str_val = '\\# estimated correspondences $\\tilde{n}_{c}$'
    elif label == 'nrCorrs_GT':
        if use_plural:
            str_val = 'numbers of GT correspondences $n_{GT}$'
        else:
            str_val = '\\# GT correspondences $n_{GT}$'
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
    elif label == 'linRef_BA_sac_us':
        if use_plural:
            str_val = 'robust estimation, linear refinement, and BA times $t_{e,l,BA}=t_{e}+t_{l}+t_{BA}$'
        else:
            str_val = 'robust estimation, linear refinement, and BA time $t_{e,l,BA}=t_{e}+t_{l}+t_{BA}$'
    elif label == 'comp_time':
        if use_plural:
            str_val = 'execution times $t_{c}$'
        else:
            str_val = 'execution time $t_{c}$'
    elif label == 'poolSize':
        if use_plural:
            str_val = 'numbers of matches $n_{pool}$ within the correspondence pool'
        else:
            str_val = '\\# of matches $n_{pool}$ within the correspondence pool'
    elif label == 'poolSize_diff':
        str_val = '$\\Delta n_{pool}$'
    elif label == 'Nr':
        if use_plural:
            str_val = 'consecutive stereo frame numbers'
        else:
            str_val = 'consecutive stereo frame number'
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
            return '\\# TP'
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
    if in_heading:
        str_val = replace_bm_in_headings(str_val)
    ex = ['and', 'of', 'for', 'to', 'with', 'on', 'in', 'within']
    if str_capitalize:
        return ' '.join([b.capitalize() if not b.isupper() and
                                           not '$' in b and
                                           not '\\' in b and
                                           not b in ex else b for b in str_val.split(' ')])
    else:
        return str_val


def capitalizeStr(str_val):
    ex = ['and', 'of', 'for', 'to', 'with', 'on', 'in', 'within']
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
    b = str_l[0]
    if b.isupper() or '$' in b or '\\' in b or len(b) == 1 or '{' in b or '}' in b or '_' in b:
        return str_val
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
    elif key == 'crt':
        return 'Continuous change in ground truth relative stereo rotation (all axis) and ' \
               'translation (every axis) from frame to frame', True
    elif key == 'cra':
        return 'Continuous change in ground truth relative stereo rotation only (all axis) from frame to frame', True
    elif key == 'cta':
        return 'Continuous change in ground truth relative translation only (every axis) from frame to frame', True
    elif key == 'crx':
        return 'Continuous change in ground truth relative stereo rotation around x-axis only ' \
               'from frame to frame', True
    elif key == 'cry':
        return 'Continuous change in ground truth relative stereo rotation around y-axis only ' \
               'from frame to frame', True
    elif key == 'crz':
        return 'Continuous change in ground truth relative stereo rotation around z-axis only ' \
               'from frame to frame', True
    elif key == 'ctx':
        return 'Continuous change in ground truth relative stereo translation in x-direction only ' \
               'from frame to frame', True
    elif key == 'cty':
        return 'Continuous change in ground truth relative stereo translation in y-direction only ' \
               'from frame to frame', True
    elif key == 'ctz':
        return 'Continuous change in ground truth relative stereo translation in z-direction only ' \
               'from frame to frame', True
    elif key == 'jrt':
        return 'Single jump of the ground truth relative stereo rotation (all axis) and ' \
               'translation (every axis) to different values after a few frames', True
    elif key == 'jra':
        return 'Single jump of the ground truth relative stereo rotation only (all axis) ' \
               'to different values after a few frames', True
    elif key == 'jta':
        return 'Single jump of the ground truth relative stereo translation only (every axis) ' \
               'to different values after a few frames', True
    elif key == 'jrx':
        return 'Single jump of the ground truth relative stereo rotation about x-axis only ' \
               'to a different value after a few frames', True
    elif key == 'jry':
        return 'Single jump of the ground truth relative stereo rotation about y-axis only ' \
               'to a different value after a few frames', True
    elif key == 'jrz':
        return 'Single jump of the ground truth relative stereo rotation about z-axis only ' \
               'to a different value after a few frames', True
    elif key == 'jtx':
        return 'Single jump of the ground truth relative stereo translation in x-direction only ' \
               'to a different value after a few frames', True
    elif key == 'jty':
        return 'Single jump of the ground truth relative stereo translation in y-direction only ' \
               'to a different value after a few frames', True
    elif key == 'jtz':
        return 'Single jump of the ground truth relative stereo translation in z-direction only ' \
               'to a different value after a few frames', True
    elif key == 'nv':
        return 'No variation in ground truth relative stereo rotation and translation ' \
               'over all available stereo frames', True
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


def get_fig_x_size(df, x_axis_column, nr_plots, fig_type):
    if not isinstance(x_axis_column, str):
        try:
            oit = iter(x_axis_column)
            x_axis_column = x_axis_column[0]
        except TypeError as te:
            pass
    data_points = df.reset_index()[x_axis_column].nunique()
    if fig_type == 'xbar' or fig_type == 'ybar':
        data_points *= int(nr_plots)
    return data_points


def check_legend_enlarge(df, x_axis_column, nr_plots, fig_type):
    data_points = get_fig_x_size(df, x_axis_column, nr_plots, fig_type)
    if fig_type == 'xbar' and data_points < 20:
        dist = -0.04 * float(data_points) + 0.73
        if dist < 0.15:
            dist = 0.15
        return dist
    return None


#Only for testing
def main():
    num_pts = int(10000)
    nr_imgs = 150
    pars1_opt = ['first_long_long_opt' + str(i) for i in range(0, 2)]
    pars2_opt = ['second_long_opt' + str(i) for i in range(0, 3)]
    pars3_opt = ['third_long_long_opt' + str(i) for i in range(0, 2)]
    gt_type_pars = ['crt', 'cra', 'cta', 'crx', 'cry', 'crz', 'ctx', 'cty', 'ctz',
                    'jrt', 'jra', 'jta', 'jrx', 'jry', 'jrz', 'jtx', 'jty', 'jtz']
    pars_kpDistr_opt = ['1corn', 'equ']
    # pars_depthDistr_opt = ['NMF', 'NM', 'F']
    pars_depthDistr_opt = ['NMF', 'NM']
    pars_nrTP_opt = ['500', '100to1000']
    pars_kpAccSd_opt = ['0.5', '1.0', '1.5']
    inlratMin_opt = list(map(str, list(np.arange(0.35, 0.85, 0.1))))
    lin_time_pars = np.array([500, 3, 0.003])
    poolSize = [10000, 40000]
    min_pts = len(pars_kpAccSd_opt) * len(pars_depthDistr_opt) * len(inlratMin_opt) * \
              nr_imgs * len(poolSize) * len(pars1_opt)
    if min_pts < num_pts:
        while min_pts < num_pts:
            pars_kpAccSd_opt += [str(float(pars_kpAccSd_opt[-1]) + 0.5)]
            min_pts = len(pars_kpAccSd_opt) * len(pars_depthDistr_opt) * len(inlratMin_opt) * \
                      nr_imgs * len(poolSize) * len(pars1_opt)
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
            'kpAccSd': np.array(build_list(pars_kpAccSd_opt, kpAccSd_mul, num_pts)),
            'USAC_parameters_USACInlratFilt': [pars3_opt[i] for i in np.random.randint(0, len(pars3_opt), num_pts)],
            'th': np.tile(np.arange(0.4, 0.9, 0.1), int(num_pts/5)),
            # 'inlratMin': np.tile(np.arange(0.05, 0.45, 0.1), int(num_pts/4)),
            'inlratMin': np.array(build_list(inlratMin_opt, inlratMin_mul, num_pts)),
            'useless': [1, 1, 2, 3] * int(num_pts/4),
            # 'R_out(0,0)': [0] * 10 + [1] * int(num_pts - 10),
            # 'R_out(0,1)': [0] * 10 + [0] * int(num_pts - 10),
            # 'R_out(0,2)': [float(0)] * 10 + [0.1] * int(num_pts - 10),
            # 'R_out(1,0)': [0] * 10 + [1] * int(num_pts - 10),
            # 'R_out(1,1)': [0] * 10 + [1] * int(num_pts - 10),
            # 'R_out(1,2)': [0] * 10 + [0] * int(num_pts - 10),
            # 'R_out(2,0)': [float(0)] * 10 + [0.1] * int(num_pts - 10),
            # 'R_out(2,1)': [0] * 10 + [0] * int(num_pts - 10),
            # 'R_out(2,2)': [0] * 10 + [0] * int(num_pts - 10),
            'Nr': list(range(0, nr_imgs)) * int(num_pts / nr_imgs),
            'inlRat_GT': np.tile(np.arange(0.25, 0.72, 0.05), int(num_pts/10))}

    eval_columns = ['K1_cxyfxfyNorm', 'K2_cxyfxfyNorm', 'K1_cxyDiffNorm', 'K2_cxyDiffNorm',
                    'K1_fxyDiffNorm', 'K2_fxyDiffNorm', 'K1_fxDiff', 'K2_fxDiff', 'K1_fyDiff',
                    'K2_fyDiff', 'K1_cxDiff', 'K2_cxDiff', 'K1_cyDiff', 'K2_cyDiff']

    data['poolSize'] = data['inlratMin'].astype(np.float) * 500 - \
                       data['inlratMin'].astype(np.float) * np.random.randint(0, 50, num_pts)
    data['poolSize'] *= (np.array(data['Nr']) + 1) * \
                        np.exp(-1 * (np.array(data['Nr']) + 1) * data['inlratMin'].astype(np.float) /
                               (data['kpAccSd'].astype(np.float) * (5 * nr_imgs / 3)))
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
    t *= (data['inlratMin'].astype(np.float).max() / data['inlratMin'].astype(np.float)) ** 2
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
    nr_scenes = int(num_pts / nr_imgs)
    data['inlratCRate'] = data['inlratMin']
    R0s_scene = np.random.randint(0, 5, nr_scenes)
    R0s = [list(np.random.randint(0, nr_imgs, it)) if it > 0 else [] for it in R0s_scene]
    data['R_out(0,0)'] = [1] * int(num_pts)
    data['R_out(0,1)'] = [1] * int(num_pts)
    data['R_out(0,2)'] = [0.1] * int(num_pts)
    data['R_out(1,0)'] = [1] * int(num_pts)
    data['R_out(1,1)'] = [1] * int(num_pts)
    data['R_out(1,2)'] = [1] * int(num_pts)
    data['R_out(2,0)'] = [0.1] * int(num_pts)
    data['R_out(2,1)'] = [1] * int(num_pts)
    data['R_out(2,2)'] = [1] * int(num_pts)
    for i, it in enumerate(R0s):
        if it:
            for it1 in it:
                data['R_out(0,0)'][i * nr_imgs + it1] = 0
                data['R_out(0,1)'][i * nr_imgs + it1] = 0
                data['R_out(0,2)'][i * nr_imgs + it1] = 0
                data['R_out(1,0)'][i * nr_imgs + it1] = 0
                data['R_out(1,1)'][i * nr_imgs + it1] = 0
                data['R_out(1,2)'][i * nr_imgs + it1] = 0
                data['R_out(2,0)'][i * nr_imgs + it1] = 0
                data['R_out(2,1)'][i * nr_imgs + it1] = 0
                data['R_out(2,2)'][i * nr_imgs + it1] = 0
    data_type = build_list(gt_type_pars, 1, nr_scenes)
    data['R_GT_n_diffAll'] = []
    data['t_GT_n_angDiff'] = []
    data['R_GT_n_diff_roll_deg'] = []
    data['R_GT_n_diff_pitch_deg'] = []
    data['R_GT_n_diff_yaw_deg'] = []
    data['t_GT_n_elemDiff_tx'] = []
    data['t_GT_n_elemDiff_ty'] = []
    data['t_GT_n_elemDiff_tz'] = []
    jumppos = int(0.3 * nr_imgs)
    jumpposn1 = jumppos - 1
    jumpposn2 = int(nr_imgs) - jumppos
    c_max = 0.01 * nr_imgs
    sc = 0
    tl = len(data_type)
    while sc < nr_scenes:
        this_type = data_type[sc % tl]
        if this_type == 'crt':
            data['R_GT_n_diffAll'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_angDiff'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_roll_deg'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_pitch_deg'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_yaw_deg'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_elemDiff_tx'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_elemDiff_ty'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_elemDiff_tz'] += list(np.arange(0, c_max, 0.01))
        elif this_type == 'cra':
            data['R_GT_n_diffAll'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_angDiff'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_roll_deg'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_pitch_deg'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_yaw_deg'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'crx':
            data['R_GT_n_diffAll'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_angDiff'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_roll_deg'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'cry':
            data['R_GT_n_diffAll'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_angDiff'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'crz':
            data['R_GT_n_diffAll'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_angDiff'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'cta':
            data['R_GT_n_diffAll'] += [0] * int(nr_imgs)
            data['t_GT_n_angDiff'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_elemDiff_ty'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_elemDiff_tz'] += list(np.arange(0, c_max, 0.01))
        elif this_type == 'ctx':
            data['R_GT_n_diffAll'] += [0] * int(nr_imgs)
            data['t_GT_n_angDiff'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'cty':
            data['R_GT_n_diffAll'] += [0] * int(nr_imgs)
            data['t_GT_n_angDiff'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += list(np.arange(0, c_max, 0.01))
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'ctz':
            data['R_GT_n_diffAll'] += [0] * int(nr_imgs)
            data['t_GT_n_angDiff'] += list(np.arange(0, c_max, 0.01))
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += list(np.arange(0, c_max, 0.01))
        elif this_type == 'jrt':
            data['R_GT_n_diffAll'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_angDiff'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_roll_deg'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_pitch_deg'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_yaw_deg'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_elemDiff_tx'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_elemDiff_ty'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_elemDiff_tz'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
        elif this_type == 'jra':
            data['R_GT_n_diffAll'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_angDiff'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_roll_deg'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_pitch_deg'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_yaw_deg'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'jrx':
            data['R_GT_n_diffAll'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_angDiff'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_roll_deg'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'jry':
            data['R_GT_n_diffAll'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_angDiff'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'jrz':
            data['R_GT_n_diffAll'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_angDiff'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'jta':
            data['R_GT_n_diffAll'] += [0] * int(nr_imgs)
            data['t_GT_n_angDiff'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_elemDiff_ty'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_elemDiff_tz'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
        elif this_type == 'jtx':
            data['R_GT_n_diffAll'] += [0] * int(nr_imgs)
            data['t_GT_n_angDiff'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'jty':
            data['R_GT_n_diffAll'] += [0] * int(nr_imgs)
            data['t_GT_n_angDiff'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['t_GT_n_elemDiff_tz'] += [0] * int(nr_imgs)
        elif this_type == 'jtz':
            data['R_GT_n_diffAll'] += [0] * int(nr_imgs)
            data['t_GT_n_angDiff'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
            data['R_GT_n_diff_roll_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_pitch_deg'] += [0] * int(nr_imgs)
            data['R_GT_n_diff_yaw_deg'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tx'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_ty'] += [0] * int(nr_imgs)
            data['t_GT_n_elemDiff_tz'] += [0] * jumpposn1 + [1] + [0] * jumpposn2
        else:
            raise ValueError('Invalid change type')
        sc += 1


    data = pd.DataFrame(data)

    test_name = 'robustness'#'correspondence_pool'#'refinement_ba_stereo'#'vfc_gms_sof'#'refinement_ba'#'usac_vs_ransac'#'testing_tests'
    test_nr = 1
    eval_nr = [1]#list(range(5, 11))
    ret = 0
    output_path = '/home/maierj/work/Sequence_Test/py_test'
    # output_path = '/home/maierj/work/Sequence_Test/py_test/refinement_ba/1'
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
                evals = list(range(1, 6))
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
                elif ev == 4:
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
                                          'func_name': 'get_best_comb_kpAccSd_1',
                                          'res_par_name': 'refineRT_BA_opts_kpAccSd'}
                    from usac_eval import get_best_comb_inlrat_1
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_refineRT_BA_opts_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['kpAccSd'],
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
                elif ev == 5:
                    fig_title_pre_str = 'Statistics on Execution for Comparison of '
                    eval_columns = ['linRef_BA_sac_us']
                    units = [('linRef_BA_sac_us', '/$\\mu s$')]
                    # it_parameters = ['refineMethod_algorithm',
                    #                  'refineMethod_costFunction',
                    #                  'BART']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    from refinement_eval import filter_nr_kps_calc_t_all
                    ret += calcSatisticAndPlot_aggregate(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_refineRT_BA_opts_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='time-agg',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         pdfsplitentry=None,
                                                         filter_func=filter_nr_kps_calc_t_all,
                                                         filter_func_args=None,
                                                         special_calcs_func=None,
                                                         special_calcs_args=None,
                                                         calc_func=None,
                                                         calc_func_args=None,
                                                         compare_source=None,
                                                         fig_type='xbar',
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
        else:
            raise ValueError('Test nr does not exist')
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
                    # it_parameters = ['stereoParameters_refineMethod_algorithm',
                    #                  'stereoParameters_refineMethod_costFunction',
                    #                  'stereoParameters_BART']
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
        else:
            raise ValueError('Test nr does not exist')
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
                    # descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                    #         'multiple stereo frames'
                    # comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                    #              'USAC_parameters_refinealg-second_long_opt0']
                    # compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                                                  compare_source=None,#compare_source,
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
                    # descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                    #         'multiple stereo frames'
                    # comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                    #              'USAC_parameters_refinealg-second_long_opt0']
                    # compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                                                             compare_source=None,#compare_source,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 3:
                    fig_title_pre_str = 'Values of R\\&t Differences over the Last 30 out of 150 Frames ' \
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
                    # descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                    #         'multiple stereo frames'
                    # comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                    #              'USAC_parameters_refinealg-second_long_opt0']
                    # compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                                                             compare_source=None,#compare_source,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 4:
                    fig_title_pre_str = 'R\\&t Differences from Frame to Frame with a Maximum Correspondence Pool ' \
                                        'Size of $\\hat{n}_{cp}=40000$ Features for Different '
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
                                                             use_marks=False,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 5:
                    fig_title_pre_str = 'Differences of R\\&t Differences from Frame to Frame with a Maximum ' \
                                        'Correspondence Pool Size of $\\hat{n}_{cp}=40000$ Features for Different '
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
                                      'keepEval': ['poolSize', 'R_diffAll', 't_angDiff_deg'],
                                      'eval_on': ['poolSize'],
                                      'diff_by': 'Nr'}
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'partition_x_axis': 'kpAccSd',
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
                                                             use_marks=False,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 6:
                    fig_title_pre_str = 'Values on R\\&t Differences from Frame to Frame with a Maximum ' \
                                        'Correspondence Pool Size of $\\hat{n}_{cp}=40000$ Features for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    calc_func_args = {'eval_on': ['poolSize']}
                    # it_parameters = ['stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator']
                    from corr_pool_eval import filter_max_pool_size, \
                        calc_rt_diff_n_matches
                    ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_corrPool_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-diff',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['poolSize', 'inlratMin'],
                                                  filter_func=filter_max_pool_size,
                                                  filter_func_args=None,
                                                  special_calcs_func=None,
                                                  special_calcs_args=None,
                                                  calc_func=calc_rt_diff_n_matches,
                                                  calc_func_args=calc_func_args,
                                                  fig_type='surface',
                                                  use_marks=False,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 7:
                    fig_title_pre_str = 'Statistics on R\\&t Differences from Frame to Frame with a Maximum ' \
                                        'Correspondence Pool Size of $\\hat{n}_{cp}=40000$ Features for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    calc_func_args = {'eval_on': ['poolSize']}
                    # it_parameters = ['stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator']
                    from corr_pool_eval import filter_max_pool_size, \
                        calc_rt_diff_n_matches
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_corrPool_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-diff',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['poolSize'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=filter_max_pool_size,
                                                  filter_func_args=None,
                                                  special_calcs_func=None,
                                                  special_calcs_args=None,
                                                  calc_func=calc_rt_diff_n_matches,
                                                  calc_func_args=calc_func_args,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=False,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 8:
                    fig_title_pre_str = 'Differences on Frame to Frame Statistics of R\\&t Errors with a Maximum ' \
                                        'Correspondence Pool Size of $\\hat{n}_{cp}=40000$ Features for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator']
                    calc_func_args = {'data_separators': ['poolSize'],
                                      'keepEval': ['R_diffAll', 't_angDiff_deg'],
                                      'eval_on': ['poolSize'],
                                      'diff_by': 'poolSize'}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'res_par_name': 'corrpool_size_converge_mean'}
                    from corr_pool_eval import filter_max_pool_size, \
                        calc_diff_stat_rt_diff_n_matches, eval_corr_pool_converge_vs_x
                    ret += calcFromFuncAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_corrPool_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-diff',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['poolSize'],
                                                  filter_func=filter_max_pool_size,
                                                  filter_func_args=None,
                                                  special_calcs_func=eval_corr_pool_converge_vs_x,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=calc_diff_stat_rt_diff_n_matches,
                                                  calc_func_args=calc_func_args,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=False,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 9:
                    fig_title_pre_str = 'Statistics on Execution Times for Comparison of '
                    eval_columns = ['stereoRefine_us']
                    units = [('stereoRefine_us', '/$\\mu s$')]
                    # it_parameters = ['stereoParameters_maxPoolCorrespondences',
                    #                  'stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    ret += calcSatisticAndPlot_aggregate(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_corrPool_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='time',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         pdfsplitentry=None,
                                                         filter_func=None,
                                                         filter_func_args=None,
                                                         special_calcs_func=None,
                                                         special_calcs_args=None,
                                                         calc_func=None,
                                                         calc_func_args=None,
                                                         compare_source=None,
                                                         fig_type='xbar',
                                                         use_marks=False,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=False)
                elif ev == 10:
                    fig_title_pre_str = 'Statistics on Execution Times over the Last 30 Stereo Frames ' \
                                        'out of 150 Frames for Comparison of '
                    eval_columns = ['stereoRefine_us']
                    units = [('stereoRefine_us', '/$\\mu s$')]
                    # it_parameters = ['stereoParameters_maxPoolCorrespondences',
                    #                  'stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator',
                                     'stereoParameters_maxPoolCorrespondences']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False}
                    from corr_pool_eval import filter_take_end_frames, eval_mean_time_poolcorrs
                    ret += calcSatisticAndPlot_aggregate(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_corrPool_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='time',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         pdfsplitentry=None,
                                                         filter_func=filter_take_end_frames,
                                                         filter_func_args=None,
                                                         special_calcs_func=eval_mean_time_poolcorrs,
                                                         special_calcs_args=special_calcs_args,
                                                         calc_func=None,
                                                         calc_func_args=None,
                                                         compare_source=None,
                                                         fig_type='xbar',
                                                         use_marks=False,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=False)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        elif test_nr == 2:
            if eval_nr[0] < 0:
                evals = list(range(11, 14))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 11:
                    fig_title_pre_str = 'Statistics on R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_maxRat3DPtsFar',
                    #                  'stereoParameters_maxDist3DPtsZ']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_rat_dist_3Dpts_inlrat'}
                    # descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                    #         'multiple stereo frames'
                    # comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                    #              'USAC_parameters_refinealg-second_long_opt0']
                    # compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                                                  compare_source=None,#compare_source,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 12:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_maxRat3DPtsFar',
                    #                  'stereoParameters_maxDist3DPtsZ']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_rat_dist_3Dpts_best_comb_scenes'}
                    # descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                    #         'multiple stereo frames'
                    # comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                    #              'USAC_parameters_refinealg-second_long_opt0']
                    # compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                                                             compare_source=None,#compare_source,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True)
                elif ev == 13:
                    fig_title_pre_str = 'Statistics on Execution Times over the Last 30 Stereo Frames ' \
                                        'out of 150 Frames for Comparison of '
                    eval_columns = ['stereoRefine_us']
                    units = [('stereoRefine_us', '/$\\mu s$')]
                    # it_parameters = ['stereoParameters_maxRat3DPtsFar',
                    #                  'stereoParameters_maxDist3DPtsZ']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False}
                    from corr_pool_eval import filter_take_end_frames, eval_mean_time_pool_3D_dist
                    ret += calcSatisticAndPlot_aggregate(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_corrPool_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='time',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         pdfsplitentry=None,
                                                         filter_func=filter_take_end_frames,
                                                         filter_func_args=None,
                                                         special_calcs_func=eval_mean_time_pool_3D_dist,
                                                         special_calcs_args=special_calcs_args,
                                                         calc_func=None,
                                                         calc_func_args=None,
                                                         compare_source=None,
                                                         fig_type='xbar',
                                                         use_marks=False,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=False)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        elif test_nr == 3:
            if eval_nr[0] < 0:
                evals = list(range(14, 16))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 14:
                    fig_title_pre_str = 'Statistics on R\\&t Differences for Specific Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    # it_parameters = ['stereoParameters_maxRat3DPtsFar',
                    #                  'stereoParameters_maxDist3DPtsZ',
                    #                  'stereoParameters_maxPoolCorrespondences',
                    #                  'stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'RT-stats', descr)
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_corrPool_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['kpAccSd'],
                                                  pdfsplitentry=['t_distDiff'],
                                                  filter_func=None,
                                                  filter_func_args=None,
                                                  special_calcs_func=None,
                                                  special_calcs_args=None,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=compare_source,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True)
                elif ev == 15:
                    fig_title_pre_str = 'Statistics on Execution Times for Specific Combinations of '
                    eval_columns = ['stereoRefine_us']
                    units = [('stereoRefine_us', '/$\\mu s$')]
                    # it_parameters = ['stereoParameters_maxRat3DPtsFar',
                    #                  'stereoParameters_maxDist3DPtsZ',
                    #                  'stereoParameters_maxPoolCorrespondences',
                    #                  'stereoParameters_minPtsDistance']
                    it_parameters = ['USAC_parameters_estimator',
                                     'USAC_parameters_refinealg']
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    comp_pars = ['USAC_parameters_estimator-first_long_long_opt1',
                                 'USAC_parameters_refinealg-second_long_opt0']
                    repl_eval = {'actual': ['stereoRefine_us'], 'old': ['linRef_BA_sac_us'], 'new': ['comp_time']}
                    compare_source = get_compare_info(comp_pars, output_path, 'refinement_ba', 1, 'time-agg', descr,
                                                      repl_eval)
                    from usac_eval import filter_nr_kps_stat
                    ret += calcSatisticAndPlot_aggregate(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_corrPool_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='time',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         pdfsplitentry=None,
                                                         filter_func=filter_nr_kps_stat,
                                                         filter_func_args=None,
                                                         special_calcs_func=None,
                                                         special_calcs_args=None,
                                                         calc_func=None,
                                                         calc_func_args=None,
                                                         compare_source=compare_source,
                                                         fig_type='xbar',
                                                         use_marks=False,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=False)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        else:
            raise ValueError('Test nr does not exist')
    elif test_name == 'robustness':
        if not test_nr:
            raise ValueError('test_nr is required robustness')
        from eval_tests_main import get_compare_info
        if test_nr == 1:
            if eval_nr[0] < 0:
                evals = list(range(1, 11))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 1:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['stereoParameters_relInlRatThLast',
                                     'stereoParameters_relInlRatThNew',
                                     'stereoParameters_minInlierRatSkip',
                                     'stereoParameters_minInlierRatioReInit',
                                     'stereoParameters_relMinInlierRatSkip']
                    it_parameters = ['USAC_parameters_estimator',
                                     'stereoParameters_maxPoolCorrespondences']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'robustness_best_comb_scenes_inlc'}
                    # filter_func_args = {'data_seperators': ['stereoParameters_relInlRatThLast',
                    #                                         'stereoParameters_relInlRatThNew',
                    #                                         'stereoParameters_minInlierRatSkip',
                    #                                         'stereoParameters_minInlierRatioReInit',
                    #                                         'stereoParameters_relMinInlierRatSkip',
                    #                                         'inlratCRate']}
                    filter_func_args = {'data_seperators': ['USAC_parameters_estimator',
                                                            'stereoParameters_maxPoolCorrespondences',
                                                            'inlratCRate']}
                    from refinement_eval import get_best_comb_scenes_1
                    from robustness_eval import get_rt_change_type, get_best_comb_scenes_1
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratCRate'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_best_comb_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True)

    return ret


if __name__ == "__main__":
    main()