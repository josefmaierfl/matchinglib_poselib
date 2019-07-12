"""
Evaluates results from the autocalibration present in a pandas DataFrame as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np, math
import ruamel.yaml as yaml
import modin.pandas as pd
import pandas as pd1
#from jinja2 import Template as ji
import jinja2 as ji
# import tempfile
# import shutil
from copy import deepcopy

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


def compile_tex(rendered_tex, out_tex_dir, out_tex_file, make_fig_index=True, out_pdf_filen=None):
    texdf = os.path.join(out_tex_dir, out_tex_file)
    with open(texdf, 'w') as outfile:
        outfile.write(rendered_tex)
    if out_pdf_filen is not None:
        rep_make = 1
        if make_fig_index:
            rep_make = 2
        # tmp_dir = tempfile.mkdtemp()
        #shutil.copy(texdf, tmp_dir)
        #tex_tmp = os.path.join(tmp_dir, out_tex_file)
        pdfpath, pdfname = os.path.split(out_pdf_filen)
        pdfname = os.path.splitext(pdfname)[0]
        stdoutf = os.path.join(pdfpath, 'stdout_' + pdfname + '.txt')
        erroutf = os.path.join(pdfpath, 'error_' + pdfname + '.txt')
        # out_tmp_path = os.path.join(tmp_dir, pdfname)
        cmdline = ['pdflatex',
                   '--jobname=' + pdfname,
                   '--output-directory=' + pdfpath,
                   '--interaction=nonstopmode',
                   texdf]
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
        # shutil.copy(out_tmp_path, pdfpath)
        # shutil.rmtree(tmp_dir)
        return retcode


def calcSatisticRt_th(data, store_path, make_fig_index=True, build_pdf=False):
    if type(data) is not pd.dataframe.DataFrame:
        data = pd.utils.from_pandas(data)
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
    if data.empty:
        raise ValueError('No data left after filtering unsuccessful estimations')
    if build_pdf:
        pdf_folder = os.path.join(store_path, 'pdf')
        try:
            os.mkdir(pdf_folder)
        except FileExistsError:
            print('Folder', pdf_folder, 'for storing pdf files already exists')
    tex_folder = os.path.join(store_path, 'tex')
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
    df = data[['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
               't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
               'USAC_parameters_estimator', 'USAC_parameters_refinealg', 'th']]
    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
             ('t_diff_ty', ''), ('t_diff_tz', '')]
    #Group by USAC parameters 5&6 and calculate the statistic
    stats = df.groupby(['USAC_parameters_estimator', 'USAC_parameters_refinealg', 'th']).describe()
    errvalnames = stats.columns.values # Includes statistic name and error value names
    grp_names = stats.index.names #As used when generating the groups
    rel_data_path = os.path.relpath(tdata_folder, tex_folder)
    base_out_name = 'data_USAC_opts_' + grp_names[0] + '-' + grp_names[1] + '_combs_vs_' + grp_names[2]
    # holds the grouped names/entries within the group names excluding the last entry th
    #grp_values = list(dict.fromkeys([i[0:2] for i in stats.index.values]))
    tex_infos = {'title': 'Statistics for USAC Option Combinations of ' +
                          tex_string_coding_style(grp_names[0]) + ' and ' +
                          tex_string_coding_style(grp_names[1]) + ' compared to ' +
                          tex_string_coding_style(grp_names[2]) + ' Values',
                 'sections': [],
                 'make_index': make_fig_index,
                 'ctrl_fig_size': True}
    pdf_nr = 0
    pdfsplitentry = ['t_distDiff']
    for it in errvalnames:
        if it[-1] != 'count':
            tmp = stats[it[0]].unstack()
            tmp = tmp[it[1]]
            tmp = tmp.T
            tmp.columns = ['%s%s' % (str(a), '-%s' % str(b) if b is not None else '') for a, b in tmp.columns]
            tmp.columns.name = '-'.join(grp_names[0:2])
            dataf_name = 'data_' + '_'.join(map(str, it)) + '_vs_' + \
                       str(grp_names[-1]) + '.csv'
            dataf_name = dataf_name.replace('%', 'perc')
            fdataf_name = os.path.join(tdata_folder, dataf_name)
            with open(fdataf_name, 'a') as f:
                f.write('# ' + str(it[-1]) + ' values for ' + str(it[0]) + '\n')
                f.write('# Column parameters: ' + '-'.join(grp_names[0:2]) + '\n')
                tmp.to_csv(index=True, sep=';', path_or_buf=f, header=True)

            #Construct tex-file
            if pdf_nr < len(pdfsplitentry):
                if pdfsplitentry[pdf_nr] == str(it[0]):
                    pdf_nr += 1
            stats_all = tmp.stack().reset_index()
            stats_all = stats_all.drop(stats_all.columns[0:2], axis=1).describe().T
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
                                                  tex_string_coding_style(str(it[0])) +
                                                  ' compared to ' + tex_string_coding_style(str(grp_names[-1])),
                                          'fig_type': 'smooth',
                                          'plots': list(tmp.columns.values),
                                          'axis_y': replace_stat_names(it[-1]) + findUnit(str(it[0]), units),
                                          'plot_x': str(grp_names[-1]),
                                          'limits': use_limits,
                                          'legend': [tex_string_coding_style(a) for a in list(tmp.columns.values)],
                                          'legend_cols': None,
                                          'use_marks': True,
                                          'pdf_nr': pdf_nr
                                          })
            nr_plots = len(tex_infos['sections'][-1]['plots'])
            max_cols = int(235 / (len(max(tex_infos['sections'][-1]['plots'], key=len)) * 3 + 16))
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
            tex_infos['sections'][-1]['legend_cols'] = use_cols

    template = ji_env.get_template('usac-testing_single_plots.tex')
    #Get number of pdfs to generate
    pdf_nr = tex_infos['sections'][-1]['pdf_nr']
    if pdf_nr == 0:
        rendered_tex = template.render(title=tex_infos['title'],
                                       make_index=tex_infos['make_index'],
                                       ctrl_fig_size=tex_infos['ctrl_fig_size'],
                                       sections=tex_infos['sections'])
        texf_name = base_out_name + '.tex'
        pdf_name = base_out_name + '.pdf'
        if build_pdf:
            res = compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index, os.path.join(pdf_folder, pdf_name))
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
                                           sections=it)
            texf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.tex'
            if build_pdf:
                pdf_name = base_out_name + '_' + str(int(it[0]['pdf_nr'])) + '.pdf'
                res = compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index,
                                  os.path.join(pdf_folder, pdf_name))
            else:
                res = compile_tex(rendered_tex, tex_folder, texf_name, make_fig_index)
    return res


def replace_stat_names(name):
    if name == 'max':
        return 'Maximum'
    elif name == 'min':
        return 'Minimum'
    elif name == 'mean':
        return name.capitalize()
    elif name == 'std':
        return 'Standard deviation'
    elif name == r'25%':
        return r'25\% percentile'
    elif name == '50%':
        return 'Median'
    elif name == '75%':
        return r'75\% percentile'
    else:
        return str(name).replace('%', '\%').capitalize()


def tex_string_coding_style(text):
    text = text.replace('_', '\\_')
    return '\\texttt{' + text + '}'

def findUnit(key, units):
    for i in units:
        if key in i:
            return i[1]
    return ''

#Only for testing
def main():
    num_pts = int(500)
    data = {'R_diffAll': [0.3, 0.5, 0.7, 0.4, 0.6] * int(num_pts/5),
            'R_diff_roll_deg': 1000 + np.abs(np.random.randn(num_pts) * 10),
            'R_diff_pitch_deg': 10 + np.random.randn(num_pts) * 5,
            'R_diff_yaw_deg': -1000 + np.abs(np.random.randn(num_pts)),
            't_angDiff_deg': [0.3, 0.5, 0.7, 0.4, 0.6] * int(num_pts/5),
            't_distDiff': np.abs(np.random.randn(num_pts) * 100),
            't_diff_tx': -10000 + np.random.randn(num_pts) * 100,
            't_diff_ty': 20000 + np.random.randn(num_pts),
            't_diff_tz': -450 + np.random.randn(num_pts),
            'USAC_parameters_estimator': np.random.randint(0, 3, num_pts),
            'USAC_parameters_refinealg': np.random.randint(0, 7, num_pts),
            'th': np.tile(np.arange(0.4, 0.9, 0.1), int(num_pts/5)),
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
    calcSatisticRt_th(data, '/home/maierj/work/Sequence_Test/py_test')


if __name__ == "__main__":
    main()