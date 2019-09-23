"""
Loads test results and calls specific functions for evaluation as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np
import ruamel.yaml as yaml
import modin.pandas as mpd
import pandas as pd

#warnings.simplefilter('ignore', category=UserWarning)

def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)
yaml.SafeLoader.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)

warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

def readOpenCVYaml(file, isstr = False):
    if isstr:
        data = file.split('\n')
    else:
        with open(file, 'r') as fi:
            data = fi.readlines()
    data = [line for line in data if line and line[0] is not '%']
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


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def eval_test(load_path, output_path, test_name, test_nr, eval_nr, comp_path, comp_pars):
    #Load test results
    res_path = os.path.join(load_path, 'results')
    if not os.path.exists(res_path):
        raise ValueError('No results folder found')
    # Get all folders that contain data
    sub_dirs = [name for name in os.listdir(res_path) if os.path.isdir(os.path.join(res_path, name))]
    sub_dirs = [name for name in sub_dirs if RepresentsInt(name)]
    if len(sub_dirs) == 0:
        raise ValueError('No subdirectories holding data found')
    data_list = []
    for sf in sub_dirs:
        res_path_it = os.path.join(res_path, sf)
        ov_file = os.path.join(res_path_it, 'allRunsOverview.yaml')
        if not os.path.exists(ov_file):
            raise ValueError('Results overview file allRunsOverview.yaml not found')
        par_data = readOpenCVYaml(ov_file)
        parSetNr = 0
        while True:
            try:
                data_set = par_data['parSetNr' + str(int(parSetNr))]
                parSetNr += 1
            except:
                break
            csvf = os.path.join(res_path_it, data_set['hashTestingPars'])
            if not os.path.exists(csvf):
                raise ValueError('Results file ' + csvf + ' not found')
            csv_data = pd.read_csv(csvf, delimiter=';', engine='c')
            #print('Loaded', csvf, 'with shape', csv_data.shape)
            #csv_data.set_index('Nr')
            addSequInfo_sep = None
            # for idx, row in csv_data.iterrows():
            for row in csv_data.itertuples():
                #tmp = row['addSequInfo'].split('_')
                tmp = row.addSequInfo.split('_')
                tmp = dict([(tmp[x], tmp[x + 1]) for x in range(0,len(tmp), 2)])
                if addSequInfo_sep:
                    for k in addSequInfo_sep.keys():
                        addSequInfo_sep[k].append(tmp[k])
                else:
                    for k in tmp.keys():
                        tmp[k] = [tmp[k]]
                    addSequInfo_sep = tmp
            addSequInfo_df = pd.DataFrame(data=addSequInfo_sep)
            csv_data = pd.concat([csv_data, addSequInfo_df], axis=1, sort=False, join_axes=[csv_data.index])
            csv_data.drop(columns=['addSequInfo'], inplace=True)
            data_set_tmp = merge_dicts(data_set)
            data_set_tmp = pd.DataFrame(data=data_set_tmp, index=[0])
            data_set_repl = pd.DataFrame(np.repeat(data_set_tmp.values, csv_data.shape[0], axis=0))
            data_set_repl.columns = data_set_tmp.columns
            csv_new = pd.concat([csv_data, data_set_repl], axis=1, sort=False, join_axes=[csv_data.index])
            data_list.append(csv_new)
    data = pd.concat(data_list, ignore_index=True, sort=False, copy=False)
    # data_dict = data.to_dict('list')
    # data = mpd.DataFrame(data_dict)
    data = mpd.utils.from_pandas(data)
    #print('Finished loading data')
    ret = 0
    if test_name == 'testing_tests':#'usac-testing':
        if not test_nr:
            raise ValueError('test_nr is required for usac-testing')
        from statistics_and_plot import calcSatisticAndPlot_2D, \
            calcSatisticAndPlot_3D, \
            calcSatisticAndPlot_2D_partitions, \
            calcFromFuncAndPlot_3D, \
            calcFromFuncAndPlot_2D_partitions, \
            calcFromFuncAndPlot_3D_partitions
        if test_nr == 1:
            if eval_nr[0] < 0:
                evals = list(range(1, 7))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 1:
                    fig_title_pre_str = 'Statistics on R\\&t Differences for USAC Option Combinations of '
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
                    fig_title_pre_str = 'Statistics on R\\&t Differences for USAC Option Combinations of '
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
                    fig_title_pre_str = 'Values of R\\&t Differences for USAC Option Combinations of '
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
                    fig_title_pre_str = 'Values of R\\&t Differences for USAC Option Combinations of '
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
                                                             # Column names for which statistics are calculated (y-axis)
                                                             eval_columns=eval_columns,
                                                             # Units in string format for every entry of eval_columns
                                                             units=units,
                                                             # Algorithm parameters to evaluate
                                                             it_parameters=it_parameters,
                                                             # Data properties to calculate results separately
                                                             partitions=['th'],
                                                             # x-axis column name
                                                             x_axis_column=['nrCorrs_GT'],
                                                             filter_func=filter_nr_kps,
                                                             filter_func_args=None,
                                                             special_calcs_func=None,
                                                             special_calcs_args=None,
                                                             calc_func=calc_Time_Model,
                                                             calc_func_args={'data_separators': ['inlRatMin', 'th']},
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
                    fig_title_pre_str = 'Statistics on R\\&t Differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['USAC_parameters_automaticSprtInit',
                                     'USAC_parameters_automaticProsacParameters',
                                     'USAC_parameters_prevalidateSample',
                                     'USAC_parameters_USACInlratFilt']
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
                    fig_title_pre_str = 'Statistics on R\\&t Differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['USAC_parameters_automaticSprtInit',
                                     'USAC_parameters_automaticProsacParameters',
                                     'USAC_parameters_prevalidateSample',
                                     'USAC_parameters_USACInlratFilt']
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
                    fig_title_pre_str = 'Values of R\\&t Differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['USAC_parameters_automaticSprtInit',
                                     'USAC_parameters_automaticProsacParameters',
                                     'USAC_parameters_prevalidateSample',
                                     'USAC_parameters_USACInlratFilt']
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
                    fig_title_pre_str = 'Values of R\\&t Differences for USAC Option Combinations of '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['USAC_parameters_automaticSprtInit',
                                     'USAC_parameters_automaticProsacParameters',
                                     'USAC_parameters_prevalidateSample',
                                     'USAC_parameters_USACInlratFilt']
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
                    it_parameters = ['USAC_parameters_automaticSprtInit',
                                     'USAC_parameters_automaticProsacParameters',
                                     'USAC_parameters_prevalidateSample',
                                     'USAC_parameters_USACInlratFilt']
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
                    it_parameters = ['USAC_parameters_automaticSprtInit',
                                     'USAC_parameters_automaticProsacParameters',
                                     'USAC_parameters_prevalidateSample',
                                     'USAC_parameters_USACInlratFilt']
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
                    it_parameters = ['USAC_parameters_automaticSprtInit',
                                     'USAC_parameters_automaticProsacParameters',
                                     'USAC_parameters_prevalidateSample',
                                     'USAC_parameters_USACInlratFilt']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'smooth',
                                          'nr_target_kps': 1000,
                                          't_data_separators': ['inlratMin']}
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
                                                  calc_func_args={'data_separators': ['inlRatMin', 'th']},
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
                    it_parameters = ['USAC_parameters_automaticSprtInit',
                                     'USAC_parameters_automaticProsacParameters',
                                     'USAC_parameters_prevalidateSample',
                                     'USAC_parameters_USACInlratFilt']
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
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=['th'],
                                                             x_axis_column=['nrCorrs_GT'],
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
                    it_parameters = ['USAC_parameters_automaticSprtInit',
                                     'USAC_parameters_automaticProsacParameters',
                                     'USAC_parameters_prevalidateSample',
                                     'USAC_parameters_USACInlratFilt']
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
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=['th'],
                                                             xy_axis_columns=['nrCorrs_GT'],
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
        from statistics_and_plot import calcSatisticAndPlot_2D, \
            calcSatisticAndPlot_3D, \
            calcSatisticAndPlot_2D_partitions, \
            calcFromFuncAndPlot_3D, \
            calcFromFuncAndPlot_2D_partitions
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
                it_parameters = ['RobMethod']
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
                it_parameters = ['RobMethod']
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
                it_parameters = ['RobMethod']
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
                it_parameters = ['RobMethod']
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
                it_parameters = ['RobMethod']
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
                it_parameters = ['RobMethod']
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
                it_parameters = ['RobMethod']
                from usac_eval import filter_nr_kps, calc_Time_Model
                ret += calcFromFuncAndPlot_2D_partitions(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_USAC_vs_RANSAC_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='time',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         partitions=['th'],
                                                         x_axis_column=['nrCorrs_GT'],
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
        from statistics_and_plot import calcSatisticAndPlot_2D, \
            calcSatisticAndPlot_2D_partitions, \
            calcFromFuncAndPlot_aggregate
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
                    it_parameters = ['refineMethod_algorithm',
                                     'refineMethod_costFunction',
                                     'BART']
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
                    it_parameters = ['refineMethod_algorithm',
                                     'refineMethod_costFunction',
                                     'BART']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refinement_ba_best_comb_scenes'}
                    from refinement_eval import get_best_comb_scenes_1
                    from usac_eval import filter_nr_kps_stat
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
                                                             filter_func=filter_nr_kps_stat,
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
                    it_parameters = ['refineMethod_algorithm',
                                     'refineMethod_costFunction',
                                     'BART']
                    special_calcs_args = {'build_pdf': (True, True),
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
                    it_parameters = ['refineMethod_algorithm',
                                     'refineMethod_costFunction']
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
                    fig_title_pre_str = 'Statistics on R\\&t Differences After Bundle Adjustment (BA) Including ' \
                                        'Intrinsics and Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['refineMethod_algorithm',
                                     'refineMethod_costFunction']
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
                    it_parameters = ['refineMethod_algorithm',
                                     'refineMethod_costFunction']
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
                    it_parameters = ['refineMethod_algorithm',
                                     'refineMethod_costFunction']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    from refinement_eval import combineK
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'error_function': combineK,
                                          'error_type_text': 'Combined Camera Matrix Errors $e_{\\mli{K1,2}}$',
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
        from statistics_and_plot import calcSatisticAndPlot_2D, \
            calcSatisticAndPlot_2D_partitions, \
            calcFromFuncAndPlot_aggregate, \
            calcSatisticAndPlot_aggregate
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
                it_parameters = ['matchesFilter_refineGMS',
                                 'matchesFilter_refineVFC',
                                 'matchesFilter_refineSOF']
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
                it_parameters = ['matchesFilter_refineGMS',
                                 'matchesFilter_refineVFC',
                                 'matchesFilter_refineSOF']
                from usac_eval import get_inlrat_diff
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
                it_parameters = ['matchesFilter_refineGMS',
                                 'matchesFilter_refineVFC',
                                 'matchesFilter_refineSOF']
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
                it_parameters = ['matchesFilter_refineGMS',
                                 'matchesFilter_refineVFC',
                                 'matchesFilter_refineSOF']
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
                it_parameters = ['matchesFilter_refineGMS',
                                 'matchesFilter_refineVFC',
                                 'matchesFilter_refineSOF']
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
                fig_title_pre_str = 'Statistics on R\\&t Differences for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                it_parameters = ['matchesFilter_refineGMS',
                                 'matchesFilter_refineVFC',
                                 'matchesFilter_refineSOF']
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
                it_parameters = ['matchesFilter_refineGMS',
                                 'matchesFilter_refineVFC',
                                 'matchesFilter_refineSOF']
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
        from statistics_and_plot import calcSatisticAndPlot_2D, \
            calcSatisticAndPlot_2D_partitions, \
            calcSatisticAndPlot_aggregate
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
                    it_parameters = ['stereoParameters_refineMethod_algorithm',
                                     'stereoParameters_refineMethod_costFunction',
                                     'stereoParameters_BART']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refRT_stereo_BA_opts_inlrat'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                    it_parameters = ['stereoParameters_refineMethod_algorithm',
                                     'stereoParameters_refineMethod_costFunction',
                                     'stereoParameters_BART']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'ref_stereo_ba_best_comb_scenes'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                    it_parameters = ['stereoParameters_matchesFilter_refineGMS',
                                     'stereoParameters_matchesFilter_refineVFC',
                                     'stereoParameters_matchesFilter_refineSOF']
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
                    it_parameters = ['stereoParameters_refineMethod_algorithm',
                                     'stereoParameters_refineMethod_costFunction']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refRT_stereo_opts_for_BA2_inlrat'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 2, 'RT-stats', descr)
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
                    fig_title_pre_str = 'Statistics on R\\&t Differences After Bundle Adjustment (BA) Including ' \
                                        'Intrinsics and Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['stereoParameters_refineMethod_algorithm',
                                     'stereoParameters_refineMethod_costFunction']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'ref_stereo_best_comb_for_BA2_scenes'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 2, 'RT-stats', descr)
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
                    it_parameters = ['stereoParameters_refineMethod_algorithm',
                                     'stereoParameters_refineMethod_costFunction']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'refRT_stereo_opts_for_BA2_K_inlrat'}
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 2, 'K-stats', descr)
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
                    it_parameters = ['stereoParameters_refineMethod_algorithm',
                                     'stereoParameters_refineMethod_costFunction']
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
                    compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 2, 'K-stats', descr)
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

    return ret


def get_compare_info(comp_pars, comp_path, test_name, test_r, eval_description_path, descr):
    if not comp_pars:
        raise ValueError('Parameter values for comparing refinement without kp aggregation missing')
    c_path = os.path.join(comp_path, test_name)
    if not os.path.exists(c_path):
        raise ValueError('Specific test compare directory ' + c_path + '  not found')
    c_path = os.path.join(c_path, str(test_r))
    if not os.path.exists(c_path):
        raise ValueError('Specific test nr compare directory ' + c_path + '  not found')
    compare_source = {'store_path': c_path,
                      'it_par_select': [a.split('-')[-1] for a in comp_pars],
                      'it_parameters': [a.split('-')[0] for a in comp_pars],
                      'eval_description_path': eval_description_path,
                      'cmp': descr
                      }
    return compare_source


def merge_dicts(in_dict, mainkey = None):
    tmp = {}
    for i in in_dict.keys():
        if isinstance(in_dict[i], dict):
            tmp1 = merge_dicts(in_dict[i], i)
            for j in tmp1.keys():
                if mainkey is not None:
                    tmp[mainkey + '_' + j] = tmp1[j]
                else:
                    tmp.update({j: tmp1[j]})
        else:
            if mainkey is not None:
                tmp.update({mainkey + '_' + i: in_dict[i]})
            else:
                tmp.update({i: in_dict[i]})
    return tmp


def main():
    parser = argparse.ArgumentParser(description='Loads test results and calls specific functions for evaluation '
                                                 'as specified in file Autocalibration-Parametersweep-Testing.xlsx')
    parser.add_argument('--path', type=str, required=True,
                        help='Main directory holding test results. This directory must hold subdirectories with the '
                             'different test names')
    parser.add_argument('--output_path', type=str, required=False,
                        help='Optional output directory. By default, (if this option is not provided, the data is '
                             'stored in a new directory inside the directory for a specific test.')
    parser.add_argument('--test_name', type=str, required=True,
                        help='Name of the main test like \'USAC-testing\' or \'USAC_vs_RANSAC\'')
    parser.add_argument('--test_nr', type=int, required=False,
                        help='Main test number within the main test specified by test_name starting with 1. '
                             'test_name=\'USAC-testing\' e.g. offers 2 tests with a lot of sub-tests for each of them')
    parser.add_argument('--eval_nr', type=int, nargs='*', default=[-1],
                        help='Evaluation number within test_nr starting with 1. test_name=\'USAC-testing\' '
                             'with test_nr=1 offers e.g. 15 different evaluations. The description and number for '
                             'each evaluation can be found in Autocalibration-Parametersweep-Testing.xlsx. '
                             'If a value of -1 [default] is provided, all evaluations are performed for a given '
                             'test_name and test_nr. Also, multiple numbers can be specified.')
    parser.add_argument('--compare_pars', type=str, required=False, nargs='*', default=[],
                        help='If provided, results from already performed evaluations can be loaded and added to '
                             'the current evaluation for comparison. The type of evaluation must be similar '
                             '(same axis, data partitions, ...). The argument must be provided as a list of strings '
                             '(e.g. --compare str1 str2 str3 ...). The list must contain the parameter-parameter '
                             'value pairs of the already performed evaluation for which the comparison should be performed '
                             '(like \'--compare_pars refineMethod_algorithm-PR_STEWENIUS '
                             'refineMethod_costFunction-PR_PSEUDOHUBER_WEIGHTS BART-extr_only\' for the evaluation '
                             'of refinement methods and bundle adjustment options).')
    parser.add_argument('--compare_path', type=str, required=False,
                        help='If provided, a different path is used for loading results for comparison. Otherwise, '
                             'the path from option --path is used. Results are only loaded, if option '
                             '--compare_pars is provided.')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise ValueError('Main directory not found')
    test_name = args.test_name.lower()
    load_path = os.path.join(args.path, args.test_name)
    if not os.path.exists(load_path):
        raise ValueError('Specific main test directory ' + load_path + '  not found')
    if args.test_nr:
        load_path = os.path.join(load_path, str(args.test_nr))
        if not os.path.exists(load_path):
            raise ValueError('Specific test directory ' + load_path + '  not found')
    if args.output_path:
        if not os.path.exists(args.output_path):
            raise ValueError('Specified output directory not found')
        output_path =  args.output_path
    else:
        output_path = os.path.join(load_path, 'evals')
        try:
            os.mkdir(output_path)
        except FileExistsError:
            raise ValueError('Directory ' + output_path + ' already exists')
    comp_path = None
    comp_pars = None
    if args.compare_pars:
        comp_pars = args.compare_pars
        if args.compare_path:
            comp_path = args.compare_path
            if not os.path.exists(comp_path):
                raise ValueError('Specific main compare directory ' + comp_path + '  not found')
        else:
            comp_path = args.path

    return eval_test(load_path, output_path, test_name, args.test_nr, args.eval_nr, comp_path, comp_pars)


if __name__ == "__main__":
    main()