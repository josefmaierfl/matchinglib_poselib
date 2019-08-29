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


def eval_test(load_path, output_path, test_name, test_nr):
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
    if test_name == 'testing_tests':#'usac-testing':
        if not test_nr:
            raise ValueError('test_nr is required for usac-testing')
        from statistics_and_plot import calcSatisticAndPlot_2D, \
            calcSatisticAndPlot_3D, \
            calcSatisticAndPlot_2D_partitions, \
            calcFromFuncAndPlot_3D, \
            calcFromFuncAndPlot_2D_partitions
        if test_nr == 1:
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
            return calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                          store_path=output_path,
                                          tex_file_pre_str='plots_USAC_opts_',
                                          fig_title_pre_str=fig_title_pre_str,
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
                                          fig_type='smooth',
                                          use_marks=True,
                                          ctrl_fig_size=True,
                                          make_fig_index=True,
                                          build_pdf=True,
                                          figs_externalize=True)
        elif test_nr == 2:
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
            return calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                          store_path=output_path,
                                          tex_file_pre_str='plots_USAC_opts_',
                                          fig_title_pre_str=fig_title_pre_str,
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
                                          fig_type='smooth',
                                          use_marks=True,
                                          ctrl_fig_size=True,
                                          make_fig_index=True,
                                          build_pdf=True,
                                          figs_externalize=True)
        elif test_nr == 3:
            fig_title_pre_str = 'Values of R\\&t Differences for USAC Option Combinations of '
            eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                            't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
            units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                     ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                     ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                     ('t_diff_ty', ''), ('t_diff_tz', '')]
            it_parameters = ['USAC_parameters_estimator',
                             'USAC_parameters_refinealg']
            special_calcs_args = {'build_pdf': (False, True),
                                  'use_marks': True,
                                  'fig_type': 'surface',
                                  'res_par_name': 'USAC_opt_refine_ops_inlrat_th'}
            from usac_eval import get_best_comb_and_th_for_inlrat_1
            return calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                          store_path=output_path,
                                          tex_file_pre_str='plots_USAC_opts_',
                                          fig_title_pre_str=fig_title_pre_str,
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
                                          build_pdf=False,
                                          figs_externalize=True)
        elif test_nr == 4:
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
            return calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                     store_path=output_path,
                                                     tex_file_pre_str='plots_USAC_opts_',
                                                     fig_title_pre_str=fig_title_pre_str,
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
                                                     fig_type='smooth',
                                                     use_marks=True,
                                                     ctrl_fig_size=True,
                                                     make_fig_index=True,
                                                     build_pdf=False,
                                                     figs_externalize=True)
        elif test_nr == 5:
            fig_title_pre_str = 'Temporal Behaviour for USAC Option Combinations of '
            eval_columns = ['robEstimationAndRef_us']
            units = []
            it_parameters = ['USAC_parameters_estimator',
                             'USAC_parameters_refinealg']
            special_calcs_args = {'build_pdf': (True, True),
                                  'use_marks': True,
                                  'fig_type': 'smooth',
                                  'nr_target_kps': 1000,
                                  't_data_separators': ['inlratMin']}
            from usac_eval import filter_nr_kps, calc_Time_Model, estimate_alg_time_fixed_kp
            return calcFromFuncAndPlot_3D(data=data.copy(deep=True),
                                          store_path=output_path,
                                          tex_file_pre_str='plots_USAC_opts_',
                                          fig_title_pre_str=fig_title_pre_str,
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
        elif test_nr == 6:
            fig_title_pre_str = 'Temporal Behaviour for USAC Option Combinations of '
            eval_columns = ['robEstimationAndRef_us']
            units = []
            it_parameters = ['USAC_parameters_estimator',
                             'USAC_parameters_refinealg']
            from usac_eval import filter_nr_kps, calc_Time_Model
            return calcFromFuncAndPlot_2D_partitions(data=data.copy(deep=True),
                                                     store_path=output_path,
                                                     tex_file_pre_str='plots_USAC_opts_',
                                                     fig_title_pre_str=fig_title_pre_str,
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
                                                     calc_func_args={'data_separators': ['inlRatMin', 'th']},
                                                     fig_type='smooth',
                                                     use_marks=True,
                                                     ctrl_fig_size=True,
                                                     make_fig_index=True,
                                                     build_pdf=True,
                                                     figs_externalize=False)
        elif test_nr == 7:
            fig_title_pre_str = 'Statistics on R\\&t Differences for USAC Option Combinations of '
            eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                            't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
            units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                     ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                     ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                     ('t_diff_ty', ''), ('t_diff_tz', '')]
            it_parameters = ['USAC_parameters_automaticSprtInit',
                             'USAC_parameters_noAutomaticProsacParamters',
                             'USAC_parameters_prevalidateSample',
                             'USAC_parameters_USACInlratFilt']
            special_calcs_args = {'build_pdf': (True, True),
                                  'use_marks': True,
                                  'res_par_name': 'USAC_opt_search_ops_th'}
            from usac_eval import get_best_comb_and_th_1
            return calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                          store_path=output_path,
                                          tex_file_pre_str='plots_USAC_opts_',
                                          fig_title_pre_str=fig_title_pre_str,
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
                                          fig_type='smooth',
                                          use_marks=True,
                                          ctrl_fig_size=True,
                                          make_fig_index=True,
                                          build_pdf=True,
                                          figs_externalize=True)
        elif test_nr == 8:
            fig_title_pre_str = 'Statistics on R\\&t Differences for USAC Option Combinations of '
            eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                            't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
            units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                     ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                     ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                     ('t_diff_ty', ''), ('t_diff_tz', '')]
            it_parameters = ['USAC_parameters_automaticSprtInit',
                             'USAC_parameters_noAutomaticProsacParamters',
                             'USAC_parameters_prevalidateSample',
                             'USAC_parameters_USACInlratFilt']
            special_calcs_args = {'build_pdf': (True, True),
                                  'use_marks': True,
                                  'res_par_name': 'USAC_opt_search_ops_inlrat'}
            from usac_eval import get_best_comb_inlrat_1
            return calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                          store_path=output_path,
                                          tex_file_pre_str='plots_USAC_opts_',
                                          fig_title_pre_str=fig_title_pre_str,
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
                                          fig_type='smooth',
                                          use_marks=True,
                                          ctrl_fig_size=True,
                                          make_fig_index=True,
                                          build_pdf=True,
                                          figs_externalize=True)
        elif test_nr == 9:
            fig_title_pre_str = 'Values of R\\&t Differences for USAC Option Combinations of '
            eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                            't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
            units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                     ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                     ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                     ('t_diff_ty', ''), ('t_diff_tz', '')]
            it_parameters = ['USAC_parameters_automaticSprtInit',
                             'USAC_parameters_noAutomaticProsacParamters',
                             'USAC_parameters_prevalidateSample',
                             'USAC_parameters_USACInlratFilt']
            special_calcs_args = {'build_pdf': (False, True),
                                  'use_marks': True,
                                  'fig_type': 'surface',
                                  'res_par_name': 'USAC_opt_search_ops_kpAccSd_th',
                                  'func_name': 'get_best_comb_and_th_for_kpacc_1'}
            from usac_eval import get_best_comb_and_th_for_inlrat_1
            return calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                          store_path=output_path,
                                          tex_file_pre_str='plots_USAC_opts_',
                                          fig_title_pre_str=fig_title_pre_str,
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
        elif test_nr == 10:
            fig_title_pre_str = 'Values of R\\&t Differences for USAC Option Combinations of '
            eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                            't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
            units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                     ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                     ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                     ('t_diff_ty', ''), ('t_diff_tz', '')]
            it_parameters = ['USAC_parameters_automaticSprtInit',
                             'USAC_parameters_noAutomaticProsacParamters',
                             'USAC_parameters_prevalidateSample',
                             'USAC_parameters_USACInlratFilt']
            special_calcs_args = {'build_pdf': (False, True),
                                  'use_marks': True,
                                  'fig_type': 'surface',
                                  'res_par_name': 'USAC_opt_search_ops_inlrat_th'}
            from usac_eval import get_best_comb_and_th_for_inlrat_1
            return calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                          store_path=output_path,
                                          tex_file_pre_str='plots_USAC_opts_',
                                          fig_title_pre_str=fig_title_pre_str,
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
                                          build_pdf=False,
                                          figs_externalize=True)
        elif test_nr == 36:
            fig_title_pre_str = 'Values of Inlier Ratio Differences for USAC Option Combinations of '
            eval_columns = ['inlRat_estimated', 'inlRat_GT']
            units = [('inlRat_diff', '')]
            it_parameters = ['USAC_parameters_automaticSprtInit',
                             'USAC_parameters_noAutomaticProsacParamters',
                             'USAC_parameters_prevalidateSample',
                             'USAC_parameters_USACInlratFilt']
            # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
            partitions = ['depthDistr', 'kpAccSd']
            special_calcs_args = {'build_pdf': (True, True), 'use_marks': True}
            from usac_eval import get_inlrat_diff, get_min_inlrat_diff
            return calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                     store_path=output_path,
                                                     tex_file_pre_str='plots_USAC_opts_',
                                                     fig_title_pre_str=fig_title_pre_str,
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
                                                     fig_type='smooth',
                                                     use_marks=True,
                                                     ctrl_fig_size=True,
                                                     make_fig_index=True,
                                                     build_pdf=True,
                                                     figs_externalize=True)
        elif test_nr == 11:
            fig_title_pre_str = 'Values of R\\&t Differences for USAC Option Combinations of '
            eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                            't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
            units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                     ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                     ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                     ('t_diff_ty', ''), ('t_diff_tz', '')]
            it_parameters = ['USAC_parameters_automaticSprtInit',
                             'USAC_parameters_noAutomaticProsacParamters',
                             'USAC_parameters_prevalidateSample',
                             'USAC_parameters_USACInlratFilt']
            # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
            partitions = ['depthDistr', 'kpDistr', 'th']  # th must be at the end
            special_calcs_args = {'build_pdf': (True, True),
                                  'use_marks': True}
            from usac_eval import get_best_comb_th_scenes_1
            return calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                     store_path=output_path,
                                                     tex_file_pre_str='plots_USAC_opts_',
                                                     fig_title_pre_str=fig_title_pre_str,
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
                                                     fig_type='smooth',
                                                     use_marks=True,
                                                     ctrl_fig_size=True,
                                                     make_fig_index=True,
                                                     build_pdf=False,
                                                     figs_externalize=True)
        elif test_nr == 12:
            fig_title_pre_str = 'Temporal Behaviour for USAC Option Combinations of '
            eval_columns = ['robEstimationAndRef_us']
            units = []
            it_parameters = ['USAC_parameters_automaticSprtInit',
                             'USAC_parameters_noAutomaticProsacParamters',
                             'USAC_parameters_prevalidateSample',
                             'USAC_parameters_USACInlratFilt']
            special_calcs_args = {'build_pdf': (True, True),
                                  'use_marks': True,
                                  'fig_type': 'smooth',
                                  'nr_target_kps': 1000,
                                  't_data_separators': ['inlratMin']}
            from usac_eval import filter_nr_kps, calc_Time_Model, estimate_alg_time_fixed_kp
            return calcFromFuncAndPlot_3D(data=data.copy(deep=True),
                                          store_path=output_path,
                                          tex_file_pre_str='plots_USAC_opts_',
                                          fig_title_pre_str=fig_title_pre_str,
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
        elif test_nr == 13:
            fig_title_pre_str = 'Temporal Behaviour for USAC Option Combinations of '
            eval_columns = ['robEstimationAndRef_us']
            units = []
            it_parameters = ['USAC_parameters_automaticSprtInit',
                             'USAC_parameters_noAutomaticProsacParamters',
                             'USAC_parameters_prevalidateSample',
                             'USAC_parameters_USACInlratFilt']
            from usac_eval import filter_nr_kps, calc_Time_Model, estimate_alg_time_fixed_kp_for_props
            special_calcs_args = {'build_pdf': (True, True),
                                  'use_marks': True,
                                  'fig_type': 'smooth',
                                  'nr_target_kps': 1000,
                                  't_data_separators': ['inlratMin', 'th'],
                                  'res_par_name': 'USAC_opt_search_min_time_inlrat_th'}
            return calcFromFuncAndPlot_2D_partitions(data=data.copy(deep=True),
                                                     store_path=output_path,
                                                     tex_file_pre_str='plots_USAC_opts_',
                                                     fig_title_pre_str=fig_title_pre_str,
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
                                                     fig_type='smooth',
                                                     use_marks=True,
                                                     ctrl_fig_size=True,
                                                     make_fig_index=True,
                                                     build_pdf=True,
                                                     figs_externalize=False)


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
                        help='Test number within the main test specified by test_name starting with 1')
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

    return eval_test(load_path, output_path, test_name, args.test_nr)


if __name__ == "__main__":
    main()