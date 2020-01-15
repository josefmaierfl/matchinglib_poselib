"""
Loads test results and calls specific functions for evaluation as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np, math
import multiprocessing
import ruamel.yaml as yaml
# import modin.pandas as mpd
import pandas as pd
from timeit import default_timer as timer
import contextlib, logging

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


def load_test_res(load_path):
    # Load test results
    start = timer()
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
            # print('Loaded', csvf, 'with shape', csv_data.shape)
            # csv_data.set_index('Nr')
            addSequInfo_sep = None
            # for idx, row in csv_data.iterrows():
            for row in csv_data.itertuples():
                # tmp = row['addSequInfo'].split('_')
                tmp = row.addSequInfo.split('_')
                tmp = dict([(tmp[x], tmp[x + 1]) for x in range(0, len(tmp), 2)])
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
    end = timer()
    load_time = end - start
    return data, load_time


def eval_test(load_path, output_path, test_name, test_nr, eval_nr, comp_path, comp_pars, cpu_use, message_path):
    data, load_time = load_test_res(load_path)
    # data = None
    # data_dict = data.to_dict('list')
    # data = mpd.DataFrame(data_dict)
    # data = mpd.utils.from_pandas(data)
    #print('Finished loading data')

    main_test_names = ['usac-testing', 'usac_vs_ransac', 'refinement_ba', 'vfc_gms_sof',
                       'refinement_ba_stereo', 'correspondence_pool', 'robustness', 'usac_vs_autocalib']
    sub_test_numbers = [2, 0, 2, 0, 2, 3, 6, 0]
    sub_sub_test_nr = [[list(range(1, 7)), list(range(7, 15)) + [36]],
                       [list(range(1, 8))],
                       [list(range(1, 6)), list(range(1, 5))],
                       [list(range(1, 8))],
                       [list(range(1, 4)), list(range(1, 5))],
                       [list(range(1, 11)), list(range(11, 14)), list(range(14, 16))],
                       [list(range(1, 6)), list(range(6, 11)), list(range(11, 15)), list(range(15, 25)),
                        list(range(25, 29)), list(range(29, 38))],
                       [list(range(1, 9))]]
    evals_w_compare = [('refinement_ba_stereo', 1, 1),
                       ('refinement_ba_stereo', 1, 2),
                       ('refinement_ba_stereo', 2, 1),
                       ('refinement_ba_stereo', 2, 2),
                       ('refinement_ba_stereo', 2, 3),
                       ('refinement_ba_stereo', 2, 4),
                       ('correspondence_pool', 3, 14),
                       ('correspondence_pool', 3, 15)]
    test_idx = main_test_names.index(test_name)
    tn_idx = 0
    if test_nr and test_nr <= sub_test_numbers[test_idx]:
        tn_idx = test_nr - 1
    evcn = list(dict.fromkeys([a[0] for a in evals_w_compare]))
    if eval_nr[0] == -1:
        used_evals = sub_sub_test_nr[test_idx][tn_idx]
    else:
        used_evals = eval_nr
    comp_pars_list = []
    if test_name in evcn:
        for i in evals_w_compare:
            if i[0] == test_name and i[1] == test_nr:
                if i[2] in used_evals and not comp_pars:
                    raise ValueError('Figure name for comparison must be provided')
                elif i[2] in used_evals and str(i[2]) not in comp_pars.keys():
                    raise ValueError('Wrong evaluation number for comparison provided')

        for i in used_evals:
            if comp_pars and str(i) in comp_pars.keys():
                comp_pars_list.append(comp_pars[str(i)])
            else:
                comp_pars_list.append(None)
    else:
        comp_pars_list = [None] * len(used_evals)

    if cpu_use == 1 and not comp_pars:
        return eval_test_exec(data, output_path, test_name, test_nr, eval_nr, comp_path, comp_pars)
    elif cpu_use == 1:
        evs_no_cmp = [a for i, a in enumerate(used_evals) if not comp_pars_list[i]]
        evs_cmp = [[i, a] for i, a in enumerate(used_evals) if comp_pars_list[i]]
        ret = eval_test_exec(data, output_path, test_name, test_nr, evs_no_cmp, None, None)
        for i in evs_cmp:
            ret += eval_test_exec(data, output_path, test_name, test_nr, [i[1]], comp_path, comp_pars_list[i[0]])
        return ret
    else:
        # parts = 1
        # rest = 0
        # if cpu_use > len(used_evals):
        #     cpu_use = len(used_evals)
        # elif cpu_use < len(used_evals):
        #     parts = int(float(len(used_evals)) / float(cpu_use))
        #     rest = len(used_evals) % cpu_use
        # cmds = [[(data, output_path, test_name, test_nr, a, comp_path, b)
        #          for a, b in zip(used_evals[(p * cpu_use):(p * (cpu_use + 1))],
        #                          comp_pars_list[(p * cpu_use):(p * (cpu_use + 1))])] for p in range(0, parts)]
        # if rest:
        #     cmds.append([(data, output_path, test_name, test_nr, a, comp_path, b)
        #                  for a, b in zip(used_evals[(len(used_evals) - rest):],
        #                                  comp_pars_list[(len(used_evals) - rest):])])
        message_path_new = os.path.join(message_path, test_name)
        try:
            os.mkdir(message_path_new)
        except FileExistsError:
            pass
        message_path_new = os.path.join(message_path_new, str(test_nr))
        try:
            os.mkdir(message_path_new)
        except FileExistsError:
            pass
        cmds = [(data, output_path, test_name, test_nr, a, comp_path, b, message_path_new)
                for a, b in zip(used_evals, comp_pars_list)]

        err_trace_base_name = 'evals_except_' + test_name + '_' + str(test_nr)
        base = err_trace_base_name
        excmess = os.path.join(message_path_new, base + '.txt')
        cnt = 1
        while os.path.exists(excmess):
            base = err_trace_base_name + '_-_' + str(cnt)
            excmess = os.path.join(message_path_new, base + '.txt')
            cnt += 1
        logging.basicConfig(filename=excmess, level=logging.DEBUG)
        ret = 0
        cnt_dot = 0
        with multiprocessing.Pool(processes=cpu_use) as pool:
            results = [pool.apply_async(eval_test_exec_std_wrap, t) for t in cmds]
            res1 = []
            for i, r in enumerate(results):
                while 1:
                    sys.stdout.flush()
                    try:
                        res = r.get(2.0)
                        ret += res
                        break
                    except multiprocessing.TimeoutError:
                        if cnt_dot >= 90:
                            print()
                            cnt_dot = 0
                        sys.stdout.write('.')
                        cnt_dot = cnt_dot + 1
                    except Exception:
                        logging.error('Fatal error within evaluation on ' + test_name +
                                      ', Test Nr ' + str(test_nr) + ', Eval Nr ' + str(used_evals[i]), exc_info=True)
                        res1.append(cmds[i][2:5])
                        ret += 1
                        break
        if os.stat(excmess).st_size == 0:
            os.remove(excmess)
        if res1:
            failed_cmds_base_name = 'cmds_evals_failed_' + test_name + '_' + str(test_nr)
            base = failed_cmds_base_name
            fcmdsmess = os.path.join(message_path_new, base + '.txt')
            cnt = 1
            while os.path.exists(fcmdsmess):
                base = failed_cmds_base_name + '_-_' + str(cnt)
                fcmdsmess = os.path.join(message_path_new, base + '.txt')
                cnt += 1
            with open(fcmdsmess, 'w') as fo1:
                fo1.write('Failed evaluations:\n')
                fo1.write('\n'.join(['  '.join(map(str, a)) for a in res1]))
        return ret


def eval_test_exec_std_wrap(data, output_path, test_name, test_nr, eval_nr, comp_path, comp_pars, message_path):
    mess_base_name = 'evals_' + test_name + '_' + str(test_nr) + '_' + str(eval_nr)
    base = mess_base_name
    errmess = os.path.join(message_path, 'stderr_' + base + '.txt')
    stdmess = os.path.join(message_path, 'stdout_' + base + '.txt')
    cnt = 1
    while os.path.exists(errmess) or os.path.exists(stdmess):
        base = mess_base_name + '_-_' + str(cnt)
        errmess = os.path.join(message_path, 'stderr_' + base + '.txt')
        stdmess = os.path.join(message_path, 'stdout_' + base + '.txt')
        cnt += 1
    with open(stdmess, 'a') as f_std, open(errmess, 'a') as f_err:
        with contextlib.redirect_stdout(f_std), contextlib.redirect_stderr(f_err):
            ret = eval_test_exec(data, output_path, test_name, test_nr, eval_nr, comp_path, comp_pars)
            # ret = test_func(data, output_path, test_name, test_nr, eval_nr, comp_path, comp_pars)
    if os.stat(errmess).st_size == 0:
        os.remove(errmess)
    if os.stat(stdmess).st_size == 0:
        os.remove(stdmess)
    return ret


# def test_func(data, output_path, test_name, test_nr, eval_nr, comp_path, comp_pars):
#     print('This is a test print')
#     # print('This is an error message', file=sys.stderr)
#     try:
#         a = 5/0
#     except ZeroDivisionError as e:
#         print('Zero: ', e)
#     # t1()
#     return 1
#
#
# def t1():
#     raise ValueError('Inside function')


def eval_test_exec(data, output_path, test_name, test_nr, eval_nr, comp_path, comp_pars):
    ret = 0
    if test_name == 'usac-testing':
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
            calcFromFuncAndPlot_aggregate, \
            calcSatisticAndPlot_aggregate
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
                elif ev == 4:
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
                    it_parameters = ['refineMethod_algorithm',
                                     'refineMethod_costFunction',
                                     'BART']
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
                    fig_title_pre_str = 'Values of R\\&t Differences After Bundle Adjustment (BA) Including ' \
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
        else:
            raise ValueError('Test nr does not exist')
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
                                              figs_externalize=False,
                                              no_tex=False,
                                              cat_sort=True)
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
                                              figs_externalize=False,
                                              no_tex=False,
                                              cat_sort=True)
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
                fig_title_pre_str = 'Values of R\\&t Differences for Comparison of '
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
            raise ValueError('test_nr is required refinement_ba_stereo')
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
                    it_parameters = ['stereoParameters_refineMethod_CorrPool_algorithm',
                                     'stereoParameters_refineMethod_CorrPool_costFunction',
                                     'stereoParameters_BART_CorrPool']
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
                    it_parameters = ['stereoParameters_refineMethod_CorrPool_algorithm',
                                     'stereoParameters_refineMethod_CorrPool_costFunction',
                                     'stereoParameters_BART_CorrPool']
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
                    it_parameters = ['stereoParameters_refineMethod_CorrPool_algorithm',
                                     'stereoParameters_refineMethod_CorrPool_costFunction',
                                     'stereoParameters_BART_CorrPool']
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
                    it_parameters = ['stereoParameters_refineMethod_CorrPool_algorithm',
                                     'stereoParameters_refineMethod_CorrPool_costFunction']
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
                    fig_title_pre_str = 'Values of R\\&t Differences After Bundle Adjustment (BA) Including ' \
                                        'Intrinsics and Structure Using Degenerate Input Camera Matrices for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['stereoParameters_refineMethod_CorrPool_algorithm',
                                     'stereoParameters_refineMethod_CorrPool_costFunction']
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
                    it_parameters = ['stereoParameters_refineMethod_CorrPool_algorithm',
                                     'stereoParameters_refineMethod_CorrPool_costFunction']
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
                    it_parameters = ['stereoParameters_refineMethod_CorrPool_algorithm',
                                     'stereoParameters_refineMethod_CorrPool_costFunction']
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
        else:
            raise ValueError('Test nr does not exist')
    elif test_name == 'correspondence_pool':
        from statistics_and_plot import calcSatisticAndPlot_2D, \
            calcSatisticAndPlot_2D_partitions, \
            calcSatisticAndPlot_aggregate, \
            calcFromFuncAndPlot_3D_partitions, \
            calcSatisticAndPlot_3D, \
            calcFromFuncAndPlot_2D
        if not test_nr:
            raise ValueError('test_nr is required correspondence_pool')
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
                    it_parameters = ['stereoParameters_maxPoolCorrespondences',
                                     'stereoParameters_minPtsDistance']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_size_pts_dist_inlrat'}
                    # descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                    #         'multiple stereo frames'
                    # compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                    it_parameters = ['stereoParameters_maxPoolCorrespondences',
                                     'stereoParameters_minPtsDistance']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_size_pts_dist_best_comb_scenes'}
                    # descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                    #         'multiple stereo frames'
                    # compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                    it_parameters = ['stereoParameters_maxPoolCorrespondences',
                                     'stereoParameters_minPtsDistance']
                    # partitions = ['kpDistr', 'depthDistr', 'nrTP', 'kpAccSd', 'th']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_size_pts_dist_end_frames_best_comb_scenes'}
                    # descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                    #         'multiple stereo frames'
                    # compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                    it_parameters = ['stereoParameters_minPtsDistance']
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
                    it_parameters = ['stereoParameters_minPtsDistance']
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
                    it_parameters = ['stereoParameters_minPtsDistance']
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
                    it_parameters = ['stereoParameters_minPtsDistance']
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
                    it_parameters = ['stereoParameters_minPtsDistance']
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
                    it_parameters = ['stereoParameters_maxPoolCorrespondences',
                                     'stereoParameters_minPtsDistance']
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
                    it_parameters = ['stereoParameters_maxPoolCorrespondences',
                                     'stereoParameters_minPtsDistance']
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
                    it_parameters = ['stereoParameters_maxRat3DPtsFar',
                                     'stereoParameters_maxDist3DPtsZ']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_rat_dist_3Dpts_inlrat'}
                    # descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                    #         'multiple stereo frames'
                    # compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                    it_parameters = ['stereoParameters_maxRat3DPtsFar',
                                     'stereoParameters_maxDist3DPtsZ']
                    partitions = ['depthDistr', 'kpAccSd']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'corrpool_rat_dist_3Dpts_best_comb_scenes'}
                    # descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                    #         'multiple stereo frames'
                    # compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                    it_parameters = ['stereoParameters_maxRat3DPtsFar',
                                     'stereoParameters_maxDist3DPtsZ']
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
                    it_parameters = ['stereoParameters_maxRat3DPtsFar',
                                     'stereoParameters_maxDist3DPtsZ',
                                     'stereoParameters_maxPoolCorrespondences',
                                     'stereoParameters_minPtsDistance']
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 1, 'RT-stats', descr)
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
                    it_parameters = ['stereoParameters_maxRat3DPtsFar',
                                     'stereoParameters_maxDist3DPtsZ',
                                     'stereoParameters_maxPoolCorrespondences',
                                     'stereoParameters_minPtsDistance']
                    descr = 'Data for comparison from pose refinement without aggregation of correspondences over ' \
                            'multiple stereo frames'
                    repl_eval = {'actual': ['stereoRefine_us'], 'old': ['linRef_BA_sac_us'], 'new': ['comp_time']}
                    compare_source = get_compare_info(comp_pars, comp_path, 'refinement_ba', 1, 'time-agg', descr,
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
        from statistics_and_plot import calcSatisticAndPlot_2D, \
            calcSatisticAndPlot_3D, \
            calcSatisticAndPlot_2D_partitions, \
            calcFromFuncAndPlot_3D, \
            calcFromFuncAndPlot_2D_partitions, \
            calcFromFuncAndPlot_3D_partitions, \
            calcSatisticAndPlot_3D_partitions, \
            calcSatisticAndPlot_aggregate
        if test_nr == 1:
            if eval_nr[0] < 0:
                evals = list(range(1, 6))
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
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'res_par_name': 'robustness_best_comb_scenes_inlc'}
                    filter_func_args = {'data_seperators': ['stereoParameters_relInlRatThLast',
                                                            'stereoParameters_relInlRatThNew',
                                                            'stereoParameters_minInlierRatSkip',
                                                            'stereoParameters_minInlierRatioReInit',
                                                            'stereoParameters_relMinInlierRatSkip',
                                                            'inlratCRate']}
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
                elif ev == 2:
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
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True, True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'res_par_name': 'robustness_best_comb_scenes_inlc_depth'}
                    filter_func_args = {'data_seperators': ['stereoParameters_relInlRatThLast',
                                                            'stereoParameters_relInlRatThNew',
                                                            'stereoParameters_minInlierRatSkip',
                                                            'stereoParameters_minInlierRatioReInit',
                                                            'stereoParameters_relMinInlierRatSkip',
                                                            'inlratCRate',
                                                            'depthDistr']}
                    from robustness_eval import get_rt_change_type, get_best_comb_3d_scenes_1
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['inlratCRate', 'depthDistr'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_best_comb_3d_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True,
                                                             cat_sort='depthDistr')
                elif ev == 3:
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
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True, True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'res_par_name': 'robustness_best_comb_scenes_inlc_kpAccSd'}
                    filter_func_args = {'data_seperators': ['stereoParameters_relInlRatThLast',
                                                            'stereoParameters_relInlRatThNew',
                                                            'stereoParameters_minInlierRatSkip',
                                                            'stereoParameters_minInlierRatioReInit',
                                                            'stereoParameters_relMinInlierRatSkip',
                                                            'inlratCRate',
                                                            'kpAccSd']}
                    from robustness_eval import get_rt_change_type, get_best_comb_3d_scenes_1
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['inlratCRate', 'kpAccSd'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_best_comb_3d_scenes_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True,
                                                             cat_sort=None)
                elif ev == 4:
                    fig_title_pre_str = 'Differences of R\\&t Differences from Frame to Frame for Different '
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
                    filter_func_args = {'data_seperators': ['stereoParameters_relInlRatThLast',
                                                            'stereoParameters_relInlRatThNew',
                                                            'stereoParameters_minInlierRatSkip',
                                                            'stereoParameters_minInlierRatioReInit',
                                                            'stereoParameters_relMinInlierRatSkip',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_scene': 'jra'}
                    calc_func_args = {'data_separators': ['Nr', 'depthDistr', 'kpAccSd', 'inlratCRate'],
                                      'keepEval': ['R_diffAll', 't_angDiff_deg'],
                                      'additional_data': ['rt_change_pos', 'rt_change_type'],
                                      'eval_on': None,
                                      'diff_by': 'Nr'}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'data_separators': ['inlratCRate',
                                                              'kpAccSd',
                                                              'depthDistr'],
                                          'eval_on': ['R_diffAll'],
                                          'change_Nr': 25,
                                          'additional_data': ['rt_change_pos', 'rt_change_type'],
                                          'scene': 'jra',
                                          'res_par_name': 'robustness_delay_jra'}
                    from corr_pool_eval import calc_rt_diff2_frame_to_frame
                    from robustness_eval import get_rt_change_type, calc_calib_delay
                    ret += calcFromFuncAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_corrPool_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-diff',
                                                             eval_columns=eval_columns,  # Column names for which statistics are calculated (y-axis)
                                                             units=units,  # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,  # Algorithm parameters to evaluate
                                                             partitions=['depthDistr', 'kpAccSd', 'inlratCRate'],  # Data properties to calculate results separately
                                                             x_axis_column=['Nr'],  # x-axis column name
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=calc_calib_delay,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_rt_diff2_frame_to_frame,
                                                             calc_func_args=calc_func_args,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=False,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True,
                                                             cat_sort=False)
                elif ev == 5:
                    fig_title_pre_str = 'Differences of R\\&t Differences from Frame to Frame for Different '
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
                    filter_func_args = {'data_seperators': ['stereoParameters_relInlRatThLast',
                                                            'stereoParameters_relInlRatThNew',
                                                            'stereoParameters_minInlierRatSkip',
                                                            'stereoParameters_minInlierRatioReInit',
                                                            'stereoParameters_relMinInlierRatSkip',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_scene': 'jta'}
                    calc_func_args = {'data_separators': ['Nr', 'depthDistr', 'kpAccSd', 'inlratCRate'],
                                      'keepEval': ['R_diffAll', 't_angDiff_deg'],
                                      'additional_data': ['rt_change_pos', 'rt_change_type'],
                                      'eval_on': None,
                                      'diff_by': 'Nr'}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'data_separators': ['inlratCRate',
                                                              'kpAccSd',
                                                              'depthDistr'],
                                          'eval_on': ['t_angDiff_deg'],
                                          'change_Nr': 25,
                                          'additional_data': ['rt_change_pos', 'rt_change_type'],
                                          'scene': 'jta',
                                          'comp_res': ['robustness_best_comb_scenes_inlc',
                                                       'robustness_best_comb_scenes_inlc_depth',
                                                       'robustness_best_comb_scenes_inlc_kpAccSd',
                                                       'robustness_delay_jra',
                                                       'robustness_delay_jta'],
                                          'res_par_name': 'robustness_delay_jta'}
                    from corr_pool_eval import calc_rt_diff2_frame_to_frame
                    from robustness_eval import get_rt_change_type, calc_calib_delay
                    ret += calcFromFuncAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_corrPool_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-diff',
                                                             eval_columns=eval_columns,  # Column names for which statistics are calculated (y-axis)
                                                             units=units,  # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,  # Algorithm parameters to evaluate
                                                             partitions=['depthDistr', 'kpAccSd', 'inlratCRate'],  # Data properties to calculate results separately
                                                             x_axis_column=['Nr'],  # x-axis column name
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=calc_calib_delay,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_rt_diff2_frame_to_frame,
                                                             calc_func_args=calc_func_args,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=False,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True,
                                                             cat_sort=False)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        elif test_nr == 2:
            if eval_nr[0] < 0:
                evals = list(range(6, 11))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 6:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_checkPoolPoseRobust']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True, True),
                                          'use_marks': True,
                                          'func_name': 'get_best_comb_scenes',
                                          'res_par_name': 'robustness_best_comb_scenes_poolr_inlc'}
                    filter_func_args = {'data_seperators': ['stereoParameters_checkPoolPoseRobust',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'check_mostLikely': True}
                    from robustness_eval import get_rt_change_type, get_best_comb_scenes_ml_1
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
                                                             special_calcs_func=get_best_comb_scenes_ml_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False)
                elif ev == 7:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_checkPoolPoseRobust']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True, True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'res_par_name': 'robustness_best_comb_scenes_poolr_inlc_depth'}
                    filter_func_args = {'data_seperators': ['stereoParameters_checkPoolPoseRobust',
                                                            'inlratCRate',
                                                            'depthDistr',
                                                            'kpAccSd'],
                                        'check_mostLikely': True}
                    from robustness_eval import get_rt_change_type, get_best_comb_3d_scenes_ml_1
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['inlratCRate', 'depthDistr'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_best_comb_3d_scenes_ml_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort='depthDistr')
                elif ev == 8:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_checkPoolPoseRobust']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True, True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'res_par_name': 'robustness_best_comb_scenes_poolr_inlc_kpAccSd'}
                    filter_func_args = {'data_seperators': ['stereoParameters_checkPoolPoseRobust',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'check_mostLikely': True}
                    from robustness_eval import get_rt_change_type, get_best_comb_3d_scenes_ml_1
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['inlratCRate', 'kpAccSd'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_best_comb_3d_scenes_ml_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort=None)
                elif ev == 9:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_checkPoolPoseRobust']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True, True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'comp_res': ['robustness_best_comb_scenes_poolr_inlc_depth',
                                                       'robustness_best_comb_scenes_poolr_inlc_kpAccSd',
                                                       'robustness_best_comb_scenes_poolr_depth_kpAccSd'],
                                          'res_par_name': 'robustness_best_comb_scenes_poolr_depth_kpAccSd'}
                    filter_func_args = {'data_seperators': ['stereoParameters_checkPoolPoseRobust',
                                                            'depthDistr',
                                                            'kpAccSd',
                                                            'inlratCRate'],
                                        'check_mostLikely': True}
                    from robustness_eval import get_rt_change_type, get_best_comb_3d_scenes_ml_1
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['depthDistr', 'kpAccSd'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_best_comb_3d_scenes_ml_1,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort='depthDistr')
                elif ev == 10:
                    fig_title_pre_str = 'Statistics on Execution Times for Specific Combinations of '
                    eval_columns = ['stereoRefine_us']
                    units = [('stereoRefine_us', '/$\\mu s$')]
                    it_parameters = ['stereoParameters_checkPoolPoseRobust']
                    filter_func_args = {'data_seperators': ['stereoParameters_checkPoolPoseRobust',
                                                            'depthDistr',
                                                            'kpAccSd',
                                                            'inlratCRate']}
                    from robustness_eval import get_rt_change_type
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_robustness_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='time',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['rt_change_type'],
                                                  pdfsplitentry=None,
                                                  filter_func=get_rt_change_type,
                                                  filter_func_args=filter_func_args,
                                                  special_calcs_func=None,
                                                  special_calcs_args=None,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=None,
                                                  fig_type='ybar',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=False,
                                                  no_tex=False,
                                                  cat_sort=True)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        elif test_nr == 3:
            if eval_nr[0] < 0:
                evals = list(range(11, 15))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 11:
                    fig_title_pre_str = 'Statistics on R\\&t Differences for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    pdfsplitentry = ['t_distDiff', 'R_mostLikely_diffAll', 't_mostLikely_distDiff']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_partitions': ['rt_change_type', 'inlratCRate']}
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True}
                    from robustness_eval import get_rt_change_type, get_ml_acc
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_robustness_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratCRate'],
                                                  pdfsplitentry=pdfsplitentry,
                                                  filter_func=get_rt_change_type,
                                                  filter_func_args=filter_func_args,
                                                  special_calcs_func=get_ml_acc,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=False,
                                                  no_tex=False,
                                                  cat_sort=True)
                elif ev == 12:
                    fig_title_pre_str = 'Values on R\\&t Differences for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_partitions': ['depthDistr']}
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True}
                    from robustness_eval import get_rt_change_type, get_ml_acc
                    ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_robustness_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['inlratCRate', 'depthDistr'],
                                                  filter_func=get_rt_change_type,
                                                  filter_func_args=filter_func_args,
                                                  special_calcs_func=get_ml_acc,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  fig_type='surface',
                                                  use_marks=True,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True,
                                                  no_tex=False,
                                                  cat_sort='depthDistr')
                elif ev == 13:
                    fig_title_pre_str = 'Values on R\\&t Differences for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_partitions': ['kpAccSd']}
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True}
                    from robustness_eval import get_rt_change_type, get_ml_acc
                    ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_robustness_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['inlratCRate', 'kpAccSd'],
                                                  filter_func=get_rt_change_type,
                                                  filter_func_args=filter_func_args,
                                                  special_calcs_func=get_ml_acc,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  fig_type='surface',
                                                  use_marks=True,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True,
                                                  no_tex=False,
                                                  cat_sort=None)
                elif ev == 14:
                    fig_title_pre_str = 'Values on R\\&t Differences for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True}
                    from robustness_eval import get_rt_change_type
                    ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_robustness_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['depthDistr', 'kpAccSd'],
                                                  filter_func=get_rt_change_type,
                                                  filter_func_args=filter_func_args,
                                                  special_calcs_func=None,
                                                  special_calcs_args=None,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  fig_type='surface',
                                                  use_marks=True,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True,
                                                  no_tex=False,
                                                  cat_sort='depthDistr')
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        elif test_nr == 4:
            if eval_nr[0] < 0:
                evals = list(range(15, 25))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 15:
                    fig_title_pre_str = 'Values on the Relative Ratio of Stable Pose Detections ' \
                                        'for Different '
                    eval_columns = ['R_diffAll', 't_angDiff_deg', 'R_mostLikely_diffAll', 't_mostLikely_angDiff_deg']
                    units = [('R_diffAll', '/\\textdegree'), ('t_angDiff_deg', '/\\textdegree'),
                             ('R_mostLikely_diffAll', '/\\textdegree'), ('t_mostLikely_angDiff_deg', '/\\textdegree')]
                    it_parameters = ['stereoParameters_minContStablePoses',
                                     'stereoParameters_minNormDistStable',
                                     'stereoParameters_absThRankingStable']
                    partitions = ['rt_change_type']
                    calc_func_args = {'data_separators': ['stereoParameters_minContStablePoses',
                                                          'stereoParameters_minNormDistStable',
                                                          'stereoParameters_absThRankingStable',
                                                          'rt_change_type',
                                                          'depthDistr',
                                                          'kpAccSd',
                                                          'inlratCRate'],
                                      'stable_type': 'poseIsStable',
                                      'remove_partitions': ['depthDistr', 'kpAccSd']}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_separators': ['rt_change_type', 'inlratCRate'],
                                          'to_int_cols': ['stereoParameters_minContStablePoses'],
                                          'on_2nd_axis': 'stereoParameters_minNormDistStable',
                                          'stable_type': 'poseIsStable',
                                          'res_par_name': 'robustness_best_pose_stable_pars'}
                    filter_func_args = {'data_seperators': ['stereoParameters_minContStablePoses',
                                                            'stereoParameters_minNormDistStable',
                                                            'stereoParameters_absThRankingStable',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True}
                    from robustness_eval import get_rt_change_type, calc_pose_stable_ratio, get_best_stability_pars
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stabiRat',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratCRate'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_best_stability_pars,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_pose_stable_ratio,
                                                             calc_func_args=calc_func_args,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort=False)
                elif ev == 16:
                    fig_title_pre_str = 'Values of R\\&t Differences on Stable Pose Detections and ' \
                                        'Standard Pose Estimations for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_minContStablePoses',
                                     'stereoParameters_minNormDistStable',
                                     'stereoParameters_absThRankingStable']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_partitions': ['rt_change_type', 'inlratCRate'],
                                          'eval_it_pars': True,
                                          'cat_sort': 'rt_change_type',
                                          'func_name': 'get_ml_acc_inlc_rt',
                                          'stable_type': 'poseIsStable',
                                          'meta_it_pars': ['stereoParameters_minContStablePoses'],
                                          'res_par_name': 'robustness_best_stable_pars_inlc_rt'}
                    filter_func_args = {'data_seperators': ['stereoParameters_minContStablePoses',
                                                            'stereoParameters_minNormDistStable',
                                                            'stereoParameters_absThRankingStable',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_poseIsStable': True}
                    from robustness_eval import get_rt_change_type, get_ml_acc
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stabi',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratCRate'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_ml_acc,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort=False)
                elif ev == 17:
                    fig_title_pre_str = 'Values of R\\&t Differences on Stable Pose Detections and ' \
                                        'Standard Pose Estimations for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_minContStablePoses',
                                     'stereoParameters_minNormDistStable',
                                     'stereoParameters_absThRankingStable']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_partitions': ['depthDistr'],
                                          'eval_it_pars': True,
                                          'func_name': 'get_ml_acc_depthDistr',
                                          'stable_type': 'poseIsStable',
                                          'meta_it_pars': ['stereoParameters_minContStablePoses'],
                                          'res_par_name': 'robustness_best_stable_pars_depthDistr'}
                    filter_func_args = {'data_seperators': ['stereoParameters_minContStablePoses',
                                                            'stereoParameters_minNormDistStable',
                                                            'stereoParameters_absThRankingStable',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_poseIsStable': True}
                    from robustness_eval import get_rt_change_type, get_ml_acc
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stabi',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['inlratCRate', 'depthDistr'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_ml_acc,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort='depthDistr')
                elif ev == 18:
                    fig_title_pre_str = 'Values of R\\&t Differences on Stable Pose Detections and ' \
                                        'Standard Pose Estimations for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_minContStablePoses',
                                     'stereoParameters_minNormDistStable',
                                     'stereoParameters_absThRankingStable']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_partitions': ['kpAccSd'],
                                          'eval_it_pars': True,
                                          'func_name': 'get_ml_acc_kpAccSd',
                                          'stable_type': 'poseIsStable',
                                          'meta_it_pars': ['stereoParameters_minContStablePoses'],
                                          'res_par_name': 'robustness_best_stable_pars_kpAccSd'}
                    filter_func_args = {'data_seperators': ['stereoParameters_minContStablePoses',
                                                            'stereoParameters_minNormDistStable',
                                                            'stereoParameters_absThRankingStable',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_poseIsStable': True}
                    from robustness_eval import get_rt_change_type, get_ml_acc
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stabi',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['inlratCRate', 'kpAccSd'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_ml_acc,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort=None)
                elif ev == 19:
                    fig_title_pre_str = 'Values of R\\&t Differences on Stable Pose Detections and ' \
                                        'Standard Pose Estimations for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_minContStablePoses',
                                     'stereoParameters_minNormDistStable',
                                     'stereoParameters_absThRankingStable']
                    partitions = ['rt_change_type']
                    filter_func_args = {'data_seperators': ['stereoParameters_minContStablePoses',
                                                            'stereoParameters_minNormDistStable',
                                                            'stereoParameters_absThRankingStable',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_poseIsStable': True}
                    from robustness_eval import get_rt_change_type
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stabi',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['depthDistr', 'kpAccSd'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=None,
                                                             special_calcs_args=None,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort='depthDistr')
                elif ev == 20:
                    fig_title_pre_str = 'Values on the Relative Ratio of Stable Most Likely Pose Detections ' \
                                        'for Different '
                    eval_columns = ['R_diffAll', 't_angDiff_deg', 'R_mostLikely_diffAll', 't_mostLikely_angDiff_deg']
                    units = [('R_diffAll', '/\\textdegree'), ('t_angDiff_deg', '/\\textdegree'),
                             ('R_mostLikely_diffAll', '/\\textdegree'), ('t_mostLikely_angDiff_deg', '/\\textdegree')]
                    it_parameters = ['stereoParameters_minContStablePoses',
                                     'stereoParameters_minNormDistStable',
                                     'stereoParameters_absThRankingStable']
                    partitions = ['rt_change_type']
                    calc_func_args = {'data_separators': ['stereoParameters_minContStablePoses',
                                                          'stereoParameters_minNormDistStable',
                                                          'stereoParameters_absThRankingStable',
                                                          'rt_change_type',
                                                          'depthDistr',
                                                          'kpAccSd',
                                                          'inlratCRate'],
                                      'stable_type': 'mostLikelyPose_stable',
                                      'remove_partitions': ['depthDistr', 'kpAccSd']}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_separators': ['rt_change_type', 'inlratCRate'],
                                          'to_int_cols': ['stereoParameters_minContStablePoses'],
                                          'on_2nd_axis': 'stereoParameters_minNormDistStable',
                                          'stable_type': 'mostLikelyPose_stable',
                                          'func_name': 'get_best_stability_pars_stMl',
                                          'res_par_name': 'robustness_best_pose_stable_pars_stMl'}
                    filter_func_args = {'data_seperators': ['stereoParameters_minContStablePoses',
                                                            'stereoParameters_minNormDistStable',
                                                            'stereoParameters_absThRankingStable',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True}
                    from robustness_eval import get_rt_change_type, calc_pose_stable_ratio, get_best_stability_pars
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stabiRatMl',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratCRate'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_best_stability_pars,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_pose_stable_ratio,
                                                             calc_func_args=calc_func_args,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort=False)
                elif ev == 21:
                    fig_title_pre_str = 'Values of R\\&t Differences on Stable Most Likely Pose Detections and ' \
                                        'Standard Pose Estimations for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_minContStablePoses',
                                     'stereoParameters_minNormDistStable',
                                     'stereoParameters_absThRankingStable']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_partitions': ['rt_change_type', 'inlratCRate'],
                                          'eval_it_pars': True,
                                          'cat_sort': 'rt_change_type',
                                          'func_name': 'get_ml_acc_inlc_rt_stMl',
                                          'stable_type': 'mostLikelyPose_stable',
                                          'meta_it_pars': ['stereoParameters_minContStablePoses'],
                                          'res_par_name': 'robustness_best_stable_pars_inlc_rt_stMl'}
                    filter_func_args = {'data_seperators': ['stereoParameters_minContStablePoses',
                                                            'stereoParameters_minNormDistStable',
                                                            'stereoParameters_absThRankingStable',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_mostLikelyPose_stable': True}
                    from robustness_eval import get_rt_change_type, get_ml_acc
                    ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stabiMl',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             x_axis_column=['inlratCRate'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_ml_acc,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort=False)
                elif ev == 22:
                    fig_title_pre_str = 'Values of R\\&t Differences on Stable Most Likely Pose Detections and ' \
                                        'Standard Pose Estimations for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_minContStablePoses',
                                     'stereoParameters_minNormDistStable',
                                     'stereoParameters_absThRankingStable']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_partitions': ['depthDistr'],
                                          'eval_it_pars': True,
                                          'func_name': 'get_ml_acc_depthDistr_stMl',
                                          'stable_type': 'mostLikelyPose_stable',
                                          'meta_it_pars': ['stereoParameters_minContStablePoses'],
                                          'res_par_name': 'robustness_best_stable_pars_depthDistr_stMl'}
                    filter_func_args = {'data_seperators': ['stereoParameters_minContStablePoses',
                                                            'stereoParameters_minNormDistStable',
                                                            'stereoParameters_absThRankingStable',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_mostLikelyPose_stable': True}
                    from robustness_eval import get_rt_change_type, get_ml_acc
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stabiMl',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['inlratCRate', 'depthDistr'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_ml_acc,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort='depthDistr')
                elif ev == 23:
                    fig_title_pre_str = 'Values of R\\&t Differences on Stable Most Likely Pose Detections and ' \
                                        'Standard Pose Estimations for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_minContStablePoses',
                                     'stereoParameters_minNormDistStable',
                                     'stereoParameters_absThRankingStable']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': False,
                                          'data_partitions': ['kpAccSd'],
                                          'eval_it_pars': True,
                                          'func_name': 'get_ml_acc_kpAccSd_stMl',
                                          'stable_type': 'mostLikelyPose_stable',
                                          'meta_it_pars': ['stereoParameters_minContStablePoses'],
                                          'res_par_name': 'robustness_best_stable_pars_kpAccSd_stMl'}
                    filter_func_args = {'data_seperators': ['stereoParameters_minContStablePoses',
                                                            'stereoParameters_minNormDistStable',
                                                            'stereoParameters_absThRankingStable',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_mostLikelyPose_stable': True}
                    from robustness_eval import get_rt_change_type, get_ml_acc
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stabiMl',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['inlratCRate', 'kpAccSd'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_ml_acc,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort=None)
                elif ev == 24:
                    fig_title_pre_str = 'Values of R\\&t Differences on Stable Most Likely Pose Detections and ' \
                                        'Standard Pose Estimations for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                    'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', ''),
                             ('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['stereoParameters_minContStablePoses',
                                     'stereoParameters_minNormDistStable',
                                     'stereoParameters_absThRankingStable']
                    partitions = ['rt_change_type']
                    filter_func_args = {'data_seperators': ['stereoParameters_minContStablePoses',
                                                            'stereoParameters_minNormDistStable',
                                                            'stereoParameters_absThRankingStable',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_mostLikelyPose_stable': True}
                    from robustness_eval import get_rt_change_type
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stabiMl',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['depthDistr', 'kpAccSd'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=None,
                                                             special_calcs_args=None,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort='depthDistr')
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        elif test_nr == 5:
            if eval_nr[0] < 0:
                evals = list(range(25, 29))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 25:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['stereoParameters_useRANSAC_fewMatches']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'smooth',
                                          'data_separators': ['inlratCRate'],
                                          'res_par_name': 'robustness_ransac_fewMatch_inlc'}
                    filter_func_args = {'data_seperators': ['stereoParameters_useRANSAC_fewMatches',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr']}
                    from robustness_eval import get_rt_change_type, get_best_robust_pool_pars
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
                                                             special_calcs_func=get_best_robust_pool_pars,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             compare_source=None,
                                                             fig_type='smooth',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort=False)
                elif ev == 26:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['stereoParameters_useRANSAC_fewMatches']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'data_separators': ['inlratCRate', 'depthDistr'],
                                          'split_fig_data': 'depthDistr',
                                          'res_par_name': 'robustness_ransac_fewMatch_inlc_depth'}
                    filter_func_args = {'data_seperators': ['stereoParameters_useRANSAC_fewMatches',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr']}
                    from robustness_eval import get_rt_change_type, get_best_robust_pool_pars
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['depthDistr', 'inlratCRate'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_best_robust_pool_pars,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort='depthDistr')
                elif ev == 27:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['stereoParameters_useRANSAC_fewMatches']
                    partitions = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'data_separators': ['inlratCRate', 'kpAccSd'],
                                          'split_fig_data': 'kpAccSd',
                                          'comp_res': ['robustness_ransac_fewMatch_inlc',
                                                       'robustness_ransac_fewMatch_inlc_depth',
                                                       'robustness_ransac_fewMatch_inlc_kpAcc'],
                                          'res_par_name': 'robustness_ransac_fewMatch_inlc_kpAcc'}
                    filter_func_args = {'data_seperators': ['stereoParameters_useRANSAC_fewMatches',
                                                            'inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr']}
                    from robustness_eval import get_rt_change_type, get_best_robust_pool_pars
                    ret += calcSatisticAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-stats',
                                                             eval_columns=eval_columns,
                                                             units=units,
                                                             it_parameters=it_parameters,
                                                             partitions=partitions,
                                                             xy_axis_columns=['kpAccSd', 'inlratCRate'],
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=get_best_robust_pool_pars,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=None,
                                                             calc_func_args=None,
                                                             fig_type='surface',
                                                             use_marks=True,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=True,
                                                             figs_externalize=True,
                                                             no_tex=False,
                                                             cat_sort=None)
                elif ev == 28:
                    fig_title_pre_str = 'Statistics on Execution Times for Different '
                    eval_columns = ['stereoRefine_us']
                    units = [('stereoRefine_us', '/$\\mu s$')]
                    it_parameters = ['stereoParameters_useRANSAC_fewMatches']
                    ret += calcSatisticAndPlot_aggregate(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_robustness_',
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
                                                         fig_type='ybar',
                                                         use_marks=False,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=False)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        elif test_nr == 6:
            if eval_nr[0] < 0:
                evals = list(range(29, 38))
            else:
                evals = eval_nr
            for ev in evals:
                if ev == 29:
                    fig_title_pre_str = 'Statistics on R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    pdfsplitentry = ['t_distDiff']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'smooth',
                                          'data_separators': ['rt_change_type']}
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr']}
                    from robustness_eval import get_rt_change_type, get_cRT_stats
                    ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_robustness_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  x_axis_column=['inlratCRate'],
                                                  pdfsplitentry=pdfsplitentry,
                                                  filter_func=get_rt_change_type,
                                                  filter_func_args=filter_func_args,
                                                  special_calcs_func=get_cRT_stats,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  compare_source=None,
                                                  fig_type='smooth',
                                                  use_marks=True,
                                                  ctrl_fig_size=True,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=False,
                                                  no_tex=False,
                                                  cat_sort=False)
                elif ev == 30:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'data_separators': ['rt_change_type', 'depthDistr']}
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr']}
                    from robustness_eval import get_rt_change_type, get_cRT_stats
                    ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_robustness_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['inlratCRate', 'depthDistr'],
                                                  filter_func=get_rt_change_type,
                                                  filter_func_args=filter_func_args,
                                                  special_calcs_func=get_cRT_stats,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  fig_type='surface',
                                                  use_marks=True,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True,
                                                  no_tex=False,
                                                  cat_sort='depthDistr')
                elif ev == 31:
                    fig_title_pre_str = 'Values of R\\&t Differences for Combinations of Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'fig_type': 'surface',
                                          'data_separators': ['rt_change_type', 'kpAccSd']}
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr']}
                    from robustness_eval import get_rt_change_type, get_cRT_stats
                    ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                                  store_path=output_path,
                                                  tex_file_pre_str='plots_robustness_',
                                                  fig_title_pre_str=fig_title_pre_str,
                                                  eval_description_path='RT-stats',
                                                  eval_columns=eval_columns,
                                                  units=units,
                                                  it_parameters=it_parameters,
                                                  xy_axis_columns=['inlratCRate', 'kpAccSd'],
                                                  filter_func=get_rt_change_type,
                                                  filter_func_args=filter_func_args,
                                                  special_calcs_func=get_cRT_stats,
                                                  special_calcs_args=special_calcs_args,
                                                  calc_func=None,
                                                  calc_func_args=None,
                                                  fig_type='surface',
                                                  use_marks=True,
                                                  ctrl_fig_size=False,
                                                  make_fig_index=True,
                                                  build_pdf=True,
                                                  figs_externalize=True,
                                                  no_tex=False,
                                                  cat_sort=None)
                elif ev == 32:
                    fig_title_pre_str = 'Differences of R\\&t Differences from Frame to Frame ' \
                                        'of Scenes with Abrupt Changes of R for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    partitions = ['depthDistr', 'kpAccSd']
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_scene': ['jra', 'jrx', 'jry', 'jrz']}
                    calc_func_args = {'data_separators': ['Nr', 'depthDistr', 'kpAccSd', 'inlratCRate'],
                                      'keepEval': ['R_diffAll', 't_angDiff_deg'],
                                      'additional_data': ['rt_change_pos'],
                                      'eval_on': None,
                                      'diff_by': 'Nr'}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'data_separators': ['inlratCRate',
                                                              'kpAccSd',
                                                              'depthDistr',
                                                              'rt_change_type'],
                                          'eval_on': ['R_diffAll'],
                                          'change_Nr': 25,
                                          'additional_data': ['rt_change_pos'],
                                          'func_name': 'calc_calib_delay_noPar_rot',
                                          'res_par_name': 'robustness_mean_frame_delay_rot'}
                    from corr_pool_eval import calc_rt_diff2_frame_to_frame
                    from robustness_eval import get_rt_change_type, calc_calib_delay_noPar
                    ret += calcFromFuncAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-diffR',
                                                             eval_columns=eval_columns,
                                                             # Column names for which statistics are calculated (y-axis)
                                                             units=units,
                                                             # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,
                                                             # Algorithm parameters to evaluate
                                                             partitions=partitions,
                                                             # Data properties to calculate results separately
                                                             xy_axis_columns=[],  # x-axis column name
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=calc_calib_delay_noPar,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_rt_diff2_frame_to_frame,
                                                             calc_func_args=calc_func_args,
                                                             fig_type='surface',
                                                             use_marks=False,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True,
                                                             cat_sort=False)
                elif ev == 33:
                    fig_title_pre_str = 'Differences of R\\&t Differences from Frame to Frame ' \
                                        'of Scenes with Abrupt Changes of t for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    partitions = ['depthDistr', 'kpAccSd']
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_scene': ['jta', 'jtx', 'jty', 'jtz']}
                    calc_func_args = {'data_separators': ['Nr', 'depthDistr', 'kpAccSd', 'inlratCRate'],
                                      'keepEval': ['R_diffAll', 't_angDiff_deg'],
                                      'additional_data': ['rt_change_pos'],
                                      'eval_on': None,
                                      'diff_by': 'Nr'}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'data_separators': ['inlratCRate',
                                                              'kpAccSd',
                                                              'depthDistr',
                                                              'rt_change_type'],
                                          'eval_on': ['t_angDiff_deg'],
                                          'change_Nr': 25,
                                          'additional_data': ['rt_change_pos'],
                                          'func_name': 'calc_calib_delay_noPar_trans',
                                          'res_par_name': 'robustness_mean_frame_delay_trans'}
                    from corr_pool_eval import calc_rt_diff2_frame_to_frame
                    from robustness_eval import get_rt_change_type, calc_calib_delay_noPar
                    ret += calcFromFuncAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-diffT',
                                                             eval_columns=eval_columns,
                                                             # Column names for which statistics are calculated (y-axis)
                                                             units=units,
                                                             # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,
                                                             # Algorithm parameters to evaluate
                                                             partitions=partitions,
                                                             # Data properties to calculate results separately
                                                             xy_axis_columns=[],  # x-axis column name
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=calc_calib_delay_noPar,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_rt_diff2_frame_to_frame,
                                                             calc_func_args=calc_func_args,
                                                             fig_type='surface',
                                                             use_marks=False,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True,
                                                             cat_sort=False)
                elif ev == 34:
                    fig_title_pre_str = 'Differences of R\\&t Differences from Frame to Frame of Scenes ' \
                                        'with Abrupt Changes of R\\&t for Different '
                    eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                    units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                             ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                             ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                             ('t_diff_ty', ''), ('t_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    partitions = ['depthDistr', 'kpAccSd']
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_scene': 'jrt'}
                    calc_func_args = {'data_separators': ['Nr', 'depthDistr', 'kpAccSd', 'inlratCRate'],
                                      'keepEval': ['R_diffAll', 't_angDiff_deg'],
                                      'additional_data': ['rt_change_pos'],
                                      'eval_on': None,
                                      'diff_by': 'Nr'}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'data_separators': ['inlratCRate',
                                                              'kpAccSd',
                                                              'depthDistr'],
                                          'eval_on': ['R_diffAll'],#['Rt_diff2']
                                          'is_jrt': True,
                                          # 'comb_rt': True,
                                          'change_Nr': 25,
                                          'additional_data': ['rt_change_pos'],
                                          'func_name': 'calc_calib_delay_noPar_rt',
                                          'res_par_name': 'robustness_mean_frame_delay_rt'}
                    from corr_pool_eval import calc_rt_diff2_frame_to_frame
                    from robustness_eval import get_rt_change_type, calc_calib_delay_noPar
                    ret += calcFromFuncAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-diffRT',
                                                             eval_columns=eval_columns,
                                                             # Column names for which statistics are calculated (y-axis)
                                                             units=units,
                                                             # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,
                                                             # Algorithm parameters to evaluate
                                                             partitions=partitions,
                                                             # Data properties to calculate results separately
                                                             xy_axis_columns=[],  # x-axis column name
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=calc_calib_delay_noPar,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_rt_diff2_frame_to_frame,
                                                             calc_func_args=calc_func_args,
                                                             fig_type='surface',
                                                             use_marks=False,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True,
                                                             cat_sort=False)
                elif ev == 35:
                    fig_title_pre_str = 'Differences of Most Likely R\\&t Differences ' \
                                        'from Frame to Frame of Scenes with Abrupt Changes of R for Different '
                    eval_columns = ['R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    partitions = ['depthDistr', 'kpAccSd']
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_scene': ['jra', 'jrx', 'jry', 'jrz']}
                    calc_func_args = {'data_separators': ['Nr', 'depthDistr', 'kpAccSd', 'inlratCRate'],
                                      'keepEval': ['R_mostLikely_diffAll', 't_mostLikely_angDiff_deg'],
                                      'additional_data': ['rt_change_pos'],
                                      'eval_on': None,
                                      'diff_by': 'Nr'}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'data_separators': ['inlratCRate',
                                                              'kpAccSd',
                                                              'depthDistr',
                                                              'rt_change_type'],
                                          'eval_on': ['R_mostLikely_diffAll'],
                                          'change_Nr': 25,
                                          'additional_data': ['rt_change_pos'],
                                          'func_name': 'calc_calib_delay_noPar_rotMl',
                                          'res_par_name': 'robustness_mean_frame_delay_rotMl'}
                    from corr_pool_eval import calc_rt_diff2_frame_to_frame
                    from robustness_eval import get_rt_change_type, calc_calib_delay_noPar
                    ret += calcFromFuncAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-diffRMl',
                                                             eval_columns=eval_columns,
                                                             # Column names for which statistics are calculated (y-axis)
                                                             units=units,
                                                             # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,
                                                             # Algorithm parameters to evaluate
                                                             partitions=partitions,
                                                             # Data properties to calculate results separately
                                                             xy_axis_columns=[],  # x-axis column name
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=calc_calib_delay_noPar,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_rt_diff2_frame_to_frame,
                                                             calc_func_args=calc_func_args,
                                                             fig_type='surface',
                                                             use_marks=False,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True,
                                                             cat_sort=False)
                elif ev == 36:
                    fig_title_pre_str = 'Differences of Most Likely R\\&t Differences ' \
                                        'from Frame to Frame of Scenes with Abrupt Changes of t for Different '
                    eval_columns = ['R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    partitions = ['depthDistr', 'kpAccSd']
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_scene': ['jta', 'jtx', 'jty', 'jtz']}
                    calc_func_args = {'data_separators': ['Nr', 'depthDistr', 'kpAccSd', 'inlratCRate'],
                                      'keepEval': ['R_mostLikely_diffAll', 't_mostLikely_angDiff_deg'],
                                      'additional_data': ['rt_change_pos'],
                                      'eval_on': None,
                                      'diff_by': 'Nr'}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'data_separators': ['inlratCRate',
                                                              'kpAccSd',
                                                              'depthDistr',
                                                              'rt_change_type'],
                                          'eval_on': ['t_mostLikely_angDiff_deg'],
                                          'change_Nr': 25,
                                          'additional_data': ['rt_change_pos'],
                                          'func_name': 'calc_calib_delay_noPar_transMl',
                                          'res_par_name': 'robustness_mean_frame_delay_transMl'}
                    from corr_pool_eval import calc_rt_diff2_frame_to_frame
                    from robustness_eval import get_rt_change_type, calc_calib_delay_noPar
                    ret += calcFromFuncAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-diffTMl',
                                                             eval_columns=eval_columns,
                                                             # Column names for which statistics are calculated (y-axis)
                                                             units=units,
                                                             # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,
                                                             # Algorithm parameters to evaluate
                                                             partitions=partitions,
                                                             # Data properties to calculate results separately
                                                             xy_axis_columns=[],  # x-axis column name
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=calc_calib_delay_noPar,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_rt_diff2_frame_to_frame,
                                                             calc_func_args=calc_func_args,
                                                             fig_type='surface',
                                                             use_marks=False,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True,
                                                             cat_sort=False)
                elif ev == 37:
                    fig_title_pre_str = 'Differences of Most Likely R\\&t Differences ' \
                                        'from Frame to Frame of Scenes with Abrupt Changes of R\\&t for Different '
                    eval_columns = ['R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                    'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                    't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                    't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz']
                    units = [('R_mostLikely_diffAll', '/\\textdegree'),
                             ('R_mostLikely_diff_roll_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_pitch_deg', '/\\textdegree'),
                             ('R_mostLikely_diff_yaw_deg', '/\\textdegree'),
                             ('t_mostLikely_angDiff_deg', '/\\textdegree'),
                             ('t_mostLikely_distDiff', ''), ('t_mostLikely_diff_tx', ''),
                             ('t_mostLikely_diff_ty', ''), ('t_mostLikely_diff_tz', '')]
                    it_parameters = ['rt_change_type']
                    partitions = ['depthDistr', 'kpAccSd']
                    filter_func_args = {'data_seperators': ['inlratCRate',
                                                            'kpAccSd',
                                                            'depthDistr'],
                                        'filter_mostLikely': True,
                                        'filter_scene': 'jrt'}
                    calc_func_args = {'data_separators': ['Nr', 'depthDistr', 'kpAccSd', 'inlratCRate'],
                                      'keepEval': ['R_mostLikely_diffAll', 't_mostLikely_angDiff_deg'],
                                      'additional_data': ['rt_change_pos'],
                                      'eval_on': None,
                                      'diff_by': 'Nr'}
                    special_calcs_args = {'build_pdf': (True, True),
                                          'use_marks': True,
                                          'data_separators': ['inlratCRate',
                                                              'kpAccSd',
                                                              'depthDistr'],
                                          'eval_on': ['R_mostLikely_diffAll'],#['Rt_diff2']
                                          'is_jrt': True,
                                          # 'comb_rt': True,
                                          'change_Nr': 25,
                                          'additional_data': ['rt_change_pos'],
                                          'func_name': 'calc_calib_delay_noPar_rtMl',
                                          'res_par_name': 'robustness_mean_frame_delay_rtMl'}
                    from corr_pool_eval import calc_rt_diff2_frame_to_frame
                    from robustness_eval import get_rt_change_type, calc_calib_delay_noPar
                    ret += calcFromFuncAndPlot_3D_partitions(data=data.copy(deep=True),
                                                             store_path=output_path,
                                                             tex_file_pre_str='plots_robustness_',
                                                             fig_title_pre_str=fig_title_pre_str,
                                                             eval_description_path='RT-diffRTMl',
                                                             eval_columns=eval_columns,
                                                             # Column names for which statistics are calculated (y-axis)
                                                             units=units,
                                                             # Units in string format for every entry of eval_columns
                                                             it_parameters=it_parameters,
                                                             # Algorithm parameters to evaluate
                                                             partitions=partitions,
                                                             # Data properties to calculate results separately
                                                             xy_axis_columns=[],  # x-axis column name
                                                             filter_func=get_rt_change_type,
                                                             filter_func_args=filter_func_args,
                                                             special_calcs_func=calc_calib_delay_noPar,
                                                             special_calcs_args=special_calcs_args,
                                                             calc_func=calc_rt_diff2_frame_to_frame,
                                                             calc_func_args=calc_func_args,
                                                             fig_type='surface',
                                                             use_marks=False,
                                                             ctrl_fig_size=True,
                                                             make_fig_index=True,
                                                             build_pdf=False,
                                                             figs_externalize=True,
                                                             no_tex=True,
                                                             cat_sort=False)
                else:
                    raise ValueError('Eval nr ' + ev + ' does not exist')
        else:
            raise ValueError('Test nr does not exist')
    elif test_name == 'usac_vs_autocalib':
        from statistics_and_plot import calcSatisticAndPlot_2D, \
            calcSatisticAndPlot_3D, \
            calcSatisticAndPlot_2D_partitions, \
            calcFromFuncAndPlot_3D, \
            calcFromFuncAndPlot_2D_partitions, \
            calcSatisticAndPlot_aggregate, \
            calcFromFuncAndPlot_2D
        if eval_nr[0] < 0:
            evals = list(range(1, 9))
        else:
            evals = eval_nr
        for ev in evals:
            if ev == 1:
                fig_title_pre_str = 'Statistics on R\\&t Differences of Scenes with Stable Stereo Poses ' \
                                    'for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                it_parameters = ['stereoRef']
                filter_func_args = {'data_seperators': ['inlratMin',
                                                        'kpAccSd',
                                                        'depthDistr'],
                                    'filter_scene': 'nv'}
                calc_func_args = {'data_separators': ['depthDistr', 'kpAccSd', 'inlratMin']}
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': True,
                                      'fig_type': 'smooth',
                                      'func_name': 'comp_pars_vs_inlRats',
                                      'res_par_name': 'usac_vs_autoc_stabRT_inlrat'}
                from usac_eval import get_best_comb_inlrat_1
                from robustness_eval import get_rt_change_type
                from usac_vs_autocalib_eval import get_accum_corrs_sequs
                ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_usacVsAuto_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='RT-stats',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              x_axis_column=['inlratMin'],
                                              pdfsplitentry=['t_distDiff'],
                                              filter_func=get_rt_change_type,
                                              filter_func_args=filter_func_args,
                                              special_calcs_func=get_best_comb_inlrat_1,
                                              special_calcs_args=special_calcs_args,
                                              calc_func=get_accum_corrs_sequs,
                                              calc_func_args=calc_func_args,
                                              compare_source=None,
                                              fig_type='smooth',
                                              use_marks=True,
                                              ctrl_fig_size=True,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=True)
            elif ev == 2:
                fig_title_pre_str = 'Values of R\\&t Differences of Scenes with Stable Stereo Poses ' \
                                    'for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                it_parameters = ['stereoRef']
                filter_func_args = {'data_seperators': ['inlratMin',
                                                        'kpAccSd',
                                                        'depthDistr'],
                                    'filter_scene': 'nv'}
                calc_func_args = {'data_separators': ['depthDistr', 'kpAccSd', 'inlratMin']}
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': True,
                                      'fig_type': 'surface',
                                      'used_x_axis': 'kpAccSd',
                                      'cols_for_mean': ['inlratMin'],
                                      'func_name': 'comp_pars_vs_kpAccSd'}
                from robustness_eval import get_rt_change_type
                from usac_vs_autocalib_eval import get_mean_y_vs_x_it, get_accum_corrs_sequs
                ret += calcSatisticAndPlot_3D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_usacVsAuto_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='RT-stats',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              xy_axis_columns=['kpAccSd', 'inlratMin'],
                                              filter_func=get_rt_change_type,
                                              filter_func_args=filter_func_args,
                                              special_calcs_func=get_mean_y_vs_x_it,
                                              special_calcs_args=special_calcs_args,
                                              calc_func=get_accum_corrs_sequs,
                                              calc_func_args=calc_func_args,
                                              fig_type='surface',
                                              use_marks=True,
                                              ctrl_fig_size=False,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=True,
                                              no_tex=False,
                                              cat_sort=None)
            elif ev == 3:
                fig_title_pre_str = 'Statistics on R\\&t Differences of Scenes with Stable Stereo Poses ' \
                                    'for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                it_parameters = ['stereoRef']
                filter_func_args = {'data_seperators': ['inlratMin',
                                                        'kpAccSd',
                                                        'depthDistr'],
                                    'filter_scene': 'nv'}
                calc_func_args = {'data_separators': ['depthDistr', 'kpAccSd', 'inlratMin']}
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': True,
                                      'fig_type': 'smooth',
                                      'func_name': 'comp_pars_vs_depthDistr',
                                      'res_par_name': 'usac_vs_autoc_stabRT_depth'}
                from usac_eval import get_best_comb_inlrat_1
                from robustness_eval import get_rt_change_type
                from usac_vs_autocalib_eval import get_accum_corrs_sequs
                ret += calcSatisticAndPlot_2D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_usacVsAuto_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='RT-stats',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              x_axis_column=['depthDistr'],
                                              pdfsplitentry=['t_distDiff'],
                                              filter_func=get_rt_change_type,
                                              filter_func_args=filter_func_args,
                                              special_calcs_func=get_best_comb_inlrat_1,
                                              special_calcs_args=special_calcs_args,
                                              calc_func=get_accum_corrs_sequs,
                                              calc_func_args=calc_func_args,
                                              compare_source=None,
                                              fig_type='smooth',
                                              use_marks=True,
                                              ctrl_fig_size=True,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=True,
                                              no_tex=False,
                                              cat_sort=True)
            elif ev == 4:
                fig_title_pre_str = 'Values of R\\&t Differences of Scenes with Stable Stereo Poses ' \
                                    'for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                it_parameters = ['stereoRef']
                partitions = ['inlratMin']
                filter_func_args = {'data_seperators': ['inlratMin',
                                                        'kpAccSd',
                                                        'depthDistr'],
                                    'filter_scene': 'nv'}
                calc_func_args = {'data_separators': ['depthDistr', 'kpAccSd', 'inlratMin']}
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': False,
                                      'fig_type': 'sharp plot',
                                      'used_x_axis': 'Nr',
                                      'cols_for_mean': ['inlratMin'],
                                      'func_name': 'comp_pars_vs_FNr'}
                from usac_eval import get_best_comb_inlrat_1
                from robustness_eval import get_rt_change_type
                from usac_vs_autocalib_eval import get_accum_corrs_sequs, get_mean_y_vs_x_it
                ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_usacVsAuto_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='RT-frameErr',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         partitions=partitions,
                                                         x_axis_column=['Nr'],
                                                         filter_func=get_rt_change_type,
                                                         filter_func_args=filter_func_args,
                                                         special_calcs_func=get_mean_y_vs_x_it,
                                                         special_calcs_args=special_calcs_args,
                                                         calc_func=get_accum_corrs_sequs,
                                                         calc_func_args=calc_func_args,
                                                         compare_source=None,
                                                         fig_type='sharp plot',
                                                         use_marks=False,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=True)
            elif ev == 5:
                fig_title_pre_str = 'Values of R\\&t Differences of Scenes with Stable Stereo Poses ' \
                                    'for Comparison of '
                eval_columns = ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz']
                units = [('R_diffAll', '/\\textdegree'), ('R_diff_roll_deg', '/\\textdegree'),
                         ('R_diff_pitch_deg', '/\\textdegree'), ('R_diff_yaw_deg', '/\\textdegree'),
                         ('t_angDiff_deg', '/\\textdegree'), ('t_distDiff', ''), ('t_diff_tx', ''),
                         ('t_diff_ty', ''), ('t_diff_tz', '')]
                it_parameters = ['stereoRef']
                partitions = ['inlratMin', 'rt_change_type']
                filter_func_args = {'data_seperators': ['inlratMin',
                                                        'kpAccSd',
                                                        'depthDistr'],
                                    'filter_scene': ['crt', 'jrt']}
                calc_func_args = {'data_separators': ['depthDistr', 'kpAccSd', 'inlratMin']}
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': False,
                                      'fig_type': 'sharp plot',
                                      'used_x_axis': 'Nr',
                                      'cols_for_mean': ['inlratMin'],
                                      'func_name': 'comp_pars_RT_vs_FNr'}
                from usac_eval import get_best_comb_inlrat_1
                from robustness_eval import get_rt_change_type
                from usac_vs_autocalib_eval import get_accum_corrs_sequs, get_mean_y_vs_x_it
                ret += calcSatisticAndPlot_2D_partitions(data=data.copy(deep=True),
                                                         store_path=output_path,
                                                         tex_file_pre_str='plots_usacVsAuto_',
                                                         fig_title_pre_str=fig_title_pre_str,
                                                         eval_description_path='RT-frameErr',
                                                         eval_columns=eval_columns,
                                                         units=units,
                                                         it_parameters=it_parameters,
                                                         partitions=partitions,
                                                         x_axis_column=['Nr'],
                                                         filter_func=get_rt_change_type,
                                                         filter_func_args=filter_func_args,
                                                         special_calcs_func=get_mean_y_vs_x_it,
                                                         special_calcs_args=special_calcs_args,
                                                         calc_func=get_accum_corrs_sequs,
                                                         calc_func_args=calc_func_args,
                                                         compare_source=None,
                                                         fig_type='sharp plot',
                                                         use_marks=False,
                                                         ctrl_fig_size=True,
                                                         make_fig_index=True,
                                                         build_pdf=True,
                                                         figs_externalize=True)
            elif ev == 6:
                fig_title_pre_str = 'Execution Times on Scenes with Stable Stereo Poses ' \
                                    'for Comparison of '
                eval_columns = ['exec_time']
                units = []
                it_parameters = ['stereoRef']
                filter_func_args = {'data_seperators': ['inlratMin',
                                                        'kpAccSd',
                                                        'depthDistr'],
                                    'filter_scene': 'nv'}
                calc_func_args = {'data_separators': ['depthDistr', 'kpAccSd', 'inlratMin'],
                                  't_data_separators': ['inlratMin'],
                                  'addit_cols': ['inlratMin']}
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': False,
                                      'nr_target_kps': 1000,
                                      'func_name': 'comp_pars_min_time_vs_inlrat'}
                from usac_vs_autocalib_eval import filter_calc_t_all_rt_change_type, \
                    estimate_alg_time_fixed_kp, \
                    accum_corrs_sequs_time_model
                ret += calcFromFuncAndPlot_2D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_usacVsAuto_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='time',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              x_axis_column=['nrCorrs_GT'],
                                              filter_func=filter_calc_t_all_rt_change_type,
                                              filter_func_args=filter_func_args,
                                              special_calcs_func=estimate_alg_time_fixed_kp,
                                              special_calcs_args=special_calcs_args,
                                              calc_func=accum_corrs_sequs_time_model,
                                              calc_func_args=calc_func_args,
                                              compare_source=None,
                                              fig_type='smooth',
                                              use_marks=True,
                                              ctrl_fig_size=True,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=False)
            elif ev == 7:
                fig_title_pre_str = 'Execution Times on Scenes with Stable Stereo Poses ' \
                                    'for Comparison of '
                eval_columns = ['exec_time']
                units = []
                it_parameters = ['stereoRef']
                filter_func_args = {'data_seperators': ['inlratMin',
                                                        'kpAccSd',
                                                        'depthDistr'],
                                    'filter_scene': 'nv'}
                calc_func_args = {'data_separators': ['depthDistr', 'kpAccSd', 'inlratMin'],
                                  't_data_separators': ['inlratMin', 'kpAccSd'],
                                  'addit_cols': ['inlratMin', 'kpAccSd']}
                special_calcs_args = {'build_pdf': (True, True),
                                      'use_marks': False,
                                      'nr_target_kps': 1000,
                                      't_data_separators': ['inlratMin', 'kpAccSd'],
                                      'func_name': 'comp_pars_min_time_vs_inlrat',
                                      'res_par_name': 'usac_vs_autoc_inlRat_min_time'}
                from usac_vs_autocalib_eval import filter_calc_t_all_rt_change_type, \
                    estimate_alg_time_fixed_kp, \
                    accum_corrs_sequs_time_model
                ret += calcFromFuncAndPlot_3D(data=data.copy(deep=True),
                                              store_path=output_path,
                                              tex_file_pre_str='plots_usacVsAuto_',
                                              fig_title_pre_str=fig_title_pre_str,
                                              eval_description_path='time',
                                              eval_columns=eval_columns,
                                              units=units,
                                              it_parameters=it_parameters,
                                              xy_axis_columns=['nrCorrs_GT'],
                                              filter_func=filter_calc_t_all_rt_change_type,
                                              filter_func_args=filter_func_args,
                                              special_calcs_func=estimate_alg_time_fixed_kp,
                                              special_calcs_args=special_calcs_args,
                                              calc_func=accum_corrs_sequs_time_model,
                                              calc_func_args=calc_func_args,
                                              fig_type='surface',
                                              use_marks=True,
                                              ctrl_fig_size=False,
                                              make_fig_index=True,
                                              build_pdf=True,
                                              figs_externalize=True)
            elif ev == 8:
                fig_title_pre_str = 'Statistics on Execution Times for Comparison of '
                eval_columns = ['exec_time']
                units = [('exec_time', '/$\\mu s$')]
                it_parameters = ['stereoRef']
                filter_func_args = {'data_seperators': ['inlratMin',
                                                        'kpAccSd',
                                                        'depthDistr'],
                                    'filter_scene': 'nv'}
                calc_func_args = {'data_separators': ['depthDistr', 'kpAccSd', 'inlratMin']}
                from usac_vs_autocalib_eval import filter_calc_t_all_rt_change_type, \
                    get_accum_corrs_sequs
                ret += calcSatisticAndPlot_aggregate(data=data.copy(deep=True),
                                                     store_path=output_path,
                                                     tex_file_pre_str='plots_corrPool_',
                                                     fig_title_pre_str=fig_title_pre_str,
                                                     eval_description_path='time',
                                                     eval_columns=eval_columns,
                                                     units=units,
                                                     it_parameters=it_parameters,
                                                     pdfsplitentry=None,
                                                     filter_func=filter_calc_t_all_rt_change_type,
                                                     filter_func_args=filter_func_args,
                                                     special_calcs_func=None,
                                                     special_calcs_args=None,
                                                     calc_func=get_accum_corrs_sequs,
                                                     calc_func_args=calc_func_args,
                                                     compare_source=None,
                                                     fig_type='ybar',
                                                     use_marks=False,
                                                     ctrl_fig_size=True,
                                                     make_fig_index=True,
                                                     build_pdf=True,
                                                     figs_externalize=False)
            else:
                raise ValueError('Eval nr ' + ev + ' does not exist')
    else:
        raise ValueError('Test ' + test_name + ' does not exist')

    return ret


def get_compare_info(comp_pars, comp_path, test_name, test_r, eval_description_path, descr, repl_eval=None,
                     unique_path=None):
    if not comp_pars:
        raise ValueError('Parameter values for comparing refinement without kp aggregation missing')
    if not comp_path and unique_path:
        c_path = unique_path
        if not os.path.exists(c_path):
            raise ValueError('Specific compare directory ' + c_path + ' not found')
    else:
        c_path = os.path.join(comp_path, test_name)
        if not os.path.exists(c_path):
            raise ValueError('Specific test compare directory ' + c_path + ' not found')
        if test_r:
            c_path = os.path.join(c_path, str(test_r))
            if not os.path.exists(c_path):
                raise ValueError('Specific test nr compare directory ' + c_path + ' not found')
        c_path = os.path.join(c_path, 'evals')
        if not os.path.exists(c_path):
            raise ValueError('Specific \'evals\' compare directory ' + c_path + ' not found')
    compare_source = {'store_path': c_path,
                      'it_par_select': [a.split('-')[-1] for a in comp_pars],
                      'it_parameters': [a.split('-')[0] for a in comp_pars],
                      'eval_description_path': eval_description_path,
                      'cmp': descr,
                      'replace_evals': repl_eval
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
    parser.add_argument('--nrCPUs', type=int, required=False, default=4,
                        help='Number of CPU cores for parallel processing. If a negative value is provided, '
                             'the program tries to find the number of available CPUs on the system - if it fails, '
                             'the absolute value of nrCPUs is used. Default: 4')
    parser.add_argument('--message_path', type=str, required=False,
                        help='Storing path for text files containing error and normal mesages during '
                             'execution of evals')
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
                             '(same axis, data partitions, ...). If provided, also argument \'comp_pars_ev_nr\' must '
                             'be provided if argument \'eval_nr\' equals -1 [default] or a list of evaluation numbers '
                             'was given. The argument \'compare_pars\' must be provided as a list of strings '
                             '(e.g. --compare str1 str2 str3 ...). The list must contain the parameter-parameter '
                             'value pairs of the already performed evaluation for which the comparison should be performed '
                             '(like \'--compare_pars refineMethod_algorithm-PR_STEWENIUS '
                             'refineMethod_costFunction-PR_PSEUDOHUBER_WEIGHTS BART-extr_only\' for the evaluation '
                             'of refinement methods and bundle adjustment options).')
    parser.add_argument('--comp_pars_ev_nr', type=str, required=False, nargs='*', default=[],
                        help='If provided, argument \'compare_pars\' must also be provided. For every '
                             'string in argument \'compare_pars\' the evaluation number at which the comparison '
                             'should be performed must be provided.')
    parser.add_argument('--compare_path', type=str, required=False,
                        help='If provided, a different path is used for loading results for comparison. Otherwise, '
                             'the part from option --path is used. Results are only loaded, if option '
                             '--compare_pars is provided.')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise ValueError('Main directory not found')
    if args.nrCPUs > 72 or args.nrCPUs == 0:
        raise ValueError("Unable to use " + str(args.nrCPUs) + " CPU cores.")
    av_cpus = os.cpu_count()
    if av_cpus:
        if args.nrCPUs < 0:
            cpu_use = av_cpus
        elif args.nrCPUs > av_cpus:
            print('Demanded ' + str(args.nrCPUs) + ' but only ' + str(av_cpus) + ' CPUs are available. Using '
                  + str(av_cpus) + ' CPUs.')
            cpu_use = av_cpus
        else:
            cpu_use = args.nrCPUs
    elif args.nrCPUs < 0:
        print('Unable to determine # of CPUs. Using ' + str(abs(args.nrCPUs)) + ' CPUs.')
        cpu_use = abs(args.nrCPUs)
    else:
        cpu_use = args.nrCPUs
    if cpu_use > 1:
        if not args.message_path:
            raise ValueError("Path for storing stdout and stderr must be provided")
        if not os.path.exists(args.message_path):
            raise ValueError("Path for storing stdout and stderr does not exist")
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
        output_path = args.output_path
    else:
        output_path = os.path.join(load_path, 'evals')
        try:
            os.mkdir(output_path)
        except FileExistsError:
            raise ValueError('Directory ' + output_path + ' already exists')
    comp_path = None
    comp_pars = None
    if args.compare_pars:
        if (args.eval_nr[0] == -1 or len(args.eval_nr) > 1) and \
           (not args.comp_pars_ev_nr or len(args.compare_pars) != len(args.comp_pars_ev_nr)):
            raise ValueError('Both arguments \'compare_pars\' and \'comp_pars_ev_nr\' must be provided and contain the '
                             'same number of elements')
        elif (args.eval_nr[0] != -1 and len(args.eval_nr) == 1):
            comp_pars = {args.eval_nr[0]: args.compare_pars}
        else:
            comp_pars = {}
            for a in list(dict.fromkeys(args.comp_pars_ev_nr)):
                comp_pars[a] = []
            for a, b in zip(args.compare_pars, args.comp_pars_ev_nr):
                comp_pars[b].append(a)
        if args.compare_path:
            comp_path = args.compare_path
            if not os.path.exists(comp_path):
                raise ValueError('Specific main compare directory ' + comp_path + '  not found')
        else:
            comp_path = args.path

    return eval_test(load_path, output_path, test_name, args.test_nr, args.eval_nr, comp_path, comp_pars, cpu_use,
                     args.message_path)


if __name__ == "__main__":
    main()