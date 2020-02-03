"""
Main script file for executing the whole test procedure for testing the autocalibration SW
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np, logging
import communication as com
import evaluation_numbers as en
import pandas as pd

token = ''
use_sms = False
test_main = True
test_ret = [0, 0, 0]
test_raise = [False, False, False]


def start_testing(path, path_confs_out, skip_tests, skip_gen_sc_conf, skip_crt_sc,
                        use_cal_tests, use_evals, img_path, store_path_sequ, load_path, cpu_use,
                        exec_sequ, message_path, exec_cal, store_path_cal, compare_pars,
                        comp_pars_ev_nr, compare_path):
    main_tests = en.get_available_main_tests()
    scene_creation = en.get_available_sequences()
    if skip_tests:
        main_tests = [a for a in main_tests if not any([b == a for b in skip_tests])]
        scene_creation = [a for a in scene_creation if not any([b == a for b in skip_tests])]
        for i in skip_tests:
            use_cal_tests.pop(i, None)
            use_evals.pop(i, None)
    for mt in main_tests:
        # Make dir for messages
        message_path_new = os.path.join(message_path, mt)
        try:
            os.mkdir(message_path_new)
        except FileExistsError:
            pass
        # Generate configuration files
        if any([b == mt for b in scene_creation]) and \
                not (skip_gen_sc_conf and any([b == mt for b in skip_gen_sc_conf])):
            ret = gen_config_files(mt, path, path_confs_out, img_path, store_path_sequ, load_path)
            if ret:
                return ret
            print('Finished generating sequence configuration files for ' + mt)
        # Generate scenes and matches
        if any([b == mt for b in scene_creation]) and \
                not (skip_crt_sc and any([b == mt for b in skip_crt_sc])):
            pcol = get_confs_path(mt, -1, path, path_confs_out)
            for pco in pcol:
                ret = gen_scenes(mt, pco, exec_sequ, message_path_new, cpu_use)
                if ret:
                    return ret
            print('Finished generating scenes and matches for ' + mt)
        # Run autocalibration and evaluation
        if any([b == mt for b in use_cal_tests.keys()]):
            if use_cal_tests[mt] is None:
                test_nr_list = [None]
                test_nrs1 = test_nr_list
            else:
                test_nr_list = use_cal_tests[mt]
                if any([b == mt for b in use_evals.keys()]):
                    test_nrs1 = list(map(int, list(dict.fromkeys(list(map(int, use_evals[mt].keys())) + test_nr_list))))
                else:
                    test_nrs1 = test_nr_list
            for tn in test_nrs1:
                if tn in test_nr_list:
                    (conf_name, conf_nr) = en.get_autocalib_sequence_config_ref(mt, tn)
                    pco = get_confs_path(conf_name, conf_nr, path, path_confs_out)
                    # Run autocalibration
                    ret = start_autocalibration(mt, tn, pco, store_path_cal, exec_cal, message_path_new, cpu_use)
                    if ret:
                        return ret
                    print('Finished testing autocalibration for main test ', mt,
                          (' and test nr ' + str(tn) if tn else ''))
                # Run evaluation
                if any([b == mt for b in use_evals.keys()]) and \
                        (tn is None or any([b == str(tn) for b in use_evals[mt].keys()])):
                    if tn is None:
                        ev_nrs = use_evals[mt]['0']
                    else:
                        ev_nrs = use_evals[mt][str(tn)]
                    ret = start_eval(mt, tn, ev_nrs, store_path_cal, cpu_use, message_path_new,
                                     compare_pars, comp_pars_ev_nr, compare_path)
                    if ret:
                        return ret
                    print('Finished evaluation for main test ', mt, (' and test nr ' + str(tn) if tn else ''))
        elif any([b == mt for b in use_evals.keys()]):
            # Run evaluation
            for nr_key in use_evals[mt].keys():
                if nr_key == '0':
                    tn = None
                    ev_nrs = use_evals[mt]
                else:
                    tn = int(nr_key)
                    ev_nrs = use_evals[mt][nr_key]
                ret = start_eval(mt, tn, ev_nrs, store_path_cal, cpu_use, message_path_new,
                                 compare_pars, comp_pars_ev_nr, compare_path)
                if ret:
                    return ret
                print('Finished evaluation for main test ', mt, (' and test nr ' + str(tn) if tn else ''))
    send_message('Testing finished without errors')
    print('Testing finished without errors')
    return 0


def start_eval(test_name, test_nr, use_evals, store_path_cal, cpu_use, message_path,
               compare_pars, comp_pars_ev_nr, compare_path):
    pyfilepath = os.path.dirname(os.path.realpath(__file__))
    pyfilename = os.path.join(pyfilepath, 'eval_tests_main.py')
    cmdline = ['python', pyfilename, '--path', store_path_cal, '--nrCPUs', str(cpu_use),
               '--message_path', message_path, '--test_name', test_name]
    if test_nr:
        cmdline += ['--test_nr', str(test_nr)]
    cmdline.append('--eval_nr')
    cmdline += [str(a) for a in use_evals]
    if compare_path:
        cmdline += ['--compare_path', compare_path]
    if comp_pars_ev_nr is not None:
        if compare_pars is None:
            raise ValueError('Compare parameters must be provided.')
        cmdline += ['--comp_pars_ev_nr', comp_pars_ev_nr]
    if compare_pars is not None:
        if comp_pars_ev_nr is None:
            raise ValueError('Evaluation numbers where comparisons should be included must be provided.')
        cmdline += ['--compare_pars', compare_pars]
    elif en.check_if_eval_needs_compare_data(test_name, test_nr, use_evals):
        comp_pars_ev_nr, compare_pars = get_compare_data(test_name, test_nr, use_evals, store_path_cal)
        cmdline += ['--comp_pars_ev_nr'] + list(map(str, comp_pars_ev_nr)) + ['--compare_pars'] + compare_pars

    tout = len(use_evals) * 4 * 3600
    try:
        if test_main:
            ret = test_ret[2]
            if test_raise[2]:
                raise ValueError('Test in start_eval')
        else:
            ret = sp.run(cmdline, shell=False, stdout=sys.stdout, stderr=sys.stderr,
                         check=True, timeout=tout).returncode
    except sp.TimeoutExpired:
        logging.error('Timeout expired for evaluating results in main test ' +
                      test_name + (' with test nr ' + str(test_nr) if test_nr else ''), exc_info=True)
        ret = 98
    except Exception:
        logging.error('Evaluation failed. Main test ' + test_name +
                      (' with test nr ' + str(test_nr) if test_nr else ''), exc_info=True)
        ret = 99
    if ret:
        if ret < 98:
            logging.error('Evaluation failed. Main test ' + test_name +
                          (' with test nr ' + str(test_nr) if test_nr else ''))
        send_message('Evaluation failed. Main test ' + test_name +
                     (' with test nr ' + str(test_nr) if test_nr else ''))
        return ret

    # After evals are finished try to find optimal parameters
    try:
        ret = find_optimal_parameters(test_name, test_nr, store_path_cal)
        if ret:
            ret = 0
        else:
            ret = 1
            send_message('Finding optimal parameters failed. They have to be manually entered. Main test ' + test_name +
                         (' with test nr ' + str(test_nr) if test_nr else ''))
    except Exception:
        logging.error('Finding optimal parameters failed due to error in main script. Main test ' + test_name +
                      (' with test nr ' + str(test_nr) if test_nr else ''), exc_info=True)
        send_message('Finding optimal parameters failed due to error in main script. Main test ' + test_name +
                     (' with test nr ' + str(test_nr) if test_nr else ''))
        ret = 99
    return ret


def find_optimal_parameters(test_name, test_nr, store_path_cal):
    path_name = os.path.join(store_path_cal, test_name)
    if not os.path.exists(path_name):
        raise ValueError('Path ' + path_name + ' for loading evaluation results does not exist')
    func_info = en.get_opt_pars_func(test_name, test_nr)
    if func_info is None:
        return True
    paths = []
    for nr in func_info['test_nrs']:
        if nr is None:
            path_nr = path_name
        else:
            path_nr = os.path.join(path_name, str(nr))
            if not os.path.exists(path_nr):
                raise ValueError('Path ' + path_nr + ' for loading evaluation results does not exist')
        if test_main:
            evals_path = path_nr
        else:
            evals_path = os.path.join(path_nr, 'evals')
        if not os.path.exists(evals_path):
            raise ValueError('Unable to locate results path of evaluations. Cannot estimate optimal parameters.')
        paths.append(evals_path)
    pars = func_info['func'](paths, func_info['pars'])
    if pars is None:
        return False
    ret = True
    if isinstance(pars, list):
        ret = False
        tmp = {}
        for i in pars:
            if i is not None:
                tmp.update(i)
        if not tmp:
            return False
        pars = tmp
    from start_test_cases import write_parameters, convert_mult_autoc_pars_out_to_in
    pars_out = convert_mult_autoc_pars_out_to_in(pars)
    write_parameters(store_path_cal, pars_out)
    return ret


def get_compare_data(test_name, test_nr, use_evals, store_path_cal):
    pars_list = en.get_load_pars_for_comparison(test_name, test_nr)
    from start_test_cases import read_pars, convert_autoc_pars_in_to_out
    pars = read_pars(store_path_cal, pars_list)
    pars_out = []
    for i in pars.items():
        if i[1] is None:
            raise ValueError('Parameter ' + i[0] + ' for comparison couldnt be read from yaml file.')
        tmp = convert_autoc_pars_in_to_out(i[0], i[1])
        for j in tmp.items():
            pars_out.append(j[0] + '-' + str(j[1]))
    ev_nums = en.get_eval_nrs_for_cmp(test_name, test_nr)
    ev_nums1 = []
    for i in ev_nums:
        if i in use_evals:
            ev_nums1.append(i)
    ev = []
    for i in ev_nums1:
        ev += [str(i)] * len(pars_out)
    pars_out *= len(ev_nums1)
    return ev, pars_out


def start_autocalibration(test_name, test_nr, gen_dirs_config_f, output_path, executable, message_path, cpu_use):
    pyfilepath = os.path.dirname(os.path.realpath(__file__))
    pyfilename = os.path.join(pyfilepath, 'start_test_cases.py')
    cmdline = ['python', pyfilename, '--path', gen_dirs_config_f, '--nrCPUs', str(cpu_use),
               '--output_path', output_path, '--executable', executable, '--message_path', message_path,
               '--test_name', test_name]
    if test_nr:
        cmdline += ['--test_nr', str(test_nr)]
    tout = int(96768000 / cpu_use)# 1209600 possibilities * (150 + 50) / 2 frames * 0.8 seconds
    try:
        if test_main:
            ret = test_ret[1]
            if test_raise[1]:
                raise ValueError('Test in start_autocalibration')
        else:
            ret = sp.run(cmdline, shell=False, stdout=sys.stdout, stderr=sys.stderr,
                         check=True, timeout=tout).returncode
    except sp.TimeoutExpired:
        logging.error('Timeout expired for testing autocalibration in main test ' +
                      test_name + (' with test nr ' + str(test_nr) if test_nr else ''), exc_info=True)
        ret = 98
    except Exception:
        logging.error('Autocalibration failed. Main test ' + test_name +
                      (' with test nr ' + str(test_nr) if test_nr else ''), exc_info=True)
        ret = 99
    if ret:
        if ret < 98:
            logging.error('Autocalibration failed. Main test ' + test_name +
                          (' with test nr ' + str(test_nr) if test_nr else ''))
        send_message('Autocalibration failed. Main test ' + test_name +
                     (' with test nr ' + str(test_nr) if test_nr else ''))
    return ret


def gen_scenes(test_name, gen_dirs_config_f, executable, message_path, cpu_use):
    pyfilepath = os.path.dirname(os.path.realpath(__file__))
    pyfilename = os.path.join(pyfilepath, 'create_scenes.py')
    cmdline = ['python', pyfilename, '--path', gen_dirs_config_f, '--nrCPUs', str(cpu_use),
               '--executable', executable, '--message_path', message_path]
    tout = int(3024000 / cpu_use)
    try:
        if test_main:
            ret = test_ret[0]
            if test_raise[0]:
                raise ValueError('Test in gen_scenes')
        else:
            ret = sp.run(cmdline, shell=False, stdout=sys.stdout, stderr=sys.stderr,
                         check=True, timeout=tout).returncode
    except sp.TimeoutExpired:
        logging.error('Timeout expired for generating scenes and matches using directory ' +
                      gen_dirs_config_f + ' for ' + test_name, exc_info=True)
        ret = 98
    except Exception:
        logging.error('Failed to generate scenes and matches using directory ' +
                      gen_dirs_config_f + ' for ' + test_name, exc_info=True)
        ret = 99
    if ret:
        if ret < 98:
            logging.error('Failed to generate scenes and matches using directory ' +
                          gen_dirs_config_f + ' for ' + test_name)
        send_message('Failed to generate scenes and matches for ' + test_name)
    return ret


def gen_config_files(test_name, path_init_confs, path_confs_out, img_path, store_path_sequ, load_path):
    if not path_init_confs:
        raise ValueError('Path containing initial configuration files must be provided.')
    if not img_path:
        raise ValueError('Path containing images for creating matches must be provided.')
    if not store_path_sequ:
        raise ValueError('Path for storing scenes and matches must be provided.')
    pars = en.get_config_file_parameters(test_name)
    pyfilepath = os.path.dirname(os.path.realpath(__file__))
    pyfilename = os.path.join(pyfilepath, 'gen_mult_scene_configs.py')
    dir_names = en.get_mult_conf_dirs_per_test(test_name)
    ret = 3
    for dir_name in dir_names:
        sub_path = os.path.join(path_init_confs, dir_name)
        if not os.path.exists(sub_path):
            raise ValueError('Sub-path ' + sub_path + ' for loading initial config files not found')
        cmdline = ['python', pyfilename, '--path', sub_path,
                   '--img_path', img_path, '--store_path', store_path_sequ]
        if path_confs_out:
            pco = os.path.join(path_confs_out, dir_name)
            try:
                os.mkdir(pco)
            except FileExistsError:
                pass
            cmdline += ['--path_confs_out', pco]
        if load_path:
            cmdline += ['--load_path', load_path]
        for key, val in pars.items():
            if not isinstance(val, bool) or val:
                cmdline.append('--' + key)
            if isinstance(val, list):
                cmdline += ['%.2f' % a for a in val]
            elif not isinstance(val, bool):
                cmdline.append(str(val))
        try:
            ret = sp.run(cmdline, shell=False, stdout=sys.stdout, stderr=sys.stderr,
                         check=True, timeout=120).returncode
        except sp.TimeoutExpired:
            logging.error('Timeout expired for generating sequence configuration files in directory ' +
                          dir_name + ' for ' + test_name, exc_info=True)
            ret = 98
        except Exception:
            logging.error('Failed to generate sequence configuration files in directory ' +
                          dir_name + ' for ' + test_name, exc_info=True)
            ret = 99
        if ret:
            if ret < 98:
                logging.error('Failed to generate sequence configuration files in directory ' +
                              dir_name + ' for ' + test_name)
            send_message('Failed to generate sequence configuration files in directory ' +
                         dir_name + ' for ' + test_name)
            return ret
    return ret


def get_confs_path(test_name, test_nr=-1, path_confs_init=None, path_confs_out=None):
    if test_nr == -1:
        dir_name = en.get_mult_conf_dirs_per_test(test_name)
    else:
        dir_name = en.get_config_file_dir(test_name, test_nr)
    if not path_confs_out and not path_confs_init:
        raise ValueError('Main path to initial configuration files is missing.')
    elif path_confs_out:
        if isinstance(dir_name, list):
            path = []
            for dn in dir_name:
                path.append(os.path.join(path_confs_out, dn))
                if not os.path.exists(path[-1]):
                    raise ValueError('Path with configuration files does not exist')
        else:
            path = os.path.join(path_confs_out, dir_name)
            if not os.path.exists(path):
                raise ValueError('Path with configuration files does not exist')
        return path
    warnings.warn('Are you sure you have not chosen a specific path for storing configuration files?', UserWarning)
    if isinstance(dir_name, list):
        path = []
        for dn in dir_name:
            path.append(os.path.join(path_confs_init, dn))
            if not os.path.exists(path[-1]):
                raise ValueError('Path with configuration files does not exist')
    else:
        path = os.path.join(path_confs_init, dir_name)
        if not os.path.exists(path):
            raise ValueError('Path with configuration files does not exist')
    return path


def get_skip_use_evals(strings):
    main_test_names, sub_test_numbers, sub_sub_test_nr = en.get_available_evals()
    if not strings:
        use_evals = dict.fromkeys(main_test_names)
        for i, mn in enumerate(main_test_names):
            if sub_test_numbers[i] == 1:
                use_evals[mn] = {'0': sub_sub_test_nr[i][0]}
            else:
                use_evals[mn] = {'1': sub_sub_test_nr[i][0]}
                for j in range(1, sub_test_numbers[i]):
                    use_evals[mn][str(j + 1)] = sub_sub_test_nr[i][j]
        return use_evals
    use_skip_evals = {}
    use_evals = {}
    is_skip = False
    if strings[0] == 'use+' or strings[0] == 'skip+':
        if strings[1] not in main_test_names:
            raise ValueError('Invalid order in given list of argument skip_use_eval_name_nr')
        else:
            use_skip_evals[strings[1]] = {}
        if len(strings) == 2:
            test_idx = main_test_names.index(strings[1])
            use_skip_evals[strings[1]] = {}
            for a in range(1, sub_test_numbers[test_idx] + 1):
                use_skip_evals[strings[1]][str(a)] = [-1]
        else:
            current_main = strings[1]
            current_test = ''
            if not any([a == current_main for a in main_test_names]):
                raise ValueError('Specified main evaluation name ' + current_main + ' not found')
            last_main = True
            last_test = False
            last_eval = False
            test_idx = main_test_names.index(current_main)
            for i in range(2, len(strings)):
                try:
                    nr = int(strings[i])
                    if nr < 0 and not last_test and not last_eval and sub_test_numbers[test_idx] > 1:
                        raise ValueError('A test number must be specified before eval numbers')
                    elif nr < 0 and not last_test and not last_eval and not last_main:
                        raise ValueError('Undefined behaviour during parsing of evaluations to execute')
                    elif nr < 0 and not last_test and not last_eval and last_main:
                        if use_skip_evals[current_main]:
                            raise ValueError(current_main + ' is used 2 times')
                        nr = abs(nr)
                        if nr not in sub_sub_test_nr[test_idx][0]:
                            raise ValueError('Eval nr ' + str(nr) +
                                             ' not available for main evaluation ' + current_main)
                        use_skip_evals[current_main] = {str(0): [nr]}
                        last_eval = True
                        last_main = False
                        current_test = '0'
                    elif nr < 0 and (last_test or last_eval):
                        nr = abs(nr)
                        eval_idx = int(current_test)
                        if sub_test_numbers[test_idx] > 1:
                            eval_idx -= 1
                        if nr not in sub_sub_test_nr[test_idx][eval_idx]:
                            raise ValueError('Eval nr ' + str(nr) +
                                             ' not available for main evaluation ' + current_main +
                                             ' and test nr ' + current_test)
                        use_skip_evals[current_main][current_test].append(nr)
                        last_eval = True
                        last_test = False
                        last_main = False
                    elif nr >= 0 and last_main:
                        if nr == 0 and sub_test_numbers[test_idx] > 1:
                            raise ValueError('Invalid test number (0) for main eval ' + current_main)
                        elif nr > sub_test_numbers[test_idx]:
                            raise ValueError('Test number ' + str(nr) + ' out of range for main eval ' + current_main)
                        if nr <= 1 and sub_test_numbers[test_idx] == 1:
                            current_test = '0'
                        else:
                            current_test = str(nr)
                        last_main = False
                        last_test = True
                        use_skip_evals[current_main] = {current_test: []}
                    elif nr >= 0 and last_test:
                        if nr == 0 and sub_test_numbers[test_idx] > 1:
                            raise ValueError('Invalid test number (0) for main eval ' + current_main)
                        elif nr > sub_test_numbers[test_idx]:
                            raise ValueError('Test number ' + str(nr) + ' out of range for main eval ' + current_main)
                        elif nr <= 1 and sub_test_numbers[test_idx] == 1:
                            raise ValueError('Main eval ' + current_main + ' does not support multiple test numbers')
                        eval_idx = int(current_test)
                        if sub_test_numbers[test_idx] > 1:
                            eval_idx -= 1
                        use_skip_evals[current_main][current_test] = sub_sub_test_nr[test_idx][eval_idx]
                        current_test = str(nr)
                        if use_skip_evals[current_main]:
                            if any([current_test == a for a in use_skip_evals[current_main].keys()]):
                                raise ValueError('Test ' + current_test + ' of ' + current_main + ' is used 2 times')
                            use_skip_evals[current_main][current_test] = []
                        else:
                            use_skip_evals[current_main] = {current_test: []}
                        last_main = False
                    elif nr >= 0 and last_eval:
                        if nr == 0 and sub_test_numbers[test_idx] > 1:
                            raise ValueError('Invalid test number (0) for main eval ' + current_main)
                        elif nr > sub_test_numbers[test_idx]:
                            raise ValueError('Test number ' + str(nr) + ' out of range for main eval ' + current_main)
                        elif nr <= 1 and sub_test_numbers[test_idx] == 1:
                            raise ValueError('Main eval ' + current_main + ' does not support multiple test numbers')
                        last_eval = False
                        last_test = True
                        current_test = str(nr)
                        if any([current_test == a for a in use_skip_evals[current_main].keys()]):
                            raise ValueError('Test ' + current_test + ' of ' + current_main + ' is used 2 times')
                        use_skip_evals[current_main][current_test] = []
                        last_main = False
                    else:
                        raise ValueError('Undefined behaviour during parsing of evaluations to execute')
                except ValueError:
                    if any([a == strings[i] for a in main_test_names]):
                        if any([strings[i] == a for a in use_skip_evals.keys()]):
                            raise ValueError('Main eval ' + strings[i] + ' is used multiple times')
                        if last_test:
                            eval_idx = int(current_test)
                            if sub_test_numbers[test_idx] > 1:
                                eval_idx -= 1
                            use_skip_evals[current_main][current_test] = sub_sub_test_nr[test_idx][eval_idx]
                            last_test = False
                        elif last_main:
                            if sub_test_numbers[test_idx] == 1:
                                use_skip_evals[current_main] = {'0': sub_sub_test_nr[test_idx][0]}
                            else:
                                use_skip_evals[current_main] = {'1': sub_sub_test_nr[test_idx][0]}
                                for j in range(1, sub_test_numbers[test_idx]):
                                    use_skip_evals[current_main][str(j + 1)] = sub_sub_test_nr[test_idx][j]
                        current_main = strings[i]
                        test_idx = main_test_names.index(current_main)
                        use_skip_evals[current_main] = {}
                        last_main = True
                        last_eval = False
                    elif strings[i][0] == 'r':
                        tmp = strings[i][1:].split('-')
                        if len(tmp) != 2:
                            raise ValueError(strings[i] + ' is no range')
                        else:
                            try:
                                e1 = int(tmp[0])
                                e2 = int(tmp[1])
                            except ValueError:
                                raise ValueError(strings[i] + ' is no range')
                            ev_list = list(range(e1, (e2 + 1)))
                            if not last_test and not last_eval and sub_test_numbers[test_idx] > 1:
                                raise ValueError('A test number must be specified before eval numbers')
                            elif not last_test and not last_eval and not last_main:
                                raise ValueError('Undefined behaviour during parsing of evaluations to execute')
                            elif not last_test and not last_eval and last_main:
                                if e1 not in sub_sub_test_nr[test_idx][0] or \
                                        e2 not in sub_sub_test_nr[test_idx][0]:
                                    raise ValueError('Eval nr range ' + strings[i][1:] +
                                                     ' not available for main evaluation ' + current_main)
                                use_skip_evals[current_main] = {str(0): ev_list}
                                last_main = False
                                current_test = '0'
                            elif last_test or last_eval:
                                eval_idx = int(current_test)
                                if sub_test_numbers[test_idx] > 1:
                                    eval_idx -= 1
                                if e1 not in sub_sub_test_nr[test_idx][eval_idx] or \
                                        e2 not in sub_sub_test_nr[test_idx][eval_idx]:
                                    raise ValueError('Eval nr range ' + strings[i][1:] +
                                                     ' not available for main evaluation ' + current_main +
                                                     ' and test nr ' + current_test)
                                use_skip_evals[current_main][current_test] += ev_list
                                last_test = False
                                last_main = False
                            else:
                                raise ValueError('Undefined behaviour during parsing of evaluations to execute')
                            last_eval = True
                    else:
                        raise ValueError('Specified main evaluation name ' + strings[i] + ' not found')
            if last_main:
                if sub_test_numbers[test_idx] == 1:
                    use_skip_evals[current_main] = {'0': sub_sub_test_nr[test_idx][0]}
                else:
                    use_skip_evals[current_main] = {'1': sub_sub_test_nr[test_idx][0]}
                    for j in range(1, sub_test_numbers[test_idx]):
                        use_skip_evals[current_main][str(j + 1)] = sub_sub_test_nr[test_idx][j]
            elif last_test:
                eval_idx = int(current_test)
                if sub_test_numbers[test_idx] > 1:
                    eval_idx -= 1
                use_skip_evals[current_main][current_test] = sub_sub_test_nr[test_idx][eval_idx]
            elif not last_eval:
                raise ValueError('Undefined behaviour during parsing of evaluations to execute')
        if strings[0] == 'skip+':
            is_skip = True
        else:
            use_evals = use_skip_evals

    elif strings[0] == 'use':
        for i in range(1, len(strings)):
            current_main = strings[i]
            if any([a == current_main for a in main_test_names]):
                test_idx = main_test_names.index(current_main)
                if sub_test_numbers[test_idx] == 1:
                    use_evals[current_main] = {'0': sub_sub_test_nr[test_idx][0]}
                else:
                    use_evals[current_main] = {'1': sub_sub_test_nr[test_idx][0]}
                    for j in range(1, sub_test_numbers[test_idx]):
                        use_evals[current_main][str(j + 1)] = sub_sub_test_nr[test_idx][j]
            else:
                raise ValueError('Specified main evaluation name ' + current_main + ' not found')
    else:
        is_skip = True
        for cm in strings:
            if any([a == cm for a in main_test_names]):
                test_idx = main_test_names.index(cm)
                if sub_test_numbers[test_idx] == 1:
                    use_skip_evals[cm] = {'0': sub_sub_test_nr[test_idx][0]}
                else:
                    use_skip_evals[cm] = {'1': sub_sub_test_nr[test_idx][0]}
                    for j in range(1, sub_test_numbers[test_idx]):
                        use_skip_evals[cm][str(j + 1)] = sub_sub_test_nr[test_idx][j]
            else:
                raise ValueError('Specified main evaluation name ' + cm + ' not found')
    if is_skip:
        for i, cm in enumerate(main_test_names):
            main_used = False
            if any([a == cm for a in use_skip_evals.keys()]):
                if sub_test_numbers[i] == 1:
                    test_used = False
                    for j in sub_sub_test_nr[i][0]:
                        if j not in use_skip_evals[cm]['0']:
                            if test_used:
                                use_evals[cm]['0'].append(j)
                            else:
                                use_evals[cm] = {'0': [j]}
                                test_used = True
                else:
                    for tn in range(0, sub_test_numbers[i]):
                        test_used = False
                        if any([a == str(tn + 1) for a in use_skip_evals[cm].keys()]):
                            for j in sub_sub_test_nr[i][tn]:
                                if j not in use_skip_evals[cm][str(tn + 1)]:
                                    if test_used:
                                        use_evals[cm][str(tn + 1)].append(j)
                                    else:
                                        if main_used:
                                            use_evals[cm][str(tn + 1)] = [j]
                                        else:
                                            use_evals[cm] = {str(tn + 1): [j]}
                                            main_used = True
                                        test_used = True
                        else:
                            if main_used:
                                use_evals[cm][str(tn + 1)] = sub_sub_test_nr[i][tn]
                            else:
                                use_evals[cm] = {str(tn + 1): sub_sub_test_nr[i][tn]}
                                main_used = True
            else:
                if sub_test_numbers[i] == 1:
                    use_evals[cm] = {'0': sub_sub_test_nr[i][0]}
                else:
                    use_evals[cm] = {'1': sub_sub_test_nr[i][0]}
                    for tn in range(1, sub_test_numbers[i]):
                        use_evals[cm][str(tn + 1)] = sub_sub_test_nr[i][tn]
    return use_evals


def get_skip_use_cal_tests(strings):
    main_test_names, sub_test_numbers = en.get_available_tests()
    if not strings:
        use_tests = dict.fromkeys(main_test_names)
        for i, mn in enumerate(main_test_names):
            if sub_test_numbers[i] == 1:
                use_tests[mn] = None
            else:
                use_tests[mn] = list(range(1, (sub_test_numbers[i] + 1)))
        return use_tests
    use_skip_tests = {}
    use_tests = {}
    is_skip = False
    if strings[0] == 'use+' or strings[0] == 'skip+':
        if strings[1] not in main_test_names:
            raise ValueError('Invalid order in given list of argument skip_use_eval_name_nr')
        else:
            use_skip_tests[strings[1]] = {}
        if len(strings) == 2:
            test_idx = main_test_names.index(strings[1])
            use_skip_tests[strings[1]] = list(range(1, (sub_test_numbers[test_idx] + 1)))
        else:
            current_main = strings[1]
            current_test = ''
            if not any([a == current_main for a in main_test_names]):
                raise ValueError('Specified main evaluation name ' + current_main + ' not found')
            last_main = True
            last_test = False
            test_idx = main_test_names.index(current_main)
            for i in range(2, len(strings)):
                try:
                    nr = int(strings[i])
                    if nr < 0:
                        raise ValueError('A test number must be positive')
                    elif nr >= 0 and last_main:
                        if nr == 0 and sub_test_numbers[test_idx] > 1:
                            raise ValueError('Invalid test number (0) for main eval ' + current_main)
                        elif nr > sub_test_numbers[test_idx]:
                            raise ValueError('Test number ' + str(nr) + ' out of range for main eval ' + current_main)
                        if nr <= 1 and sub_test_numbers[test_idx] == 1:
                            use_skip_tests[current_main] = None
                        else:
                            use_skip_tests[current_main] = [nr]
                        last_main = False
                        last_test = True
                    elif nr >= 0 and last_test:
                        if nr == 0 and sub_test_numbers[test_idx] > 1:
                            raise ValueError('Invalid test number (0) for main eval ' + current_main)
                        elif nr > sub_test_numbers[test_idx]:
                            raise ValueError('Test number ' + str(nr) + ' out of range for main eval ' + current_main)
                        elif nr <= 1 and sub_test_numbers[test_idx] == 1:
                            raise ValueError('Main eval ' + current_main + ' does not support multiple test numbers')
                        use_skip_tests[current_main].append(nr)
                        last_main = False
                    else:
                        raise ValueError('Undefined behaviour during parsing of evaluations to execute')
                except ValueError:
                    if any([a == strings[i] for a in main_test_names]):
                        if any([strings[i] == a for a in use_skip_tests.keys()]):
                            raise ValueError('Main eval ' + strings[i] + ' is used multiple times')
                        if last_main:
                            if sub_test_numbers[test_idx] == 1:
                                use_skip_tests[current_main] = None
                            else:
                                use_skip_tests[current_main] = list(range(1, (sub_test_numbers[test_idx] + 1)))
                        current_main = strings[i]
                        test_idx = main_test_names.index(current_main)
                        use_skip_tests[current_main] = []
                        last_main = True
                    else:
                        raise ValueError('Specified main evaluation name ' + strings[i] + ' not found')
            if last_main:
                if sub_test_numbers[test_idx] == 1:
                    use_skip_tests[current_main] = None
                else:
                    use_skip_tests[current_main] = list(range(1, (sub_test_numbers[test_idx] + 1)))
        if strings[0] == 'skip+':
            is_skip = True
        else:
            use_tests = use_skip_tests
    elif strings[0] == 'use':
        for i in range(1, len(strings)):
            current_main = strings[i]
            if any([a == current_main for a in main_test_names]):
                test_idx = main_test_names.index(current_main)
                if sub_test_numbers[test_idx] == 1:
                    use_tests[current_main] = None
                else:
                    use_tests[current_main] = list(range(1, (sub_test_numbers[test_idx] + 1)))
            else:
                raise ValueError('Specified main evaluation name ' + current_main + ' not found')
    else:
        is_skip = True
        for cm in strings:
            if any([a == cm for a in main_test_names]):
                test_idx = main_test_names.index(cm)
                if sub_test_numbers[test_idx] == 1:
                    use_skip_tests[cm] = None
                else:
                    use_skip_tests[cm] = list(range(1, (sub_test_numbers[test_idx] + 1)))
            else:
                raise ValueError('Specified main evaluation name ' + cm + ' not found')
    if is_skip:
        for i, cm in enumerate(main_test_names):
            main_used = False
            if any([a == cm for a in use_skip_tests.keys()]):
                if sub_test_numbers[i] != 1:
                    for tn in range(1, (sub_test_numbers[i] + 1)):
                        if tn not in use_skip_tests[cm]:
                            if main_used:
                                use_tests[cm].append(tn)
                            else:
                                use_tests[cm] = [tn]
                                main_used = True
            else:
                if sub_test_numbers[i] == 1:
                    use_tests[cm] = None
                else:
                    use_tests[cm] = list(range(1, (sub_test_numbers[i] + 1)))
    return use_tests


def enable_messaging():
    global token, use_sms
    use_sms, token = com.decrypt_token()


def send_message(text):
    if use_sms:
        try:
            com.send_sms(text, token)
        except Exception:
            logging.error('Failed to send SMS', exc_info=True)
            warnings.warn('Unable to send messages!', UserWarning)


def enable_logging(path):
    base = 'error_log_main_level_%03d'
    excmess = os.path.join(path, (base % 1) + '.txt')
    cnt = 2
    while os.path.exists(excmess):
        excmess = os.path.join(path, (base % cnt) + '.txt')
        cnt += 1
    logging.basicConfig(filename=excmess, level=logging.DEBUG)


def retry_scenes_gen_dir(filename, exec_sequ, message_path, cpu_use):
    pyfilepath = os.path.dirname(os.path.realpath(__file__))
    pyfilename = os.path.join(pyfilepath, 'create_scenes.py')
    cmdline = ['python', pyfilename, '--retry_dirs_file', filename,
               '--nrCPUs', str(cpu_use), '--executable', exec_sequ, '--message_path', message_path]
    try:
        ret = sp.run(cmdline, shell=False, stdout=sys.stdout, stderr=sys.stderr,
                     check=True, timeout=18000).returncode
    except sp.TimeoutExpired:
        logging.error('Timeout expired for generating sequences.', exc_info=True)
        ret = 98
    except Exception:
        logging.error('Failed to generate scene and/or matches during retry', exc_info=True)
        ret = 99
    if ret:
        if ret < 98:
            logging.error('Retrying generation of sequences based on all configuration files in directories failed.')
        send_message('Retrying generation of sequences based on all configuration files in directories failed.')
    return ret


def retry_scenes_gen_cmds(filename, message_path, cpu_use):
    pyfilepath = os.path.dirname(os.path.realpath(__file__))
    pyfilename = os.path.join(pyfilepath, 'create_scenes.py')
    cmdline = ['python', pyfilename, '--retry_cmds_file', filename,
               '--nrCPUs', str(cpu_use), '--message_path', message_path]
    try:
        ret = sp.run(cmdline, shell=False, stdout=sys.stdout, stderr=sys.stderr,
                     check=True, timeout=18000).returncode
    except sp.TimeoutExpired:
        logging.error('Timeout expired for generating sequences.', exc_info=True)
        ret = 98
    except Exception:
        logging.error('Failed to generate scene and/or matches during retry', exc_info=True)
        ret = 99
    if ret:
        if ret < 98:
            logging.error('Retrying generation of sequences based on single command lines failed.')
        send_message('Retrying generation of sequences based on single command lines failed.')
    return ret


def retry_autocalibration(filename, nrCall, store_path, exec_cal, message_path, cpu_use):
    pyfilepath = os.path.dirname(os.path.realpath(__file__))
    pyfilename = os.path.join(pyfilepath, 'retry_test_cases.py')
    cmdline = ['python', pyfilename, '--csv_file', filename, '--nrCPUs', str(cpu_use),
               '--message_path', message_path, '--nrCall', str(nrCall), '--output_path', store_path,
               '--executable', exec_cal]
    cf = pd.read_csv(filename, delimiter=';')
    if cf.empty:
        raise ValueError("File " + filename + " is empty.")
    t_out = int(cf.shape[0]) * 200 * 15
    try:
        ret = sp.run(cmdline, shell=False, stdout=sys.stdout, stderr=sys.stderr,
                     check=True, timeout=t_out).returncode
    except sp.TimeoutExpired:
        logging.error('Timeout expired for testing autocalibration during retry.', exc_info=True)
        ret = 98
    except Exception:
        logging.error('Failed to test autocalibration during retry', exc_info=True)
        ret = 99
    if ret:
        if ret < 98:
            logging.error('Failed to test autocalibration during retry')
        send_message('Failed to test autocalibration during retry.')
    return ret


def main():
    parser = argparse.ArgumentParser(description='Main script file for executing the whole test procedure for '
                                                 'testing the autocalibration SW')
    parser.add_argument('--path', type=str, required=False,
                        help='Directory holding directories with template configuration files')
    parser.add_argument('--path_confs_out', type=str, required=False,
                        help='Optional directory for writing configuration files. If not available, '
                             'the directory is derived from argument \'path\' if '
                             'option \'complete_res_path\' is not provided.')
    parser.add_argument('--skip_tests', type=str, nargs='+', required=False,
                        help='List of test names that should be completely skipped. '
                             'Possible tests: usac-testing, usac_vs_ransac, refinement_ba, vfc_gms_sof, '
                             'refinement_ba_stereo, correspondence_pool, robustness, usac_vs_autocalib; '
                             'Format: test1 test2 ...')
    parser.add_argument('--skip_gen_sc_conf', type=str, nargs='+', required=False,
                        help='List of test names for which the generation of configuration files out of '
                             'initial configuration files should be skipped as they are already available. '
                             'Possible tests: usac-testing, correspondence_pool, robustness, usac_vs_autocalib; '
                             'Format: test1 test2 ...')
    parser.add_argument('--skip_crt_sc', type=str, nargs='+', required=False,
                        help='List of test names for which the creation of scenes '
                             'should be skipped as they are already available. '
                             'Possible tests: usac-testing, correspondence_pool, robustness, usac_vs_autocalib; '
                             'Format: test1 test2 ...')
    parser.add_argument('--crt_sc_dirs_file', type=str, required=False,
                        help='Optional (only used when scene creation process failed for entire directory/ies): '
                             'File holding directory names which include configuration files to generate '
                             'multiple scenes.')
    parser.add_argument('--crt_sc_cmds_file', type=str, required=False,
                        help='Optional (only used when scene creation process failed for a few scenes): '
                             'File holding command lines to generate scenes.')
    parser.add_argument('--skip_use_test_name_nr', type=str, nargs='+', required=False,
                        help='List of test names for which testing the autocalibration SW '
                             'should be skipped as they were already tested. If the first element of the list '
                             'equals \'use\', the given tests are not skipped but are the only ones that are executed. '
                             'If the first element of the list equals \'use+\', pairs of test names and multiple '
                             'test numbers can be specified to execute specific sub-tests (numbers) within main tests '
                             '(name) and , e.g.: use+ correspondence_pool 2 robustness 3 5 6; If the first element of '
                             'the list equals \'skip+\', pairs of test names and multiple test numbers can be '
                             'specified which should be skipped, e.g.: skip+ correspondence_pool 2 robustness 3 5 6; '
                             'Possible tests: usac-testing, usac_vs_ransac, refinement_ba, vfc_gms_sof, '
                             'refinement_ba_stereo, correspondence_pool, robustness, usac_vs_autocalib')
    parser.add_argument('--skip_use_eval_name_nr', type=str, nargs='+', required=False,
                        help='List of evaluation names for which evaluating results '
                             'should be skipped as they were already performed. If the first element of the list '
                             'equals \'use\', the given evals are not skipped but are the only ones that are executed. '
                             'If the first element of the list equals \'use+\', pairs of eval names, multiple '
                             'test numbers, and eval numbers can be specified to run specific eval numbers '
                             '(negative numbers) within test numbers (positive numbers) in main evaluations (name). '
                             'If no eval nr is specified, all available are executed. Also ranges can be given for '
                             'eval numbers only using an \'r\' in front of the range (e.g. r3-4). e.g. for performing '
                             'evaluations on correspondence_pool, test number 2, eval numbers 11, and 13-14 in '
                             'addition to evaluations on robustness, test number 3 with evals 12-14 and test numbers 5 '
                             'and 6 with all available evaluations, use: '
                             'e.g.: use+ correspondence_pool 2 -11 r13-14 robustness 3 r12-14 5 6; The same syntax can '
                             'be used for skipping evaluations using \'skip+\' as first element in the list.'
                             'the list equals \'skip+\', pairs of eval names and multiple eval numbers can be '
                             'specified which should be skipped, e.g.: skip+ correspondence_pool 2 robustness 3 5 6; '
                             'Possible tests: usac-testing, usac_vs_ransac, refinement_ba, vfc_gms_sof, '
                             'refinement_ba_stereo, correspondence_pool, robustness, usac_vs_autocalib')
    parser.add_argument('--img_path', type=str, required=False,
                        help='Path to images')
    parser.add_argument('--store_path_sequ', type=str, required=False,
                        help='Storing path for generated scenes and matches')
    parser.add_argument('--load_path', type=str, required=False,
                        help='Optional loading path for generated scenes and matches. '
                             'If not provided, store_path is used.')
    parser.add_argument('--nrCPUs', type=int, required=False, default=-16,
                        help='Number of CPU cores for parallel processing. If a negative value is provided, '
                             'the program tries to find the number of available CPUs on the system - if it fails, '
                             'the absolute value of nrCPUs is used. Default: -16')
    parser.add_argument('--exec_sequ', type=str, required=False,
                        help='Executable of the application generating the sequences')
    parser.add_argument('--message_path', type=str, required=False,
                        help='Storing path for text files containing error and normal messages')
    parser.add_argument('--exec_cal', type=str, required=False,
                        help='Executable of the autocalibration SW')
    parser.add_argument('--store_path_cal', type=str, required=False,
                        help='Main output path for results of the autocalibration. This directory is also used for '
                             'loading and storing the YML file with found optimal parameters. For every different '
                             'test a new directory with the name of option test_name is created. '
                             'Within this directory another directory is created with the name of option test_nr.')
    parser.add_argument('--cal_retry_file', type=str, required=False,
                        help='Path and filename of \'commands_and_parameters_unsuccessful_$.csv\' which holds commands '
                             'and parameters of failed test runs of the autocalibration SW. If used, parameter '
                             'cal_retry_nrCall must also be set')
    parser.add_argument('--cal_retry_nrCall', type=int, required=False,
                        help='Number which specifies, how often these test cases where already called before. The '
                             'smallest possible value is 1. There shouldn\'t be a csv file '
                             '\'commands_and_parameters_unsuccessful_$.csv\' with an equal or higher value of $ in '
                             'the same folder.')
    parser.add_argument('--compare_pars', type=str, required=False, nargs='*', default=None,
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
    parser.add_argument('--comp_pars_ev_nr', type=str, required=False, nargs='*', default=None,
                        help='If provided, argument \'compare_pars\' must also be provided. For every '
                             'string in argument \'compare_pars\' the evaluation number at which the comparison '
                             'should be performed must be provided.')
    parser.add_argument('--compare_path', type=str, required=False,
                        help='If provided, a different path is used for loading results for comparison. Otherwise, '
                             'the part from option --store_path_cal is used. Results are only loaded, if option '
                             '--compare_pars is provided.')
    parser.add_argument('--complete_res_path', type=str, required=False, default='default',
                        help='If provided or a folder \'results\' exists in the parent directory (../), '
                             'the full path structure for storing data is generated at the given '
                             'location except for parameters that were explicitely provided. Moreover, the input path '
                             'for initial configuration files is expected to be in \'Config_Files\' within the '
                             'directory holding this python file except option \'path\' is provided. '
                             'The input images should be located in a folder called \'images\' one folder level up '
                             'compared to the directory holding this python file (../images/). The latter also holds '
                             'for the executable generating sequences which should be located in '
                             '\'../generateVirtualSequence/build/\' and should be named '
                             '\'virtualSequenceLib-CMD-interface\'. The executable of the autocalibration SW should '
                             'similarly be located at \'../matchinglib_poselib/build/\' and called '
                             '\'noMatch_poselib-test\'.')
    args = parser.parse_args()
    if args.path and not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' holding directories with template scene '
                                                    'configuration files does not exist')
    if args.path_confs_out and not os.path.exists(args.path_confs_out):
        raise ValueError('Directory ' + args.path_confs_out + ' for storing config files does not exist')
    main_test_names = en.get_available_main_tests()
    if args.skip_tests:
        for i in args.skip_tests:
            if i not in main_test_names:
                raise ValueError('Cannot skip test ' + i + ' as it does not exist.')
    scenes_test_names = en.get_available_sequences()
    if args.skip_gen_sc_conf:
        for i in args.skip_gen_sc_conf:
            if i not in scenes_test_names:
                raise ValueError('Cannot skip generation of configuration files for '
                                 'scenes with test name ' + i + ' as it does not exist.')
    if args.skip_crt_sc:
        for i in args.skip_crt_sc:
            if i not in scenes_test_names:
                raise ValueError('Cannot skip creation of scenes with test name ' + i + ' as it does not exist.')
    if args.crt_sc_dirs_file and not os.path.exists(args.crt_sc_dirs_file):
        raise ValueError('File ' + args.crt_sc_dirs_file + ' does not exist.')
    elif args.crt_sc_dirs_file and not args.exec_sequ:
        raise ValueError('Executable for generating scenes must be provided')
    if args.crt_sc_cmds_file and not os.path.exists(args.crt_sc_cmds_file):
        raise ValueError('File ' + args.crt_sc_cmds_file + ' does not exist.')
    if args.img_path and not os.path.exists(args.img_path):
        raise ValueError("Image path does not exist")
    if args.store_path_sequ and not os.path.exists(args.store_path_sequ):
        raise ValueError("Path for storing sequences does not exist")
    if args.store_path_sequ and len(os.listdir(args.store_path_sequ)) != 0 and \
            not (args.crt_sc_dirs_file or args.crt_sc_cmds_file):
        raise ValueError("Path for storing sequences is not empty")
    if args.load_path:
        if not os.path.exists(args.load_path):
            raise ValueError("Path for loading sequences does not exist")
    if args.complete_res_path and args.complete_res_path == 'default':
        pyfilepath = os.path.dirname(os.path.realpath(__file__))
        parent = os.path.dirname(pyfilepath)
        res_folder = os.path.join(parent, 'results')
        if os.path.exists(res_folder):
            args.complete_res_path = res_folder
        else:
            args.complete_res_path = None
    if not args.complete_res_path and (not args.message_path or not os.path.exists(args.message_path)):
        raise ValueError("Path for storing stdout and stderr does not exist")
    if args.exec_sequ:
        if not os.path.isfile(args.exec_sequ):
            raise ValueError('Executable ' + args.exec_sequ + ' for generating scenes does not exist')
        elif not os.access(args.exec_sequ, os.X_OK):
            raise ValueError('Unable to execute ' + args.exec_sequ)
    if args.exec_cal:
        if not os.path.isfile(args.exec_cal):
            raise ValueError('Executable ' + args.exec_cal + ' of autocalibration does not exist')
        elif not os.access(args.exec_cal, os.X_OK):
            raise ValueError('Unable to execute ' + args.exec_cal)
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
    print('Using ', cpu_use, ' CPUs for testing')
    sys.stdout.flush()
    if args.store_path_cal and not os.path.exists(args.store_path_cal):
        raise ValueError("Path for storing test results from autocalibration does not exist")

    if args.complete_res_path:
        if not os.path.exists(args.complete_res_path):
            raise ValueError('Folder ' + args.complete_res_path + ' does not exist')
        pyfilepath = os.path.dirname(os.path.realpath(__file__))
        parent = os.path.dirname(pyfilepath)
        if not args.path:
            args.path = os.path.join(pyfilepath, 'Config_Files')
            if not os.path.exists(args.path):
                raise ValueError('Missing initial configuration main folder within python files folder')
        if not args.path_confs_out:
            args.path_confs_out = os.path.join(args.complete_res_path, 'conf_files_generated')
            try:
                os.mkdir(args.path_confs_out)
            except FileExistsError:
                pass
        if not args.img_path:
            args.img_path = os.path.join(parent, 'images')
            if not os.path.exists(args.img_path):
                raise ValueError('Missing image folder at: ' + args.img_path)
        if not args.store_path_sequ:
            args.store_path_sequ = os.path.join(args.complete_res_path, 'sequences_generated')
            try:
                os.mkdir(args.store_path_sequ)
            except FileExistsError:
                pass
        if not args.exec_sequ:
            args.exec_sequ = os.path.join(parent, 'generateVirtualSequence/build/virtualSequenceLib-CMD-interface')
            if not os.path.isfile(args.exec_sequ):
                raise ValueError('Executable ' + args.exec_sequ + ' for generating scenes does not exist')
            elif not os.access(args.exec_sequ, os.X_OK):
                raise ValueError('Unable to execute ' + args.exec_sequ)
        if not args.message_path:
            args.message_path = os.path.join(args.complete_res_path, 'messages')
            try:
                os.mkdir(args.message_path)
            except FileExistsError:
                pass
        if not args.exec_cal:
            args.exec_cal = os.path.join(parent, 'matchinglib_poselib/build/noMatch_poselib-test')
            if not os.path.isfile(args.exec_cal):
                raise ValueError('Executable ' + args.exec_cal + ' of autocalibration does not exist')
            elif not os.access(args.exec_cal, os.X_OK):
                raise ValueError('Unable to execute ' + args.exec_cal)
        if not args.store_path_cal:
            args.store_path_cal = os.path.join(args.complete_res_path, 'testing_results')
            try:
                os.mkdir(args.store_path_cal)
            except FileExistsError:
                pass

    if args.cal_retry_file or args.cal_retry_nrCall:
        if not (args.cal_retry_file and args.cal_retry_nrCall):
            raise ValueError('Both parameters cal_retry_file and cal_retry_nrCall must be provided')
        if args.cal_retry_nrCall < 1:
            raise ValueError("Invalid nrCall number")
        if not os.path.exists(args.cal_retry_file):
            raise ValueError('File ' + args.cal_retry_file + ' does not exist.')
        if not args.store_path_cal:
            raise ValueError('Argument store_path_cal must be provided')
        if not args.exec_cal:
            raise ValueError('Argument exec_cal must be provided')

    enable_messaging()
    enable_logging(args.message_path)

    if args.crt_sc_dirs_file and args.crt_sc_cmds_file:
        ret = retry_scenes_gen_dir(args.crt_sc_dirs_file, args.exec_sequ, args.message_path, cpu_use)
        ret += retry_scenes_gen_cmds(args.crt_sc_cmds_file, args.message_path, cpu_use)
        sys.exit(ret)
    if args.crt_sc_dirs_file:
        ret = retry_scenes_gen_dir(args.crt_sc_dirs_file, args.exec_sequ, args.message_path, cpu_use)
        sys.exit(ret)
    if args.crt_sc_cmds_file:
        ret = retry_scenes_gen_cmds(args.crt_sc_cmds_file, args.message_path, cpu_use)
        sys.exit(ret)

    if args.cal_retry_file:
        ret = retry_autocalibration(args.cal_retry_file, args.cal_retry_nrCall, args.store_path_cal,
                                    args.exec_cal, args.message_path, cpu_use)
        sys.exit(ret)

    try:
        use_evals = get_skip_use_evals(args.skip_use_eval_name_nr)
        use_cal_tests = get_skip_use_cal_tests(args.skip_use_test_name_nr)

        ret = start_testing(args.path, args.path_confs_out, args.skip_tests, args.skip_gen_sc_conf, args.skip_crt_sc,
                            use_cal_tests, use_evals, args.img_path, args.store_path_sequ, args.load_path, cpu_use,
                            args.exec_sequ, args.message_path, args.exec_cal, args.store_path_cal, args.compare_pars,
                            args.comp_pars_ev_nr, args.compare_path)
    except Exception:
        logging.error('Error in main file', exc_info=True)
        ret = 99
        send_message('Error in main file')
    sys.exit(ret)


if __name__ == "__main__":
    main()