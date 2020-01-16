"""
Main script file for executing the whole test procedure for testing the autocalibration SW
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np


def get_skip_use_evals(strings):
    import evaluation_numbers as en
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
    import evaluation_numbers as en
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


def main():
    parser = argparse.ArgumentParser(description='Main script file for executing the whole test procedure for '
                                                 'testing the autocalibration SW')
    parser.add_argument('--path', type=str, required=False,
                        help='Directory holding directories with template configuration files')
    parser.add_argument('--path_confs_out', type=str, required=False,
                        help='Optional directory for writing configuration files. If not available, '
                             'the directory is derived from argument \'path\'.')
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
                             'the absolute value of nrCPUs is used. Default: 4')
    parser.add_argument('--exec_sequ', type=str, required=False,
                        help='Executable of the application generating the sequences')
    parser.add_argument('--message_path', type=str, required=True,
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
    if args.path and not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' holding directories with template scene '
                                                    'configuration files does not exist')
    import evaluation_numbers as en
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
    if not os.path.exists(args.message_path):
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
    if args.store_path_cal and not os.path.exists(args.store_path_cal):
        raise ValueError("Path for storing test results from autocalibration does not exist")
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

    use_evals = get_skip_use_evals(args.skip_use_eval_name_nr)
    use_cal_tests = get_skip_use_cal_tests(args.skip_use_test_name_nr)

    return start_testing(args.path, args.path_confs_out)


    return 0


if __name__ == "__main__":
    main()