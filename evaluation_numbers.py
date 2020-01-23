"""
Holds variables containing all available evaluation names and their number of tests in addition to lists of eval numbers
for every test
"""


def get_available_evals():
    main_test_names = ['usac-testing', 'usac_vs_ransac', 'refinement_ba', 'vfc_gms_sof',
                       'refinement_ba_stereo', 'correspondence_pool', 'robustness', 'usac_vs_autocalib']
    sub_test_numbers = [2, 1, 2, 1, 2, 3, 6, 1]
    sub_sub_test_nr = [[list(range(1, 7)), list(range(7, 15)) + [36]],
                       [list(range(1, 8))],
                       [list(range(1, 6)), list(range(1, 5))],
                       [list(range(1, 8))],
                       [list(range(1, 4)), list(range(1, 5))],
                       [list(range(1, 11)), list(range(11, 14)), list(range(14, 16))],
                       [list(range(1, 6)), list(range(6, 11)), list(range(11, 15)), list(range(15, 25)),
                        list(range(25, 29)), list(range(29, 38))],
                       [list(range(1, 9))]]
    return main_test_names, sub_test_numbers, sub_sub_test_nr


def get_available_tests():
    main_test_names = ['usac-testing', 'usac_vs_ransac', 'refinement_ba', 'vfc_gms_sof',
                       'refinement_ba_stereo', 'correspondence_pool', 'robustness', 'usac_vs_autocalib']
    sub_test_numbers = [2, 1, 2, 1, 2, 3, 6, 1]
    return main_test_names, sub_test_numbers


def get_available_main_tests():
    main_test_names = ['usac-testing', 'usac_vs_ransac', 'refinement_ba', 'vfc_gms_sof',
                       'refinement_ba_stereo', 'correspondence_pool', 'robustness', 'usac_vs_autocalib']
    return main_test_names


def get_available_sequences():
    return ['usac-testing', 'correspondence_pool', 'robustness', 'usac_vs_autocalib']


def get_config_file_dir(test_name):
    if test_name not in get_available_sequences():
        raise ValueError('Cannot find config files for given test name ' + test_name)
    dirs = {'usac-testing': 'USAC',
            'correspondence_pool': 'Corr_Pool',
            'robustness': 'Robustness',
            'usac_vs_autocalib': 'USAC_vs_Autocalib'}
    return dirs[test_name]


def get_config_file_parameters(test_name):
    if test_name not in get_available_sequences():
        raise ValueError('Cannot create config files for given test name ' + test_name)
    parameters = {'usac-testing': {'inlier_range': [0.1, 0.9, 0.1],
                                   'kpAccRange': [0.5, 4.0, 0.5],
                                   'treatTPasCorrs': True,
                                   'treatTPasCorrsConfsSel': 'TP-100to1000'},
                  'correspondence_pool': {'inlier_range': [0.1, 0.9, 0.1],
                                          'kpAccRange': [0.5, 4.0, 0.5]},
                  'robustness': {'inlchrate_range': [0.1, 0.7, 0.15, 1.0],
                                 'kpAccRange': [0.5, 3.5, 1.0]},
                  'usac_vs_autocalib': {'inlier_range': [0.1, 0.9, 0.2],
                                        'kpAccRange': [0.5, 3.5, 1.0]}}
    return parameters[test_name]


def check_calc_opt_pars(test_name, test_nr):
    parameters = {'usac-testing': {'1': ['USAC_parameters_estimator', 'USAC_parameters_refinealg'],
                                   '2': ['USAC_parameters_automaticSprtInit',
                                         'USAC_parameters_automaticProsacParameters',
                                         'USAC_parameters_prevalidateSample', 'USAC_parameters_USACInlratFilt', 'th']},
                  'usac_vs_ransac': ['RobMethod'],
                  'refinement_ba': {'1': None, '2': ['refineMethod_algorithm', 'refineMethod_costFunction', 'BART']},
                  'vfc_gms_sof': None,
                  'refinement_ba_stereo': {'1': None, '2': ['stereoParameters_refineMethod_CorrPool_algorithm',
                                                            'stereoParameters_refineMethod_CorrPool_costFunction',
                                                            'stereoParameters_BART_CorrPool']},
                  'correspondence_pool': {'1': ['stereoParameters_minPtsDistance',
                                                'stereoParameters_maxPoolCorrespondences'],
                                          '2': ['stereoParameters_maxRat3DPtsFar', 'stereoParameters_maxDist3DPtsZ'],
                                          '3': None},
                  'robustness': {'1': ['stereoParameters_relInlRatThLast', 'stereoParameters_relInlRatThNew',
                                       'stereoParameters_minInlierRatSkip', 'stereoParameters_relMinInlierRatSkip',
                                       'stereoParameters_minInlierRatioReInit'],
                                 '2': ['stereoParameters_checkPoolPoseRobust'],
                                 '3': None,
                                 '4': ['stereoParameters_minContStablePoses', 'stereoParameters_minNormDistStable',
                                       'stereoParameters_absThRankingStable'],
                                 '5': ['stereoParameters_useRANSAC_fewMatches'],
                                 '6': None},
                  'usac_vs_autocalib': None}
    pars_sel = parameters[test_name]
    if pars_sel is None:
        return None
    elif isinstance(pars_sel, list):
        return pars_sel
    return pars_sel[str(test_nr)]


# def get_res_par_keys(par_est)