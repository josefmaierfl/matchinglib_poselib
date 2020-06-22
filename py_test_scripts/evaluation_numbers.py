"""
Holds variables containing all available evaluation names and their number of tests in addition to lists of eval numbers
for every test
"""


def get_available_evals():
    main_test_names = ['usac-testing', 'usac_vs_ransac', 'refinement_ba', 'vfc_gms_sof',
                       'refinement_ba_stereo', 'correspondence_pool', 'robustness', 'usac_vs_autocalib']
    sub_test_numbers = [2, 1, 2, 1, 2, 3, 6, 1]
    sub_sub_test_nr = [[list(range(1, 7)) + list(range(37, 40)), list(range(7, 15)) + [36] + list(range(40, 43))],
                       [list(range(1, 11))],
                       [list(range(1, 6)), list(range(1, 5))],
                       [list(range(1, 10))],
                       [list(range(1, 4)), list(range(1, 5))],
                       [list(range(1, 11)) + [16], list(range(11, 14)), list(range(14, 16))],
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


def get_config_file_dir(test_name, test_nr=None, return_mult_dirs=False):
    if test_name not in get_available_sequences():
        raise ValueError('Cannot find config files for given test name ' + test_name)
    dirs = {'usac-testing': 'USAC',
            'correspondence_pool': 'Corr_Pool',
            'robustness': {'1': 'Robustness_small',
                           '2': 'Robustness_small',
                           '3': 'Robustness_small',
                           '4': 'Robustness_small',
                           '5': 'Robustness_small',
                           '6': 'Robustness_large'},
            'usac_vs_autocalib': 'USAC_vs_Autocalib'}
    tmp = dirs[test_name]
    if isinstance(tmp, dict):
        if return_mult_dirs:
            return list(dict.fromkeys(list(tmp.values())))
        elif test_nr is None:
            raise ValueError('test_nr must be provided for selecting configuration folder')
        tmp = tmp[str(test_nr)]
    elif return_mult_dirs:
        return [tmp]
    return tmp


def get_mult_conf_dirs_per_test(test_name):
    return get_config_file_dir(test_name, None, True)


def get_autocalib_sequence_config_ref(test_name, test_nr):
    data_from_test = {'usac-testing': {'1': ('usac-testing', 1),
                                       '2': ('usac-testing', 1)},
                      'usac_vs_ransac': ('usac-testing', 1),
                      'refinement_ba': {'1': ('usac-testing', 1),
                                        '2': ('usac-testing', 1)},
                      'vfc_gms_sof': ('usac-testing', 1),
                      'refinement_ba_stereo': {'1': ('usac-testing', 1),
                                               '2': ('usac-testing', 1)},
                      'correspondence_pool': {'1': ('correspondence_pool', 1),
                                              '2': ('correspondence_pool', 1),
                                              '3': ('usac-testing', 1)},
                      'robustness': {'1': ('robustness', 1),
                                     '2': ('robustness', 1),
                                     '3': ('robustness', 1),
                                     '4': ('robustness', 1),
                                     '5': ('robustness', 1),
                                     '6': ('robustness', 6)},
                      'usac_vs_autocalib': ('usac_vs_autocalib', None)}
    tmp = data_from_test[test_name]
    if isinstance(tmp, dict):
        tmp = tmp[str(test_nr)]
    return tmp


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


def get_opt_pars_func(test_name, test_nr):
    pars = check_calc_opt_pars(test_name, test_nr)
    if pars is None:
        return None
    import calc_opt_parameters as cop
    pars_info = {'usac-testing': {'1': {'func': cop.get_usac_testing_1, 'test_nrs': [1]},
                                  '2': {'func': cop.get_usac_testing_2, 'test_nrs': [1, 2]}},
                 'usac_vs_ransac': {'func': cop.get_usac_vs_ransac, 'test_nrs': [None]},
                 'refinement_ba': {'2': {'func': cop.get_refinement_ba_2, 'test_nrs': [1, 2]}},
                 'refinement_ba_stereo': {'2': {'func': cop.get_refinement_ba_stereo_2, 'test_nrs': [1, 2]}},
                 'correspondence_pool': {'1': {'func': cop.get_correspondence_pool_1, 'test_nrs': [1]},
                                         '2': {'func': cop.get_correspondence_pool_2, 'test_nrs': [2]}},
                 'robustness': {'1': {'func': cop.get_robustness_1, 'test_nrs': [1]},
                                '2': {'func': cop.get_robustness_2, 'test_nrs': [2]},
                                '4': {'func': cop.get_robustness_4, 'test_nrs': [4]},
                                '5': {'func': cop.get_robustness_5, 'test_nrs': [5]}}}
    tmp = pars_info[test_name]
    if test_nr:
        tmp = tmp[str(test_nr)]
    tmp['pars'] = pars
    return tmp


def get_evals_with_compare():
    evals_w_compare = [('refinement_ba_stereo', 1, 1),
                       ('refinement_ba_stereo', 1, 2),
                       ('refinement_ba_stereo', 2, 1),
                       ('refinement_ba_stereo', 2, 2),
                       ('refinement_ba_stereo', 2, 3),
                       ('refinement_ba_stereo', 2, 4),
                       ('correspondence_pool', 3, 14),
                       ('correspondence_pool', 3, 15)]
    return evals_w_compare


def get_eval_nrs_for_cmp(test_name, test_nr):
    evals_w_compare = get_evals_with_compare()
    ev_nr = []
    for i in evals_w_compare:
        if i[0] == test_name and i[1] == test_nr:
            ev_nr.append(i[2])
    return ev_nr


def check_if_eval_needs_compare_data(test_name, test_nr, eval_nr=None):
    if test_nr is None:
        return False
    test_names, test_nrs = get_available_tests()
    test_idx = test_names.index(test_name)
    max_testnr = test_nrs[test_idx]
    if max_testnr == 1:
        return False
    ec = get_evals_with_compare()
    if eval_nr is None:
        for a in ec:
            if a[0] == test_name and a[1] == test_nr:
                return True
    elif isinstance(eval_nr, list):
        for a in ec:
            if a[0] == test_name and a[1] == test_nr and any(a[2] == b for b in eval_nr):
                return True
    else:
        for a in ec:
            if a[0] == test_name and a[1] == test_nr and a[2] == eval_nr:
                return True
    return False


def get_load_pars_for_comparison(test_name, test_nr):
    load_pars = {'refinement_ba_stereo': {'1': ['refineRT', 'bart'],
                                          '2': ['refineRT']},
                 'correspondence_pool': {'3': ['refineRT', 'bart']}}
    return load_pars[test_name][str(test_nr)]


def get_used_eval_cols(test_name, test_nr):
    test_cols = {'usac-testing': {'1': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                        't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz', 'kpDistr',
                                        'USAC_parameters_estimator', 'USAC_parameters_refinealg', 'th', 'inlratMin',
                                        'depthDistr', 'kpAccSd', 'nrCorrs_GT', 'robEstimationAndRef_us', 'nrTP'],
                                  '2': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                        't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                        'USAC_parameters_automaticSprtInit', 'inlRat_estimated', 'inlRat_GT',
                                        'USAC_parameters_automaticProsacParameters', 'nrTP', 'kpDistr',
                                        'USAC_parameters_prevalidateSample', 'USAC_parameters_USACInlratFilt', 'th',
                                        'inlratMin', 'kpAccSd', 'depthDistr', 'robEstimationAndRef_us', 'nrCorrs_GT']},
                 'usac_vs_ransac': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                    't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz', 'RobMethod',
                                    'th', 'inlratMin', 'inlRat_estimated', 'inlRat_GT', 'depthDistr', 'kpAccSd',
                                    'robEstimationAndRef_us', 'nrCorrs_GT', 'nrTP', 'kpDistr'],
                 'refinement_ba': {'1': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                         't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                         'refineMethod_algorithm', 'refineMethod_costFunction', 'BART', 'inlratMin',
                                         'depthDistr', 'kpAccSd', 'linRef_BA_us', 'nrCorrs_GT', 'nrTP', 'kpDistr',
                                         'linRefinement_us', 'bundleAdjust_us', 'robEstimationAndRef_us'],
                                   '2': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                         't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                         'refineMethod_algorithm', 'refineMethod_costFunction', 'inlratMin',
                                         'depthDistr', 'kpAccSd', 'K1_cxyfxfyNorm', 'K2_cxyfxfyNorm', 'K1_cxyDiffNorm',
                                         'K2_cxyDiffNorm', 'K1_fxyDiffNorm', 'K2_fxyDiffNorm', 'K1_fxDiff',
                                         'K2_fxDiff', 'K1_fyDiff', 'K2_fyDiff', 'K1_cxDiff', 'K2_cxDiff',
                                         'K1_cyDiff', 'K2_cyDiff', 'nrTP', 'kpDistr']},
                 'vfc_gms_sof': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                 't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                 'matchesFilter_refineGMS', 'matchesFilter_refineVFC', 'matchesFilter_refineSOF',
                                 'inlratMin', 'inlRat_estimated', 'inlRat_GT', 'kpAccSd', 'kpDistr', 'depthDistr',
                                 'filtering_us', 'nrCorrs_GT', 'nrTP'],
                 'refinement_ba_stereo': {'1': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                                'stereoParameters_refineMethod_CorrPool_algorithm',
                                                'stereoParameters_refineMethod_CorrPool_costFunction',
                                                'stereoParameters_BART_CorrPool', 'inlratMin', 'depthDistr', 'kpAccSd',
                                                'stereoRefine_us', 'kpDistr'],
                                          '2': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                                't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                                'stereoParameters_refineMethod_CorrPool_algorithm',
                                                'stereoParameters_refineMethod_CorrPool_costFunction', 'inlratMin',
                                                'depthDistr', 'kpAccSd', 'K1_cxyfxfyNorm', 'K2_cxyfxfyNorm',
                                                'K1_cxyDiffNorm', 'K2_cxyDiffNorm', 'K1_fxyDiffNorm', 'K2_fxyDiffNorm',
                                                'K1_fxDiff', 'K2_fxDiff', 'K1_fyDiff', 'K2_fyDiff', 'K1_cxDiff',
                                                'K2_cxDiff', 'K1_cyDiff', 'K2_cyDiff', 'kpDistr']},
                 'correspondence_pool': {'1': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                               't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                               'stereoParameters_maxPoolCorrespondences',
                                               'stereoParameters_minPtsDistance', 'inlratMin', 'depthDistr',
                                               'kpAccSd', 'Nr', 'poolSize', 'stereoRefine_us', 'kpDistr'],
                                         '2': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                               't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                               'stereoParameters_maxRat3DPtsFar', 'stereoParameters_maxDist3DPtsZ',
                                               'inlratMin', 'depthDistr', 'kpAccSd', 'stereoRefine_us', 'Nr',
                                               'kpDistr'],
                                         '3': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                               't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                               'stereoParameters_maxRat3DPtsFar', 'stereoParameters_maxDist3DPtsZ',
                                               'stereoParameters_maxPoolCorrespondences',
                                               'stereoParameters_minPtsDistance', 'kpAccSd', 'stereoRefine_us',
                                               'nrTP', 'kpDistr']},
                 'robustness': {'1': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                      't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                      'stereoParameters_relInlRatThLast', 'stereoParameters_relInlRatThNew',
                                      'stereoParameters_minInlierRatSkip', 'stereoParameters_minInlierRatioReInit',
                                      'stereoParameters_relMinInlierRatSkip', 'Nr', 'R_GT_n_diffAll', 't_GT_n_angDiff',
                                      'R_GT_n_diff_roll_deg', 'R_GT_n_diff_pitch_deg', 'R_GT_n_diff_yaw_deg',
                                      't_GT_n_elemDiff_tx', 't_GT_n_elemDiff_ty', 't_GT_n_elemDiff_tz', 'inlratCRate',
                                      'depthDistr', 'kpAccSd', 'kpDistr'],
                                '2': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                      't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                      'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                      'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                      't_mostLikely_angDiff_deg', 't_mostLikely_distDiff', 't_mostLikely_diff_tx',
                                      't_mostLikely_diff_ty', 't_mostLikely_diff_tz',
                                      'stereoParameters_checkPoolPoseRobust', 'R_GT_n_diffAll', 't_GT_n_angDiff',
                                      'R_GT_n_diff_roll_deg', 'R_GT_n_diff_pitch_deg', 'R_GT_n_diff_yaw_deg',
                                      't_GT_n_elemDiff_tx', 't_GT_n_elemDiff_ty', 't_GT_n_elemDiff_tz', 'Nr',
                                      'inlratCRate', 'depthDistr', 'kpAccSd', 'stereoRefine_us',
                                      'R_mostLikely(0,0)', 'R_mostLikely(0,1)', 'R_mostLikely(0,2)',
                                      'R_mostLikely(1,0)', 'R_mostLikely(1,1)', 'R_mostLikely(1,2)',
                                      'R_mostLikely(2,0)', 'R_mostLikely(2,1)', 'R_mostLikely(2,2)', 'kpDistr'],
                                '3': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                      't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                      'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                      'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                      't_mostLikely_angDiff_deg', 't_mostLikely_distDiff', 't_mostLikely_diff_tx',
                                      't_mostLikely_diff_ty', 't_mostLikely_diff_tz', 'R_GT_n_diffAll',
                                      't_GT_n_angDiff', 'R_GT_n_diff_roll_deg', 'R_GT_n_diff_pitch_deg',
                                      'R_GT_n_diff_yaw_deg', 't_GT_n_elemDiff_tx', 't_GT_n_elemDiff_ty',
                                      't_GT_n_elemDiff_tz', 'Nr', 'inlratCRate', 'depthDistr', 'kpAccSd',
                                      'R_mostLikely(0,0)', 'R_mostLikely(0,1)', 'R_mostLikely(0,2)',
                                      'R_mostLikely(1,0)', 'R_mostLikely(1,1)', 'R_mostLikely(1,2)',
                                      'R_mostLikely(2,0)', 'R_mostLikely(2,1)', 'R_mostLikely(2,2)', 'kpDistr'],
                                '4': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                      't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                      'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                      'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                      't_mostLikely_angDiff_deg', 't_mostLikely_distDiff', 't_mostLikely_diff_tx',
                                      't_mostLikely_diff_ty', 't_mostLikely_diff_tz',
                                      'stereoParameters_minContStablePoses', 'stereoParameters_minNormDistStable',
                                      'stereoParameters_absThRankingStable', 'R_GT_n_diffAll', 't_GT_n_angDiff',
                                      'R_GT_n_diff_roll_deg', 'R_GT_n_diff_pitch_deg', 'R_GT_n_diff_yaw_deg',
                                      't_GT_n_elemDiff_tx', 't_GT_n_elemDiff_ty', 't_GT_n_elemDiff_tz', 'Nr',
                                      'inlratCRate', 'depthDistr', 'kpAccSd', 'poseIsStable',
                                      'R_mostLikely(0,0)', 'R_mostLikely(0,1)', 'R_mostLikely(0,2)',
                                      'R_mostLikely(1,0)', 'R_mostLikely(1,1)', 'R_mostLikely(1,2)',
                                      'R_mostLikely(2,0)', 'R_mostLikely(2,1)', 'R_mostLikely(2,2)',
                                      'mostLikelyPose_stable', 'kpDistr'],
                                '5': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                      't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                      'stereoParameters_useRANSAC_fewMatches', 'inlratCRate', 'depthDistr', 'kpAccSd',
                                      'R_GT_n_diffAll', 't_GT_n_angDiff', 'R_GT_n_diff_roll_deg',
                                      'R_GT_n_diff_pitch_deg', 'R_GT_n_diff_yaw_deg', 't_GT_n_elemDiff_tx',
                                      't_GT_n_elemDiff_ty', 't_GT_n_elemDiff_tz', 'Nr', 'stereoRefine_us', 'kpDistr'],
                                '6': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                      't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                      'R_GT_n_diffAll', 't_GT_n_angDiff', 'R_GT_n_diff_roll_deg',
                                      'R_GT_n_diff_pitch_deg', 'R_GT_n_diff_yaw_deg', 't_GT_n_elemDiff_tx',
                                      't_GT_n_elemDiff_ty', 't_GT_n_elemDiff_tz', 'Nr', 'inlratCRate', 'depthDistr',
                                      'kpAccSd', 'R_mostLikely_diffAll', 'R_mostLikely_diff_roll_deg',
                                      'R_mostLikely_diff_pitch_deg', 'R_mostLikely_diff_yaw_deg',
                                      't_mostLikely_angDiff_deg', 't_mostLikely_distDiff',
                                      't_mostLikely_diff_tx', 't_mostLikely_diff_ty', 't_mostLikely_diff_tz',
                                      'R_mostLikely(0,0)', 'R_mostLikely(0,1)', 'R_mostLikely(0,2)',
                                      'R_mostLikely(1,0)', 'R_mostLikely(1,1)', 'R_mostLikely(1,2)',
                                      'R_mostLikely(2,0)', 'R_mostLikely(2,1)', 'R_mostLikely(2,2)', 'kpDistr']},
                 'usac_vs_autocalib': ['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
                                       't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
                                       'stereoRef', 'inlratMin', 'kpAccSd', 'depthDistr', 'R_GT_n_diffAll',
                                       't_GT_n_angDiff', 'R_GT_n_diff_roll_deg', 'R_GT_n_diff_pitch_deg',
                                       'R_GT_n_diff_yaw_deg', 't_GT_n_elemDiff_tx', 't_GT_n_elemDiff_ty',
                                       't_GT_n_elemDiff_tz', 'Nr', 'linRefinement_us', 'bundleAdjust_us',
                                       'robEstimationAndRef_us', 'stereoRefine_us', 'kpDistr', 'accumCorrs']}
    cols_sel = test_cols[test_name]
    if test_nr is not None:
        return cols_sel[str(test_nr)]
    return cols_sel