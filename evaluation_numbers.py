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