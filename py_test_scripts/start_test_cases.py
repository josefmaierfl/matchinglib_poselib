"""
Execute different test scenarios for the autocalibration based on name and test number in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, warnings, time, subprocess as sp
import ruamel.yaml as yaml, numpy as np

def choose_test(path_ov_file, executable, cpu_cnt, message_path, output_path, test_name, test_nr):
    args = ['--path', path_ov_file, '--nrCPUs', str(cpu_cnt), '--executable', executable,
            '--message_path', message_path]

    #Set to False when performing actual tests:
    testing_test = False
    pars = {}

    #Set settings based on best results after testing for further testing
    # Change cfgUSAC parameters 5 & 6 based on result of test_nr 1 of usac-testing
    pars['usac56'] = []
    # Change cfgUSAC parameters 1-3 based on result of test_nr 2 of usac-testing
    pars['usac123'] = []
    # Change USACInlratFilt parameter based on result of test_nr 2 of usac-testing
    pars['USACInlratFilt'] = None
    # Change th parameter based on results of test_nr 1 & 2 of usac-testing
    pars['th'] = None
    # Change refineRT parameters based on result of test_nr 1 of refinement_ba
    pars['refineRT'] = []
    # Select robust method for testing VFC, GMS, and SOF: If best USAC solution is without GMS or VFC (for estimating
    # some USAC parameters (cfgUSAC digit 1 >1)), try using them for pre-filtering, otherwise use RANSAC
    # and compare with its original results
    pars['robMFilt'] = None
    # Change BART parameter based on results of test_nr 1 & 2 of refinement_ba
    pars['bart'] = None
    # Change refineRT_stereo parameters based on result of test_nr 1 of refinement_ba_stereo
    pars['refineRT_stereo'] = []
    # Change BART_stereo parameter based on results of test_nr 1 & 2 of refinement_ba_stereo
    pars['bart_stereo'] = None
    # Change minPtsDistance parameter based on result of test_nr 1 of correspondence_pool
    pars['minPtsDistance'] = None
    # Change maxPoolCorrespondences parameter based on result of test_nr 1 of correspondence_pool
    pars['maxPoolCorrespondences'] = None
    # Change maxRat3DPtsFar parameter based on result of test_nr 2 of correspondence_pool
    pars['maxRat3DPtsFar'] = None
    # Change maxDist3DPtsZ parameter based on result of test_nr 2 of correspondence_pool
    pars['maxDist3DPtsZ'] = None
    # Change relInlRatThLast parameter based on result of test_nr 1 of robustness
    pars['relInlRatThLast'] = None
    # Change relInlRatThNew parameter based on result of test_nr 1 of robustness
    pars['relInlRatThNew'] = None
    # Change minInlierRatSkip parameter based on result of test_nr 1 of robustness
    pars['minInlierRatSkip'] = None
    # Change relMinInlierRatSkip parameter based on result of test_nr 1 of robustness
    pars['relMinInlierRatSkip'] = None
    # Change minInlierRatioReInit parameter based on result of test_nr 1 of robustness
    pars['minInlierRatioReInit'] = None
    # Change checkPoolPoseRobust parameter based on result of test_nr 2 of robustness
    pars['checkPoolPoseRobust'] = None
    # Change minContStablePoses parameter based on result of test_nr 3 of robustness
    pars['minContStablePoses'] = None
    # Change minNormDistStable parameter based on result of test_nr 3 of robustness
    pars['minNormDistStable'] = None
    # Change absThRankingStable parameter based on result of test_nr 3 of robustness
    pars['absThRankingStable'] = None
    # Change useRANSAC_fewMatches to true or false based on result of test_nr 4 of robustness
    pars['useRANSAC_fewMatches'] = None

    if test_name != 'usac-testing' or test_nr > 1:
        if test_name == 'usac-testing' and test_nr == 2:
            pars_list = ['usac56']
        elif test_name == 'usac_vs_ransac':
            pars_list = ['usac56', 'usac123', 'USACInlratFilt']
        elif test_name == 'refinement_ba':
            pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th']
        elif test_name == 'vfc_gms_sof':
            pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'robMFilt']
        elif test_name == 'refinement_ba_stereo':
            pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'refineRT', 'bart']
        elif test_name == 'correspondence_pool':
            if test_nr == 1:
                pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'refineRT', 'bart',
                             'refineRT_stereo', 'bart_stereo']
            elif test_nr == 2:
                pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'refineRT', 'bart',
                             'refineRT_stereo', 'bart_stereo', 'minPtsDistance', 'maxPoolCorrespondences']
            elif test_nr == 3:
                pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'refineRT', 'bart',
                             'refineRT_stereo', 'bart_stereo', 'minPtsDistance', 'maxPoolCorrespondences',
                             'maxRat3DPtsFar', 'maxDist3DPtsZ']
        elif test_name == 'robustness':
            if test_nr == 1:
                pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'refineRT', 'bart',
                             'refineRT_stereo', 'bart_stereo', 'minPtsDistance', 'maxPoolCorrespondences',
                             'maxRat3DPtsFar', 'maxDist3DPtsZ']
            elif test_nr == 2:
                pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'refineRT', 'bart',
                             'refineRT_stereo', 'bart_stereo', 'minPtsDistance', 'maxPoolCorrespondences',
                             'maxRat3DPtsFar', 'maxDist3DPtsZ', 'relInlRatThLast', 'relInlRatThNew',
                             'minInlierRatSkip', 'relMinInlierRatSkip', 'minInlierRatioReInit']
            elif test_nr == 3 or test_nr == 4:
                pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'refineRT', 'bart',
                             'refineRT_stereo', 'bart_stereo', 'minPtsDistance', 'maxPoolCorrespondences',
                             'maxRat3DPtsFar', 'maxDist3DPtsZ', 'relInlRatThLast', 'relInlRatThNew',
                             'minInlierRatSkip', 'relMinInlierRatSkip', 'minInlierRatioReInit',
                             'checkPoolPoseRobust']
            elif test_nr == 5:
                pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'refineRT', 'bart',
                             'refineRT_stereo', 'bart_stereo', 'minPtsDistance', 'maxPoolCorrespondences',
                             'maxRat3DPtsFar', 'maxDist3DPtsZ', 'relInlRatThLast', 'relInlRatThNew',
                             'minInlierRatSkip', 'relMinInlierRatSkip', 'minInlierRatioReInit',
                             'checkPoolPoseRobust', 'minContStablePoses', 'minNormDistStable', 'absThRankingStable']
            elif test_nr == 6:
                pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'refineRT', 'bart',
                             'refineRT_stereo', 'bart_stereo', 'minPtsDistance', 'maxPoolCorrespondences',
                             'maxRat3DPtsFar', 'maxDist3DPtsZ', 'relInlRatThLast', 'relInlRatThNew',
                             'minInlierRatSkip', 'relMinInlierRatSkip', 'minInlierRatioReInit',
                             'checkPoolPoseRobust', 'minContStablePoses', 'minNormDistStable', 'absThRankingStable',
                             'useRANSAC_fewMatches']
        elif test_name == 'usac_vs_autocalib':
            pars_list = ['usac56', 'usac123', 'USACInlratFilt', 'th', 'refineRT', 'bart',
                         'refineRT_stereo', 'bart_stereo', 'minPtsDistance', 'maxPoolCorrespondences',
                         'maxRat3DPtsFar', 'maxDist3DPtsZ', 'relInlRatThLast', 'relInlRatThNew',
                         'minInlierRatSkip', 'relMinInlierRatSkip', 'minInlierRatioReInit',
                         'checkPoolPoseRobust', 'minContStablePoses', 'minNormDistStable', 'absThRankingStable',
                         'useRANSAC_fewMatches']
        else:
            raise ValueError('Unknown test name ' + test_name)
        pars_opt = read_pars(output_path, pars_list)
        for i in pars_opt.keys():
            pars[i] = pars_opt[i]
    else:
        try:
            write_par_file_template(output_path)
        except ValueError:
            raise ValueError('Parameter file for storing optimal parameters already exists.')

    if test_name == 'usac-testing':
        args += ['--refineRT', '0', '0']
        args += ['--RobMethod', 'USAC']
        args += ['--th', '0.6', '2.0', '0.2']
        args += ['--useGTCamMat']
        if not test_nr:
            raise ValueError('test_nr is required for usac-testing')
        if test_nr == 1:
            args += ['--inlier_ratios', '0.2', '0.4', '0.6', '0.8']
            args += ['--kp_accs', '1.5']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5', '0', '0', '0', '0', '1', '1']
            args += ['--USACInlratFilt', '0']
        elif test_nr == 2:
            # args += ['--depths', 'NMF']
            # args += ['--nr_keypoints', '500']
            # args += ['--kp_pos_distr', 'equ']
            if not pars['usac56']:
                raise ValueError('Enter best test results for parameters 5 & 6 of usac-testing')
            args += ['--cfgUSAC', '3', '1', '1', '0'] + list(map(str, pars['usac56'])) + ['1', '1', '1', '0', '0', '0']
            args += ['--USACInlratFilt', '2']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for usac-testing')
    elif test_name == 'usac_vs_ransac':
        args += ['--refineRT', '0', '0']
        args += ['--RobMethod', 'USAC', 'RANSAC']
        args += ['--th', '0.6', '2.0', '0.2']
        args += ['--useGTCamMat']
        if not pars['usac56'] or not pars['usac123']:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, pars['usac123'])) + ['0'] + list(map(str, pars['usac56']))
        if pars['USACInlratFilt'] is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(pars['USACInlratFilt'])]
    elif test_name == 'refinement_ba':
        args += ['--RobMethod', 'USAC']
        if not pars['usac56'] or not pars['usac123']:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, pars['usac123'])) + ['0'] + list(map(str, pars['usac56']))
        if pars['USACInlratFilt'] is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(pars['USACInlratFilt'])]
        if pars['th'] is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(pars['th'])]
        if not test_nr:
            raise ValueError('test_nr is required for refinement_ba')
        if test_nr == 1:
            args += ['--refineRT', '0', '0', '1', '1', '1']#, '-13']
            args += ['--BART', '0', '1']
            args += ['--useGTCamMat']
        elif test_nr == 2:
            # if not pars['refineRT']:
            #     raise ValueError('Enter best test results for refineRT of refinement_ba')
            # args += ['--refineRT'] + list(map(str, pars['refineRT']))
            args += ['--refineRT', '0', '0', '1', '1', '1']
            args += ['--BART', '2']
            args += ['--nr_keypoints', '500']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for refinement_ba')
    elif test_name == 'vfc_gms_sof':
        if pars['robMFilt'] is None:
            raise ValueError('Enter robust method for testing VFC, GMS, and SOF')
        args += ['--RobMethod', pars['robMFilt']]
        if pars['robMFilt'] == 'USAC':
            if not pars['usac56'] or not pars['usac123']:
                raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
            args += ['--cfgUSAC'] + list(map(str, pars['usac123'])) + ['0'] + list(map(str, pars['usac56']))
        else:
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
        args += ['--USACInlratFilt', '0']
        if pars['th'] is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(pars['th'])]
        args += ['--refineRT', '0', '0']
        args += ['--useGTCamMat']
        args += ['--refineVFC']
        args += ['--refineSOF']
        args += ['--refineGMS']
    elif test_name == 'refinement_ba_stereo':
        args += ['--RobMethod', 'USAC']
        if not pars['usac56'] or not pars['usac123']:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, pars['usac123'])) + ['0'] + list(map(str, pars['usac56']))
        if pars['USACInlratFilt'] is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(pars['USACInlratFilt'])]
        if pars['th'] is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(pars['th'])]
        if not pars['refineRT']:
            raise ValueError('Enter best test results for refineRT of refinement_ba')
        args += ['--refineRT'] + list(map(str, pars['refineRT']))
        if pars['bart'] is None:
            raise ValueError('Enter best test results for BART of refinement_ba')
        args += ['--BART', str(pars['bart'])]
        args += ['--stereoRef']
        args += ['--minStartAggInlRat', '0.075']
        args += ['--nr_keypoints', '500']
        args += ['--maxPoolCorrespondences', '5000']
        args += ['--minInlierRatSkip', '0.075']
        if not test_nr:
            raise ValueError('test_nr is required for refinement_ba_stereo')
        if test_nr == 1:
            args += ['--refineRT_stereo', '0', '0', '1', '1', '1', '-13']
            args += ['--BART_stereo', '0', '1']
            args += ['--useGTCamMat']
        elif test_nr == 2:
            # if not pars['refineRT_stereo']:
            #     raise ValueError('Enter best test results for refineRT_stereo of refinement_ba_stereo')
            # args += ['--refineRT_stereo'] + list(map(str, pars['refineRT_stereo']))
            args += ['--refineRT_stereo', '0', '0', '1', '1', '1']
            args += ['--BART_stereo', '2']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for refinement_ba_stereo')
    elif test_name == 'correspondence_pool':
        args += ['--RobMethod', 'USAC']
        if not pars['usac56'] or not pars['usac123']:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, pars['usac123'])) + ['0'] + list(map(str, pars['usac56']))
        if pars['USACInlratFilt'] is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(pars['USACInlratFilt'])]
        if pars['th'] is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(pars['th'])]
        if not pars['refineRT']:
            raise ValueError('Enter best test results for refineRT of refinement_ba')
        args += ['--refineRT'] + list(map(str, pars['refineRT']))
        if pars['bart'] is None:
            raise ValueError('Enter best test results for BART of refinement_ba')
        args += ['--BART', str(pars['bart'])]
        args += ['--stereoRef']
        args += ['--minStartAggInlRat', '0.075']
        args += ['--minInlierRatSkip', '0.075']
        args += ['--useGTCamMat']
        if not pars['refineRT_stereo']:
            raise ValueError('Enter best test results for refineRT_stereo of refinement_ba_stereo')
        args += ['--refineRT_stereo'] + list(map(str, pars['refineRT_stereo']))
        if pars['bart_stereo'] is None:
            raise ValueError('Enter best test results for BART_stereo of refinement_ba_stereo')
        args += ['--BART_stereo', str(pars['bart_stereo'])]
        if not test_nr:
            raise ValueError('test_nr is required for correspondence_pool')
        if test_nr == 1:
            args += ['--minPtsDistance', '1.5', '15.5', '2.0']
            args += ['--maxPoolCorrespondences', '300', '1000', '100', '1000', '2000', '200', '2000', '5000', '500',
                     '5000', '10000', '1000', '10000', '20000', '2000', '20000', '30000', '5000']
        elif test_nr == 2:
            if pars['minPtsDistance'] is None:
                raise ValueError('Enter best test results for minPtsDistance of correspondence_pool')
            args += ['--minPtsDistance', str(pars['minPtsDistance'])]
            if pars['maxPoolCorrespondences'] is None:
                raise ValueError('Enter best test results for maxPoolCorrespondences of correspondence_pool')
            args += ['--maxPoolCorrespondences', str(pars['maxPoolCorrespondences'])]
            args += ['--maxRat3DPtsFar', '0.1', '0.8', '0.1']
            args += ['--maxDist3DPtsZ', '20.0', '300.0', '20.0']
        elif test_nr == 3:
            #Use generated scenes for testing USAC
            warnings.warn("Warning: Be sure to use the scenes from testing USAC")
            time.sleep(5.0)
            args += ['--depths', 'NMF']
            args += ['--nr_keypoints', '500']
            args += ['--kp_pos_distr', '1corn']
            args += ['--inlier_ratios', '0.6']
            if pars['minPtsDistance'] is None:
                raise ValueError('Enter best test results for minPtsDistance of correspondence_pool')
            args += ['--minPtsDistance', str(pars['minPtsDistance'])]
            if pars['maxPoolCorrespondences'] is None:
                raise ValueError('Enter best test results for maxPoolCorrespondences of correspondence_pool')
            args += ['--maxPoolCorrespondences', str(pars['maxPoolCorrespondences'])]
            if pars['maxRat3DPtsFar'] is None:
                raise ValueError('Enter best test results for maxRat3DPtsFar of correspondence_pool')
            args += ['--maxRat3DPtsFar', str(pars['maxRat3DPtsFar'])]
            if pars['maxDist3DPtsZ'] is None:
                raise ValueError('Enter best test results for maxDist3DPtsZ of correspondence_pool')
            args += ['--maxDist3DPtsZ', str(pars['maxDist3DPtsZ'])]
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for correspondence_pool')
    elif test_name == 'robustness':
        args += ['--RobMethod', 'USAC']
        if not pars['usac56'] or not pars['usac123']:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, pars['usac123'])) + ['0'] + list(map(str, pars['usac56']))
        if pars['USACInlratFilt'] is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(pars['USACInlratFilt'])]
        if pars['th'] is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(pars['th'])]
        if not pars['refineRT']:
            raise ValueError('Enter best test results for refineRT of refinement_ba')
        args += ['--refineRT'] + list(map(str, pars['refineRT']))
        if pars['bart'] is None:
            raise ValueError('Enter best test results for BART of refinement_ba')
        args += ['--BART', str(pars['bart'])]
        args += ['--stereoRef']
        args += ['--useGTCamMat']
        if not pars['refineRT_stereo']:
            raise ValueError('Enter best test results for refineRT_stereo of refinement_ba_stereo')
        args += ['--refineRT_stereo'] + list(map(str, pars['refineRT_stereo']))
        if pars['bart_stereo'] is None:
            raise ValueError('Enter best test results for BART_stereo of refinement_ba_stereo')
        args += ['--BART_stereo', str(pars['bart_stereo'])]
        if pars['minPtsDistance'] is None:
            raise ValueError('Enter best test results for minPtsDistance of correspondence_pool')
        args += ['--minPtsDistance', str(pars['minPtsDistance'])]
        if pars['maxPoolCorrespondences'] is None:
            raise ValueError('Enter best test results for maxPoolCorrespondences of correspondence_pool')
        args += ['--maxPoolCorrespondences', str(pars['maxPoolCorrespondences'])]
        warnings.warn("Warning: Are you sure the selected number of maxPoolCorrespondences is small enough "
                      "to not reach an overall runtime of 0.8s per frame?")
        time.sleep(5.0)
        if pars['maxRat3DPtsFar'] is None:
            raise ValueError('Enter best test results for maxRat3DPtsFar of correspondence_pool')
        args += ['--maxRat3DPtsFar', str(pars['maxRat3DPtsFar'])]
        if pars['maxDist3DPtsZ'] is None:
            raise ValueError('Enter best test results for maxDist3DPtsZ of correspondence_pool')
        args += ['--maxDist3DPtsZ', str(pars['maxDist3DPtsZ'])]
        args += ['--minStartAggInlRat', '0.2']
        if not test_nr:
            raise ValueError('test_nr is required for correspondence_pool')
        if test_nr == 1:
            warnings.warn("Warning: Are you sure you have selected the SMALL dataset for testing?")
            time.sleep(5.0)
            args += ['--absThRankingStable', '0.1']
            args += ['--nr_keypoints', '30to500']
            args += ['--relInlRatThLast', '0.1', '0.5', '0.1']
            args += ['--relInlRatThNew', '0.05', '0.55', '0.1']
            args += ['--minInlierRatSkip', '0.1', '0.55', '0.075']
            args += ['--relMinInlierRatSkip', '0.4', '0.8', '0.1']#This test was forgotten in the Test-Description
            args += ['--minInlierRatioReInit', '0.3', '0.8', '0.1']
        elif test_nr == 2:
            warnings.warn("Warning: Are you sure you have selected the SMALL dataset for testing?")
            time.sleep(5.0)
            args += ['--absThRankingStable', '0.1']
            args += ['--nr_keypoints', '30to500']
            if pars['relInlRatThLast'] is None:
                raise ValueError('Enter best test results for relInlRatThLast of robustness')
            args += ['--relInlRatThLast', str(pars['relInlRatThLast'])]
            if pars['relInlRatThNew'] is None:
                raise ValueError('Enter best test results for relInlRatThNew of robustness')
            args += ['--relInlRatThNew', str(pars['relInlRatThNew'])]
            if pars['minInlierRatSkip'] is None:
                raise ValueError('Enter best test results for minInlierRatSkip of robustness')
            args += ['--minInlierRatSkip', str(pars['minInlierRatSkip'])]
            if pars['relMinInlierRatSkip'] is None:
                raise ValueError('Enter best test results for relMinInlierRatSkip of robustness')
            args += ['--relMinInlierRatSkip', str(pars['relMinInlierRatSkip'])]
            if pars['minInlierRatioReInit'] is None:
                raise ValueError('Enter best test results for minInlierRatioReInit of robustness')
            args += ['--minInlierRatioReInit', str(pars['minInlierRatioReInit'])]
            args += ['--checkPoolPoseRobust', '0', '5', '1']
        elif test_nr == 3:
            warnings.warn("Warning: Are you sure you have selected the SMALL dataset for testing?")
            time.sleep(5.0)
            args += ['--nr_keypoints', '30to500']
            if pars['relInlRatThLast'] is None:
                raise ValueError('Enter best test results for relInlRatThLast of robustness')
            args += ['--relInlRatThLast', str(pars['relInlRatThLast'])]
            if pars['relInlRatThNew'] is None:
                raise ValueError('Enter best test results for relInlRatThNew of robustness')
            args += ['--relInlRatThNew', str(pars['relInlRatThNew'])]
            if pars['minInlierRatSkip'] is None:
                raise ValueError('Enter best test results for minInlierRatSkip of robustness')
            args += ['--minInlierRatSkip', str(pars['minInlierRatSkip'])]
            if pars['relMinInlierRatSkip'] is None:
                raise ValueError('Enter best test results for relMinInlierRatSkip of robustness')
            args += ['--relMinInlierRatSkip', str(pars['relMinInlierRatSkip'])]
            if pars['minInlierRatioReInit'] is None:
                raise ValueError('Enter best test results for minInlierRatioReInit of robustness')
            args += ['--minInlierRatioReInit', str(pars['minInlierRatioReInit'])]
            if pars['checkPoolPoseRobust'] is None:
                raise ValueError('Enter best test results for checkPoolPoseRobust of robustness')
            args += ['--checkPoolPoseRobust', str(pars['checkPoolPoseRobust'])]
        elif test_nr == 4:
            warnings.warn("Warning: Are you sure you have selected the SMALL dataset for testing?")
            time.sleep(5.0)
            args += ['--nr_keypoints', '30to500']
            if pars['relInlRatThLast'] is None:
                raise ValueError('Enter best test results for relInlRatThLast of robustness')
            args += ['--relInlRatThLast', str(pars['relInlRatThLast'])]
            if pars['relInlRatThNew'] is None:
                raise ValueError('Enter best test results for relInlRatThNew of robustness')
            args += ['--relInlRatThNew', str(pars['relInlRatThNew'])]
            if pars['minInlierRatSkip'] is None:
                raise ValueError('Enter best test results for minInlierRatSkip of robustness')
            args += ['--minInlierRatSkip', str(pars['minInlierRatSkip'])]
            if pars['relMinInlierRatSkip'] is None:
                raise ValueError('Enter best test results for relMinInlierRatSkip of robustness')
            args += ['--relMinInlierRatSkip', str(pars['relMinInlierRatSkip'])]
            if pars['minInlierRatioReInit'] is None:
                raise ValueError('Enter best test results for minInlierRatioReInit of robustness')
            args += ['--minInlierRatioReInit', str(pars['minInlierRatioReInit'])]
            if pars['checkPoolPoseRobust'] is None:
                raise ValueError('Enter best test results for checkPoolPoseRobust of robustness')
            args += ['--checkPoolPoseRobust', str(pars['checkPoolPoseRobust'])]
            args += ['--minContStablePoses', '3', '5', '1']
            args += ['--minNormDistStable', '0.25', '0.75', '0.1']
            args += ['--absThRankingStable', '0.05', '0.5', '0.075']
        elif test_nr == 5:
            warnings.warn("Warning: Are you sure you have selected the SMALL dataset for testing?")
            time.sleep(5.0)
            args += ['--nr_keypoints', '20to160']
            if pars['relInlRatThLast'] is None:
                raise ValueError('Enter best test results for relInlRatThLast of robustness')
            args += ['--relInlRatThLast', str(pars['relInlRatThLast'])]
            if pars['relInlRatThNew'] is None:
                raise ValueError('Enter best test results for relInlRatThNew of robustness')
            args += ['--relInlRatThNew', str(pars['relInlRatThNew'])]
            if pars['minInlierRatSkip'] is None:
                raise ValueError('Enter best test results for minInlierRatSkip of robustness')
            args += ['--minInlierRatSkip', str(pars['minInlierRatSkip'])]
            if pars['relMinInlierRatSkip'] is None:
                raise ValueError('Enter best test results for relMinInlierRatSkip of robustness')
            args += ['--relMinInlierRatSkip', str(pars['relMinInlierRatSkip'])]
            if pars['minInlierRatioReInit'] is None:
                raise ValueError('Enter best test results for minInlierRatioReInit of robustness')
            args += ['--minInlierRatioReInit', str(pars['minInlierRatioReInit'])]
            if pars['checkPoolPoseRobust'] is None:
                raise ValueError('Enter best test results for checkPoolPoseRobust of robustness')
            args += ['--checkPoolPoseRobust', str(pars['checkPoolPoseRobust'])]
            if pars['minContStablePoses'] is None:
                raise ValueError('Enter best test results for minContStablePoses of robustness')
            args += ['--minContStablePoses', str(pars['minContStablePoses'])]
            if pars['minNormDistStable'] is None:
                raise ValueError('Enter best test results for minNormDistStable of robustness')
            args += ['--minNormDistStable', str(pars['minNormDistStable'])]
            if pars['absThRankingStable'] is None:
                raise ValueError('Enter best test results for absThRankingStable of robustness')
            args += ['--absThRankingStable', str(pars['absThRankingStable'])]
            args += ['--useRANSAC_fewMatches', str(2)]
        elif test_nr == 6:
            warnings.warn("Warning: Are you sure you have selected the LARGE dataset for testing?")
            time.sleep(5.0)
            if pars['relInlRatThLast'] is None:
                raise ValueError('Enter best test results for relInlRatThLast of robustness')
            args += ['--relInlRatThLast', str(pars['relInlRatThLast'])]
            if pars['relInlRatThNew'] is None:
                raise ValueError('Enter best test results for relInlRatThNew of robustness')
            args += ['--relInlRatThNew', str(pars['relInlRatThNew'])]
            if pars['minInlierRatSkip'] is None:
                raise ValueError('Enter best test results for minInlierRatSkip of robustness')
            args += ['--minInlierRatSkip', str(pars['minInlierRatSkip'])]
            if pars['relMinInlierRatSkip'] is None:
                raise ValueError('Enter best test results for relMinInlierRatSkip of robustness')
            args += ['--relMinInlierRatSkip', str(pars['relMinInlierRatSkip'])]
            if pars['minInlierRatioReInit'] is None:
                raise ValueError('Enter best test results for minInlierRatioReInit of robustness')
            args += ['--minInlierRatioReInit', str(pars['minInlierRatioReInit'])]
            if pars['checkPoolPoseRobust'] is None:
                raise ValueError('Enter best test results for checkPoolPoseRobust of robustness')
            args += ['--checkPoolPoseRobust', str(pars['checkPoolPoseRobust'])]
            if pars['minContStablePoses'] is None:
                raise ValueError('Enter best test results for minContStablePoses of robustness')
            args += ['--minContStablePoses', str(pars['minContStablePoses'])]
            if pars['minNormDistStable'] is None:
                raise ValueError('Enter best test results for minNormDistStable of robustness')
            args += ['--minNormDistStable', str(pars['minNormDistStable'])]
            if pars['absThRankingStable'] is None:
                raise ValueError('Enter best test results for absThRankingStable of robustness')
            args += ['--absThRankingStable', str(pars['absThRankingStable'])]
            if pars['useRANSAC_fewMatches'] is None:
                raise ValueError('Enter best test result for useRANSAC_fewMatches of robustness')
            if pars['useRANSAC_fewMatches']:
                args += ['--useRANSAC_fewMatches']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for robustness')
    elif test_name == 'usac_vs_autocalib':
        args += ['--RobMethod', 'USAC']
        if not pars['usac56'] or not pars['usac123']:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, pars['usac123'])) + ['0'] + list(map(str, pars['usac56']))
        if pars['USACInlratFilt'] is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(pars['USACInlratFilt'])]
        if pars['th'] is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(pars['th'])]
        if not pars['refineRT']:
            raise ValueError('Enter best test results for refineRT of refinement_ba')
        args += ['--refineRT'] + list(map(str, pars['refineRT']))
        if pars['bart'] is None:
            raise ValueError('Enter best test results for BART of refinement_ba')
        args += ['--BART', str(pars['bart'])]
        args += ['--stereoRef']
        args += ['--useGTCamMat']
        if not pars['refineRT_stereo']:
            raise ValueError('Enter best test results for refineRT_stereo of refinement_ba_stereo')
        args += ['--refineRT_stereo'] + list(map(str, pars['refineRT_stereo']))
        if pars['bart_stereo'] is None:
            raise ValueError('Enter best test results for BART_stereo of refinement_ba_stereo')
        args += ['--BART_stereo', str(pars['bart_stereo'])]
        if pars['minPtsDistance'] is None:
            raise ValueError('Enter best test results for minPtsDistance of correspondence_pool')
        args += ['--minPtsDistance', str(pars['minPtsDistance'])]
        if pars['maxPoolCorrespondences'] is None:
            raise ValueError('Enter best test results for maxPoolCorrespondences of correspondence_pool')
        args += ['--maxPoolCorrespondences', str(pars['maxPoolCorrespondences'])]
        if pars['maxRat3DPtsFar'] is None:
            raise ValueError('Enter best test results for maxRat3DPtsFar of correspondence_pool')
        args += ['--maxRat3DPtsFar', str(pars['maxRat3DPtsFar'])]
        if pars['maxDist3DPtsZ'] is None:
            raise ValueError('Enter best test results for maxDist3DPtsZ of correspondence_pool')
        args += ['--maxDist3DPtsZ', str(pars['maxDist3DPtsZ'])]
        args += ['--minStartAggInlRat', '0.07']
        if pars['relInlRatThLast'] is None:
            raise ValueError('Enter best test results for relInlRatThLast of robustness')
        args += ['--relInlRatThLast', str(pars['relInlRatThLast'])]
        if pars['relInlRatThNew'] is None:
            raise ValueError('Enter best test results for relInlRatThNew of robustness')
        args += ['--relInlRatThNew', str(pars['relInlRatThNew'])]
        if pars['minInlierRatSkip'] is None:
            raise ValueError('Enter best test results for minInlierRatSkip of robustness')
        args += ['--minInlierRatSkip', str(pars['minInlierRatSkip'])]
        if pars['relMinInlierRatSkip'] is None:
            raise ValueError('Enter best test results for relMinInlierRatSkip of robustness')
        args += ['--relMinInlierRatSkip', str(pars['relMinInlierRatSkip'])]
        if pars['minInlierRatioReInit'] is None:
            raise ValueError('Enter best test results for minInlierRatioReInit of robustness')
        args += ['--minInlierRatioReInit', str(pars['minInlierRatioReInit'])]
        if pars['checkPoolPoseRobust'] is None:
            raise ValueError('Enter best test results for checkPoolPoseRobust of robustness')
        args += ['--checkPoolPoseRobust', str(pars['checkPoolPoseRobust'])]
        if pars['minContStablePoses'] is None:
            raise ValueError('Enter best test results for minContStablePoses of robustness')
        args += ['--minContStablePoses', str(pars['minContStablePoses'])]
        if pars['minNormDistStable'] is None:
            raise ValueError('Enter best test results for minNormDistStable of robustness')
        args += ['--minNormDistStable', str(pars['minNormDistStable'])]
        if pars['absThRankingStable'] is None:
            raise ValueError('Enter best test results for absThRankingStable of robustness')
        args += ['--absThRankingStable', str(pars['absThRankingStable'])]
        if pars['useRANSAC_fewMatches'] is None:
            raise ValueError('Enter best test result for useRANSAC_fewMatches of robustness')
        if pars['useRANSAC_fewMatches']:
            args += ['--useRANSAC_fewMatches']
        args += ['--accumCorrs', '1', '5', '1', '10', '20', '5']
        args += ['--accumCorrsCompare']
    elif test_name == 'testing_tests' and testing_test:#Only for testing the script exec_autocalib.py
        if not test_nr:
            raise ValueError('test_nr is required for testing the test interface')
        if test_nr == 1:
            args += ['--refineRT', '0', '0']
            args += ['--RobMethod', 'USAC']
            args += ['--th', '0.6', '2.0', '0.2']
            args += ['--useGTCamMat']
            args += ['--inlier_ratios', '0.2', '0.4', '0.6', '0.8']
            args += ['--kp_accs', '1.5']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5', '0', '0', '0', '0', '1', '1']
            args += ['--USACInlratFilt', '0']
        elif test_nr == 2:
            args += ['--refineRT', '0', '0']
            args += ['--RobMethod', 'USAC']
            args += ['--th', '0.6', '2.0', '0.2']
            args += ['--useGTCamMat']
            args += ['--depths', 'NMF']
            args += ['--nr_keypoints', '500']
            args += ['--kp_pos_distr', 'equ']
            pars['usac56'] = [2, 5]
            if not pars['usac56']:
                raise ValueError('Enter best test results for parameters 5 & 6 of usac-testing')
            args += ['--cfgUSAC', '3', '1', '1', '0'] + list(map(str, pars['usac56'])) + ['1', '1', '1', '0', '0', '0']
            args += ['--USACInlratFilt', '2']
        elif test_nr == 3:
            args += ['--refineRT', '0', '0']
            args += ['--RobMethod', 'USAC', 'RANSAC']
            args += ['--th', '0.6', '2.0', '0.2']
            args += ['--useGTCamMat']
            pars['usac56'] = [2, 5]
            pars['usac123'] = [3,1,1]
            if not pars['usac56'] or not pars['usac123']:
                raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
            args += ['--cfgUSAC'] + list(map(str, pars['usac123'])) + ['0'] + list(map(str, pars['usac56']))
            pars['USACInlratFilt'] = 0
            if pars['USACInlratFilt'] is None:
                raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
            args += ['--USACInlratFilt', str(pars['USACInlratFilt'])]
        elif test_nr == 4:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '0', '0', '1', '1', '-13']
            args += ['--BART', '0', '1']
            args += ['--useGTCamMat']
        elif test_nr == 5:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            pars['refineRT'] = [4,2]
            args += ['--refineRT'] + list(map(str, pars['refineRT']))
            args += ['--BART', '0', '1']
            args += ['--nr_keypoints', '500']
        elif test_nr == 6:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '1', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '0', '0']
            args += ['--useGTCamMat']
            args += ['--refineVFC']
            args += ['--refineSOF']
            args += ['--refineGMS']
        elif test_nr == 7:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '4', '2']
            args += ['--BART', '1']
            args += ['--stereoRef']
            args += ['--minStartAggInlRat', '0.075']
            args += ['--nr_keypoints', '500']
            args += ['--maxPoolCorrespondences', '5000']
            args += ['--minInlierRatSkip', '0.075']
            args += ['--refineRT_stereo', '0', '0', '1', '1', '-13']
            args += ['--BART_stereo', '0', '1']
            args += ['--useGTCamMat']
        elif test_nr == 8:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '4', '2']
            args += ['--BART', '1']
            args += ['--stereoRef']
            args += ['--minStartAggInlRat', '0.075']
            args += ['--nr_keypoints', '500']
            args += ['--maxPoolCorrespondences', '5000']
            args += ['--minInlierRatSkip', '0.075']
            args += ['--refineRT_stereo', '4', '2']
            args += ['--BART_stereo', '0', '1']
        elif test_nr == 9:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '4', '2']
            args += ['--BART', '0']
            args += ['--stereoRef']
            args += ['--minStartAggInlRat', '0.075']
            args += ['--minInlierRatSkip', '0.075']
            args += ['--useGTCamMat']
            args += ['--refineRT_stereo', '4', '2']
            args += ['--BART_stereo', str(1)]
            args += ['--minPtsDistance', '1.5', '15.5', '2.0']
            args += ['--maxPoolCorrespondences', '300', '1000', '100', '1000', '2000', '200', '2000', '5000', '500',
                     '5000', '10000', '1000', '10000', '20000', '2000', '20000', '30000', '5000']
        elif test_nr == 10:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '4', '2']
            args += ['--BART', '0']
            args += ['--stereoRef']
            args += ['--minStartAggInlRat', '0.075']
            args += ['--minInlierRatSkip', '0.075']
            args += ['--useGTCamMat']
            args += ['--refineRT_stereo', '4', '2']
            args += ['--BART_stereo', '1']
            args += ['--minPtsDistance', '3.0']
            args += ['--maxPoolCorrespondences', '5000']
            args += ['--maxRat3DPtsFar', '0.1', '0.8', '0.1']
            args += ['--maxDist3DPtsZ', '20.0', '300.0', '20.0']
        elif test_nr == 11:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '4', '2']
            args += ['--BART', '0']
            args += ['--stereoRef']
            args += ['--minStartAggInlRat', '0.075']
            args += ['--minInlierRatSkip', '0.075']
            args += ['--useGTCamMat']
            args += ['--refineRT_stereo', '4', '2']
            args += ['--BART_stereo', '1']
            warnings.warn("Warning: Be sure to use the scenes from testing USAC")
            time.sleep(5.0)
            args += ['--depths', 'NMF']
            args += ['--nr_keypoints', '500']
            args += ['--kp_pos_distr', '1corn']
            args += ['--inlier_ratios', '0.6']
            args += ['--minPtsDistance', '3.0']
            args += ['--maxPoolCorrespondences', '5000']
            args += ['--maxRat3DPtsFar', '0.5']
            args += ['--maxDist3DPtsZ', '50.0']
        elif test_nr == 12:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '4', '2']
            args += ['--BART', '0']
            args += ['--stereoRef']
            args += ['--useGTCamMat']
            args += ['--refineRT_stereo', '4', '2']
            args += ['--BART_stereo', '1']
            args += ['--minPtsDistance', '3.0']
            args += ['--maxPoolCorrespondences', '5000']
            warnings.warn("Warning: Are you sure the selected number of maxPoolCorrespondences is small enough "
                          "to not reach an overall runtime of 0.8s per frame?")
            time.sleep(5.0)
            args += ['--maxRat3DPtsFar', '0.5']
            args += ['--maxDist3DPtsZ', '50.0']
            args += ['--minStartAggInlRat', '0.2']
            warnings.warn("Warning: Are you sure you have selected the SMALL dataset for testing?")
            time.sleep(5.0)
            args += ['--absThRankingStable', '0.1']
            args += ['--nr_keypoints', '30to500']#Comment later on
            args += ['--relInlRatThLast', '0.1', '0.5', '0.1']
            args += ['--relInlRatThNew', '0.05', '0.55', '0.1']
            args += ['--minInlierRatSkip', '0.1', '0.55', '0.075']
            args += ['--relMinInlierRatSkip', '0.4', '0.8', '0.1']  # This test was forgotten in the Test-Description
            args += ['--minInlierRatioReInit', '0.3', '0.8', '0.1']
        elif test_nr == 13:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '4', '2']
            args += ['--BART', '0']
            args += ['--stereoRef']
            args += ['--useGTCamMat']
            args += ['--refineRT_stereo', '4', '2']
            args += ['--BART_stereo', '1']
            args += ['--minPtsDistance', '3.0']
            args += ['--maxPoolCorrespondences', '5000']
            args += ['--maxRat3DPtsFar', '0.5']
            args += ['--maxDist3DPtsZ', '50.0']
            args += ['--minStartAggInlRat', '0.2']
            args += ['--absThRankingStable', '0.1']
            #args += ['--nr_keypoints', '30to500']
            args += ['--relInlRatThLast', '0.35']
            args += ['--relInlRatThNew', '0.2']
            args += ['--minInlierRatSkip', '0.3']
            args += ['--relMinInlierRatSkip', '0.7']
            args += ['--minInlierRatioReInit', '0.65']
            args += ['--checkPoolPoseRobust', '0', '5', '1']
        elif test_nr == 14:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '4', '2']
            args += ['--BART', '0']
            args += ['--stereoRef']
            args += ['--useGTCamMat']
            args += ['--refineRT_stereo', '4', '2']
            args += ['--BART_stereo', '1']
            args += ['--minPtsDistance', '3.0']
            args += ['--maxPoolCorrespondences', '5000']
            args += ['--maxRat3DPtsFar', '0.5']
            args += ['--maxDist3DPtsZ', '50.0']
            args += ['--minStartAggInlRat', '0.2']
            #args += ['--nr_keypoints', '30to500']
            args += ['--relInlRatThLast', '0.35']
            args += ['--relInlRatThNew', '0.2']
            args += ['--minInlierRatSkip', '0.3']
            args += ['--relMinInlierRatSkip', '0.7']
            args += ['--minInlierRatioReInit', '0.65']
            args += ['--checkPoolPoseRobust', '3']
            args += ['--minContStablePoses', '3', '5', '1']
            args += ['--minNormDistStable', '0.25', '0.75', '0.1']
            args += ['--absThRankingStable', '0.05', '0.5', '0.075']
        elif test_nr == 15:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '4', '2']
            args += ['--BART', '0']
            args += ['--stereoRef']
            args += ['--useGTCamMat']
            args += ['--refineRT_stereo', '4', '2']
            args += ['--BART_stereo', '1']
            args += ['--minPtsDistance', '3.0']
            args += ['--maxPoolCorrespondences', '5000']
            args += ['--maxRat3DPtsFar', '0.5']
            args += ['--maxDist3DPtsZ', '50.0']
            args += ['--minStartAggInlRat', '0.2']
            #args += ['--nr_keypoints', '20to160']
            args += ['--relInlRatThLast', '0.35']
            args += ['--relInlRatThNew', '0.2']
            args += ['--minInlierRatSkip', '0.3']
            args += ['--relMinInlierRatSkip', '0.7']
            args += ['--minInlierRatioReInit', '0.65']
            args += ['--checkPoolPoseRobust', '3']
            args += ['--minContStablePoses', '5']
            args += ['--minNormDistStable', '0.5']
            args += ['--absThRankingStable', '0.25']
            args += ['--useRANSAC_fewMatches', '2']
        elif test_nr == 16:
            args += ['--RobMethod', 'USAC']
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
            args += ['--USACInlratFilt', '0']
            args += ['--th', '0.85']
            args += ['--refineRT', '4', '2']
            args += ['--BART', '0']
            args += ['--stereoRef']
            args += ['--useGTCamMat']
            args += ['--refineRT_stereo', '4', '2']
            args += ['--BART_stereo', '1']
            args += ['--minPtsDistance', '3.0']
            args += ['--maxPoolCorrespondences', '5000']
            args += ['--maxRat3DPtsFar', '0.5']
            args += ['--maxDist3DPtsZ', '50.0']
            args += ['--minStartAggInlRat', '0.2']
            # args += ['--nr_keypoints', '20to160']
            args += ['--relInlRatThLast', '0.35']
            args += ['--relInlRatThNew', '0.2']
            args += ['--minInlierRatSkip', '0.3']
            args += ['--relMinInlierRatSkip', '0.7']
            args += ['--minInlierRatioReInit', '0.65']
            args += ['--checkPoolPoseRobust', '3']
            args += ['--minContStablePoses', '5']
            args += ['--minNormDistStable', '0.5']
            args += ['--absThRankingStable', '0.25']
            args += ['--useRANSAC_fewMatches']
            args += ['--accumCorrs', '1', '5', '1', '10', '20', '5']
            args += ['--accumCorrsCompare']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for testing_tests')
    else:
        raise ValueError('test_name ' + test_name + ' is not supported')

    output_path = os.path.join(output_path, test_name)
    try:
        os.mkdir(output_path)
    except FileExistsError:
        print('Directory ' + output_path + ' already exists')
    if test_nr:
        output_path = os.path.join(output_path, str(test_nr))
        try:
            os.mkdir(output_path)
        except FileExistsError:
            print('Directory ' + output_path + ' already exists')
    args += ['--output_path', output_path]

    pyfilepath = os.path.dirname(os.path.realpath(__file__))
    pyfilename = os.path.join(pyfilepath, 'exec_autocalib.py')
    try:
        cmdline = ['python', pyfilename] + args
        retcode = sp.run(cmdline, shell=False, check=True).returncode
        if retcode != 0:
            print("Child was terminated by signal", retcode, file=sys.stderr)
        else:
            print("Child returned", -retcode, file=sys.stderr)
    except sp.CalledProcessError as e:
        print("Execution failed:", e, file=sys.stderr)
        retcode = 4
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)
        retcode = 4
    except:
        print('Unknown exception:', file=sys.stderr)
        e = sys.exc_info()
        print(str(e), file=sys.stderr)
        retcode = 5

    return retcode


def write_par_file_template(path):
    pfile = os.path.join(path, 'optimal_autocalib_pars.yml')
    if os.path.exists(pfile):
        raise ValueError('Parameter file already exists')
    from usac_eval import NoAliasDumper

    usac56 = [None, None] #[2, 5]
    usac123 = [None, None, None] #[3, 1, 1]
    USACInlratFilt = None #0
    th = None #0.85
    refineRT = [None, None] #[4, 2]
    robMFilt = None #'USAC'
    bart = None #1
    refineRT_stereo = [None, None] #[4, 2]
    bart_stereo = None #1
    minPtsDistance = None #3.0
    maxPoolCorrespondences = None #8000
    maxRat3DPtsFar = None #0.5
    maxDist3DPtsZ = None #50.0
    relInlRatThLast = None #0.35
    relInlRatThNew = None #0.2
    minInlierRatSkip = None #0.38
    relMinInlierRatSkip = None #0.7
    minInlierRatioReInit = None #0.67
    checkPoolPoseRobust = None #3
    minContStablePoses = None #3
    minNormDistStable = None #0.5
    absThRankingStable = None #0.075
    useRANSAC_fewMatches = None #False

    pars = {'usac56': usac56,
            'usac123': usac123,
            'USACInlratFilt': USACInlratFilt,
            'th': th,
            'refineRT': refineRT,
            'robMFilt': robMFilt,
            'bart': bart,
            'refineRT_stereo': refineRT_stereo,
            'bart_stereo': bart_stereo,
            'minPtsDistance': minPtsDistance,
            'maxPoolCorrespondences': maxPoolCorrespondences,
            'maxRat3DPtsFar': maxRat3DPtsFar,
            'maxDist3DPtsZ': maxDist3DPtsZ,
            'relInlRatThLast': relInlRatThLast,
            'relInlRatThNew': relInlRatThNew,
            'minInlierRatSkip': minInlierRatSkip,
            'relMinInlierRatSkip': relMinInlierRatSkip,
            'minInlierRatioReInit': minInlierRatioReInit,
            'checkPoolPoseRobust': checkPoolPoseRobust,
            'minContStablePoses': minContStablePoses,
            'minNormDistStable': minNormDistStable,
            'absThRankingStable': absThRankingStable,
            'useRANSAC_fewMatches': useRANSAC_fewMatches}
    with open(pfile, 'w') as fo:
        yaml.dump(pars, stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    return 0


def read_pars_yaml(path):
    from usac_eval import readYaml
    pfile = os.path.join(path, 'optimal_autocalib_pars.yml')
    if os.path.exists(pfile):
        try:
            ydata = readYaml(pfile)
        except BaseException:
            raise ValueError('Unable to read parameter file')
    else:
        warnings.warn('Parameter file does not exist. Creating a template.', UserWarning)
        try:
            write_par_file_template(path)
        except ValueError:
            raise ValueError('Parameter file exists but cannot be read.')
        ydata = readYaml(pfile)
    return ydata


def read_pars(path, pars_list):
    data = read_pars_yaml(path)
    ret = {}
    for i in pars_list:
        try:
            ret[i] = data[i]
        except:
            warnings.warn('Parameter ' + i + ' not found in YAML file. Setting it not none', UserWarning)
            ret[i] = None
    for i in ret.items():
        if isinstance(i[1], list) and None in i[1]:
            ret[i[0]] = None
    return ret


def write_parameters(path, pars):
    if not isinstance(pars, dict):
        raise ValueError('Parameters must be provided in dict format')
    data = read_pars_yaml(path)
    for i in pars.items():
        data[i[0]] = i[1]
    pfile = os.path.join(path, 'optimal_autocalib_pars.yml')
    from usac_eval import NoAliasDumper
    with open(pfile, 'w') as fo:
        yaml.dump(data, stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    return 0


def convert_mult_autoc_pars_out_to_in(pars):
    if not isinstance(pars, dict):
        raise ValueError('Wrong input format for converting parameter names')
    grps = [['USAC_parameters_automaticSprtInit',
             'USAC_parameters_automaticProsacParameters',
             'USAC_parameters_prevalidateSample'],
            ['USAC_parameters_estimator',
             'USAC_parameters_refinealg'],
            ['refineMethod_algorithm',
             'refineMethod_costFunction'],
            ['stereoParameters_refineMethod_CorrPool_algorithm',
             'stereoParameters_refineMethod_CorrPool_costFunction']]
    pars_k = pars.keys()
    dellist = []
    res_list = []
    res_pars = []
    for grp in grps:
        cnt = 0
        for elem in grp:
            for key in pars_k:
                if key == elem:
                    if cnt == 0:
                        res_list.append([key])
                        res_pars.append([pars[key]])
                    else:
                        res_list[-1].append(key)
                        res_pars[-1].append(pars[key])
                    cnt += 1
                    dellist.append(key)
        if cnt > 0 and len(grp) != cnt:
            raise ValueError('Insufficient parameter names for conversion')
    for key in pars_k:
        if key not in dellist:
            res_list.append(key)
            res_pars.append(pars[key])
    pars_out = {}
    for name, value in zip(res_list, res_pars):
        pars_out.update(convert_autoc_pars_out_to_in(name, value))
    return pars_out


def convert_autoc_pars_out_to_in(par_name, par_value):
    if isinstance(par_name, list):
        ret = []
        main_par = None
        for i, j in zip(par_name, par_value):
            if not isinstance(i, str) or not isinstance(j, str):
                raise ValueError('Input parameters must be of type str')
            if i == 'USAC_parameters_automaticSprtInit':
                if not ret:
                    ret = [None, None, None]
                    main_par = 'usac123'
                if j == 'SPRT_DEFAULT_INIT':
                    ret[0] = 0
                elif j == 'SPRT_DELTA_AUTOM_INIT':
                    ret[0] = 1
                elif j == 'SPRT_EPSILON_AUTOM_INIT':
                    ret[0] = 2
                elif j == 'SPRT_DELTA_AND_EPSILON_AUTOM_INIT':
                    ret[0] = 3
                else:
                    raise ValueError('Invalid value \'' + j + '\' for parameter ' + i)
            elif i == 'USAC_parameters_automaticProsacParameters':
                if not ret:
                    ret = [None, None, None]
                    main_par = 'usac123'
                if j == 'enabled':
                    ret[1] = 1
                elif j == 'disabled':
                    ret[1] = 0
                else:
                    raise ValueError('Invalid value \'' + j + '\' for parameter ' + i)
            elif i == 'USAC_parameters_prevalidateSample':
                if not ret:
                    ret = [None, None, None]
                    main_par = 'usac123'
                if j == 'enabled':
                    ret[2] = 1
                elif j == 'disabled':
                    ret[2] = 0
                else:
                    raise ValueError('Invalid value \'' + j + '\' for parameter ' + i)
            elif i == 'USAC_parameters_estimator':
                if not ret:
                    ret = [None, None]
                    main_par = 'usac56'
                if j == 'POSE_NISTER':
                    ret[0] = 0
                elif j == 'POSE_EIG_KNEIP':
                    ret[0] = 1
                elif j == 'POSE_STEWENIUS':
                    ret[0] = 2
                else:
                    raise ValueError('Invalid value \'' + j + '\' for parameter ' + i)
            elif i == 'USAC_parameters_refinealg':
                if not ret:
                    ret = [None, None]
                    main_par = 'usac56'
                if j == 'REF_WEIGHTS':
                    ret[1] = 0
                elif j == 'REF_8PT_PSEUDOHUBER':
                    ret[1] = 1
                elif j == 'REF_EIG_KNEIP':
                    ret[1] = 2
                elif j == 'REF_EIG_KNEIP_WEIGHTS':
                    ret[1] = 3
                elif j == 'REF_STEWENIUS':
                    ret[1] = 4
                elif j == 'REF_STEWENIUS_WEIGHTS':
                    ret[1] = 5
                elif j == 'REF_NISTER':
                    ret[1] = 6
                elif j == 'REF_NISTER_WEIGHTS':
                    ret[1] = 7
                else:
                    raise ValueError('Invalid value \'' + j + '\' for parameter ' + i)
            elif i == 'refineMethod_algorithm' or i == 'stereoParameters_refineMethod_algorithm':
                if not ret:
                    ret = [None, None]
                    main_par = 'refineRT'
                if j == 'PR_NO_REFINEMENT':
                    ret[0] = 0
                elif j == 'PR_8PT':
                    ret[0] = 2
                elif j == 'PR_NISTER':
                    ret[0] = 3
                elif j == 'PR_STEWENIUS':
                    ret[0] = 4
                elif j == 'PR_KNEIP':
                    ret[0] = 5
                else:
                    raise ValueError('Invalid value \'' + j + '\' for parameter ' + i)
            elif i == 'refineMethod_costFunction' or i == 'stereoParameters_refineMethod_costFunction':
                if not ret:
                    ret = [None, None]
                    main_par = 'refineRT'
                if j == 'PR_TORR_WEIGHTS':
                    ret[1] = 1
                elif j == 'PR_PSEUDOHUBER_WEIGHTS':
                    ret[1] = 2
                elif j == 'PR_NO_WEIGHTS':
                    ret[1] = 0
                else:
                    raise ValueError('Invalid value \'' + j + '\' for parameter ' + i)
            elif i == 'stereoParameters_refineMethod_CorrPool_algorithm':
                if not ret:
                    ret = [None, None]
                    main_par = 'refineRT_stereo'
                if j == 'PR_NO_REFINEMENT':
                    ret[0] = 0
                elif j == 'PR_8PT':
                    ret[0] = 2
                elif j == 'PR_NISTER':
                    ret[0] = 3
                elif j == 'PR_STEWENIUS':
                    ret[0] = 4
                elif j == 'PR_KNEIP':
                    ret[0] = 5
                else:
                    raise ValueError('Invalid value \'' + j + '\' for parameter ' + i)
            elif i == 'stereoParameters_refineMethod_CorrPool_costFunction':
                if not ret:
                    ret = [None, None]
                    main_par = 'refineRT_stereo'
                if j == 'PR_TORR_WEIGHTS':
                    ret[1] = 1
                elif j == 'PR_PSEUDOHUBER_WEIGHTS':
                    ret[1] = 2
                elif j == 'PR_NO_WEIGHTS':
                    ret[1] = 0
                else:
                    raise ValueError('Invalid value \'' + j + '\' for parameter ' + i)
            else:
                raise ValueError('Unknown parameter ' + i)
        if not main_par:
            raise ValueError('Main parameter name not found.')
        for i in ret:
            if i is None:
                raise ValueError('Invalid number of parameters provided for ' + main_par)
        return {main_par: ret}
    elif isinstance(par_name, str):
        if par_name == 'th' or par_name == 'USAC_parameters_th_pixels' or par_name == 'stereoParameters_th_pix_user':
            if not isinstance(par_value, float):
                raise ValueError('Threshold must be a float value')
            return {'th': par_value}
        elif par_name == 'RobMethod' or par_name == 'stereoParameters_RobMethod':
            if par_value not in ['USAC', 'RANSAC']:
                raise ValueError('Invalid value \'' + par_value + '\' for parameter ' + par_name)
            return {'robMFilt': par_value}
        elif par_name == 'USAC_parameters_USACInlratFilt':
            if not (par_value == 'GMS' or par_value == 'VFC'):
                raise ValueError('Invalid value \'' + par_value + '\' for parameter ' + par_name)
            if par_value == 'GMS':
                return {'USACInlratFilt': 0}
            else:
                return {'USACInlratFilt': 1}
        elif par_name == 'BART' or par_name == 'stereoParameters_BART':
            if par_value == 'disabled':
                return {'bart': 0}
            elif par_value == 'extr_only':
                return {'bart': 1}
            elif par_value == 'extr_intr':
                return {'bart': 2}
            else:
                raise ValueError('Invalid value \'' + par_value + '\' for parameter ' + par_name)
        elif par_name == 'stereoParameters_BART_CorrPool':
            if par_value == 'disabled':
                return {'bart_stereo': 0}
            elif par_value == 'extr_only':
                return {'bart_stereo': 1}
            elif par_value == 'extr_intr':
                return {'bart_stereo': 2}
            else:
                raise ValueError('Invalid value \'' + par_value + '\' for parameter ' + par_name)
        elif par_name == 'stereoParameters_checkPoolPoseRobust':
            if isinstance(par_value, float):
                par_value = int(par_value)
            elif not isinstance(par_value, int):
                raise ValueError('Parameter ' + par_name + ' must be of type float or int')
            return {'checkPoolPoseRobust': par_value}
        elif par_name == 'stereoParameters_useRANSAC_fewMatches':
            if par_value == 'enabled':
                return {'useRANSAC_fewMatches': True}
            elif par_value == 'disabled':
                return {'useRANSAC_fewMatches': False}
            else:
                raise ValueError('Invalid value \'' + par_value + '\' for parameter ' + par_name)
        elif par_name == 'stereoParameters_maxPoolCorrespondences':
            if isinstance(par_value, float):
                par_value = int(par_value)
            elif not isinstance(par_value, int):
                raise ValueError('Parameter ' + par_name + ' must be of type float or int')
            return {'maxPoolCorrespondences': par_value}
        elif par_name == 'stereoParameters_maxDist3DPtsZ':
            if isinstance(par_value, int):
                par_value = float(par_value)
            elif not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float or int')
            return {'maxDist3DPtsZ': par_value}
        elif par_name == 'stereoParameters_maxRat3DPtsFar':
            if not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float')
            return {'maxRat3DPtsFar': par_value}
        elif par_name == 'stereoParameters_minStartAggInlRat':
            # Set to a fixed value - no evals for this value
            if not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float')
            return {'minStartAggInlRat': par_value}
        elif par_name == 'stereoParameters_minInlierRatSkip':
            if not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float')
            return {'minInlierRatSkip': par_value}
        elif par_name == 'stereoParameters_relInlRatThLast':
            if not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float')
            return {'relInlRatThLast': par_value}
        elif par_name == 'stereoParameters_relInlRatThNew':
            if not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float')
            return {'relInlRatThNew': par_value}
        elif par_name == 'stereoParameters_relMinInlierRatSkip':
            if not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float')
            return {'relMinInlierRatSkip': par_value}
        elif par_name == 'stereoParameters_minInlierRatioReInit':
            if not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float')
            return {'minInlierRatioReInit': par_value}
        elif par_name == 'stereoParameters_maxSkipPairs':
            # currently not used in test framework
            return {'maxSkipPairs': par_value}
        elif par_name == 'stereoParameters_minNormDistStable':
            if not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float')
            return {'minNormDistStable': par_value}
        elif par_name == 'stereoParameters_absThRankingStable':
            if not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float')
            return {'absThRankingStable': par_value}
        elif par_name == 'stereoParameters_minContStablePoses':
            if isinstance(par_value, float):
                par_value = int(par_value)
            elif not isinstance(par_value, int):
                raise ValueError('Parameter ' + par_name + ' must be of type float or int')
            return {'minContStablePoses': par_value}
        elif par_name == 'stereoParameters_raiseSkipCnt':
            # currently not used in test framework
            return {'raiseSkipCnt': par_value}
        elif par_name == 'stereoParameters_minPtsDistance':
            if isinstance(par_value, int):
                par_value = float(par_value)
            elif not isinstance(par_value, float):
                raise ValueError('Parameter ' + par_name + ' must be of type float or int')
            return {'minPtsDistance': par_value}
        else:
            raise ValueError('Unknown parameter ' + par_name)
    else:
        raise ValueError('Type of parameter ' + par_name + ' must be str or list')


def convert_autoc_pars_in_to_out(par_name, par_value):
    if par_name == 'usac123':
        if not isinstance(par_value, list) or len(par_value) != 3:
            raise ValueError('Wrong input parameters for usac123')
        pars_out = {}
        for idx, i in enumerate(par_value):
            if idx == 0:
                if i == 0:
                    pars_out['USAC_parameters_automaticSprtInit'] = 'SPRT_DEFAULT_INIT'
                elif i == 1:
                    pars_out['USAC_parameters_automaticSprtInit'] = 'SPRT_DELTA_AUTOM_INIT'
                elif i == 2:
                    pars_out['USAC_parameters_automaticSprtInit'] = 'SPRT_EPSILON_AUTOM_INIT'
                elif i == 3:
                    pars_out['USAC_parameters_automaticSprtInit'] = 'SPRT_DELTA_AND_EPSILON_AUTOM_INIT'
                else:
                    raise ValueError('Invalid value \'' + str(i) + '\' for parameter usac123[' + str(idx) + ']')
            elif idx == 1:
                if i == 0:
                    pars_out['USAC_parameters_automaticProsacParameters'] = 'disabled'
                elif i == 1:
                    pars_out['USAC_parameters_automaticProsacParameters'] = 'enabled'
                else:
                    raise ValueError('Invalid value \'' + str(i) + '\' for parameter usac123[' + str(idx) + ']')
            elif idx == 2:
                if i == 0:
                    pars_out['USAC_parameters_prevalidateSample'] = 'disabled'
                elif i == 1:
                    pars_out['USAC_parameters_prevalidateSample'] = 'enabled'
                else:
                    raise ValueError('Invalid value \'' + str(i) + '\' for parameter usac123[' + str(idx) + ']')
    elif par_name == 'usac56':
        if not isinstance(par_value, list) or len(par_value) != 2:
            raise ValueError('Wrong input parameters for usac56')
        pars_out = {}
        for idx, i in enumerate(par_value):
            if idx == 0:
                if i == 0:
                    pars_out['USAC_parameters_estimator'] = 'POSE_NISTER'
                elif i == 1:
                    pars_out['USAC_parameters_estimator'] = 'POSE_EIG_KNEIP'
                elif i == 2:
                    pars_out['USAC_parameters_estimator'] = 'POSE_STEWENIUS'
                else:
                    raise ValueError('Invalid value \'' + str(i) + '\' for parameter usac56[' + str(idx) + ']')
            elif idx == 1:
                if i == 0:
                    pars_out['USAC_parameters_refinealg'] = 'REF_WEIGHTS'
                elif i == 1:
                    pars_out['USAC_parameters_refinealg'] = 'REF_8PT_PSEUDOHUBER'
                elif i == 2:
                    pars_out['USAC_parameters_refinealg'] = 'REF_EIG_KNEIP'
                elif i == 3:
                    pars_out['USAC_parameters_refinealg'] = 'REF_EIG_KNEIP_WEIGHTS'
                elif i == 4:
                    pars_out['USAC_parameters_refinealg'] = 'REF_STEWENIUS'
                elif i == 5:
                    pars_out['USAC_parameters_refinealg'] = 'REF_STEWENIUS_WEIGHTS'
                elif i == 6:
                    pars_out['USAC_parameters_refinealg'] = 'REF_NISTER'
                elif i == 7:
                    pars_out['USAC_parameters_refinealg'] = 'REF_NISTER_WEIGHTS'
                else:
                    raise ValueError('Invalid value \'' + str(i) + '\' for parameter usac56[' + str(idx) + ']')
    elif par_name == 'refineRT':
        if not isinstance(par_value, list) or len(par_value) != 2:
            raise ValueError('Wrong input parameters for refineRT')
        pars_out = {}
        for idx, i in enumerate(par_value):
            if idx == 0:
                if i == 0:
                    pars_out['refineMethod_algorithm'] = 'PR_NO_REFINEMENT'
                elif i == 2:
                    pars_out['refineMethod_algorithm'] = 'PR_8PT'
                elif i == 3:
                    pars_out['refineMethod_algorithm'] = 'PR_NISTER'
                elif i == 4:
                    pars_out['refineMethod_algorithm'] = 'PR_STEWENIUS'
                elif i == 5:
                    pars_out['refineMethod_algorithm'] = 'PR_KNEIP'
                else:
                    raise ValueError('Invalid value \'' + str(i) + '\' for parameter refineRT[' + str(idx) + ']')
            elif idx == 1:
                if i == 0:
                    pars_out['refineMethod_costFunction'] = 'PR_NO_WEIGHTS'
                elif i == 1:
                    pars_out['refineMethod_costFunction'] = 'PR_TORR_WEIGHTS'
                elif i == 2:
                    pars_out['refineMethod_costFunction'] = 'PR_PSEUDOHUBER_WEIGHTS'
                else:
                    raise ValueError('Invalid value \'' + str(i) + '\' for parameter refineRT[' + str(idx) + ']')
    elif par_name == 'refineRT_stereo':
        if not isinstance(par_value, list) or len(par_value) != 2:
            raise ValueError('Wrong input parameters for refineRT_stereo')
        pars_out = {}
        for idx, i in enumerate(par_value):
            if idx == 0:
                if i == 0:
                    pars_out['stereoParameters_refineMethod_CorrPool_algorithm'] = 'PR_NO_REFINEMENT'
                elif i == 2:
                    pars_out['stereoParameters_refineMethod_CorrPool_algorithm'] = 'PR_8PT'
                elif i == 3:
                    pars_out['stereoParameters_refineMethod_CorrPool_algorithm'] = 'PR_NISTER'
                elif i == 4:
                    pars_out['stereoParameters_refineMethod_CorrPool_algorithm'] = 'PR_STEWENIUS'
                elif i == 5:
                    pars_out['stereoParameters_refineMethod_CorrPool_algorithm'] = 'PR_KNEIP'
                else:
                    raise ValueError('Invalid value \'' + str(i) + '\' for parameter refineRT_stereo[' + str(idx) + ']')
            elif idx == 1:
                if i == 0:
                    pars_out['stereoParameters_refineMethod_CorrPool_costFunction'] = 'PR_NO_WEIGHTS'
                elif i == 1:
                    pars_out['stereoParameters_refineMethod_CorrPool_costFunction'] = 'PR_TORR_WEIGHTS'
                elif i == 2:
                    pars_out['stereoParameters_refineMethod_CorrPool_costFunction'] = 'PR_PSEUDOHUBER_WEIGHTS'
                else:
                    raise ValueError('Invalid value \'' + str(i) + '\' for parameter refineRT_stereo[' + str(idx) + ']')
    elif par_name == 'th':
        if isinstance(par_value, int):
            par_value = float(par_value)
        elif not isinstance(par_value, float):
            raise ValueError('Wrong input format for th')
        pars_out = {'th': par_value}
    elif par_name == 'robMFilt':
        if not isinstance(par_value, str):
            raise ValueError('Wrong input format for robMFilt')
        if par_value not in ['USAC', 'RANSAC']:
            raise ValueError('Invalid value \'' + par_value + '\' for parameter ' + par_name)
        pars_out = {'RobMethod': par_value}
    elif par_name == 'USACInlratFilt':
        if isinstance(par_value, float):
            par_value = int(par_value)
        elif not isinstance(par_value, int):
            raise ValueError('Wrong input format for USACInlratFilt')
        if not (par_value == 0 or par_value == 1):
            raise ValueError('Invalid value \'' + str(par_value) + '\' for parameter ' + par_name)
        if par_value == 0:
            pars_out = {'USAC_parameters_USACInlratFilt': 'GMS'}
        else:
            pars_out = {'USAC_parameters_USACInlratFilt': 'VFC'}
    elif par_name == 'bart':
        if isinstance(par_value, float):
            par_value = int(par_value)
        elif not isinstance(par_value, int):
            raise ValueError('Wrong input format for bart')
        if par_value == 0:
            pars_out = {'BART': 'disabled'}
        elif par_value == 1:
            pars_out = {'BART': 'extr_only'}
        elif par_value == 2:
            pars_out = {'BART': 'extr_intr'}
        else:
            raise ValueError('Invalid value \'' + par_value + '\' for parameter ' + par_name)
    elif par_name == 'bart_stereo':
        if isinstance(par_value, float):
            par_value = int(par_value)
        elif not isinstance(par_value, int):
            raise ValueError('Wrong input format for bart_stereo')
        if par_value == 0:
            pars_out = {'stereoParameters_BART_CorrPool': 'disabled'}
        elif par_value == 1:
            pars_out = {'stereoParameters_BART_CorrPool': 'extr_only'}
        elif par_value == 2:
            pars_out = {'stereoParameters_BART_CorrPool': 'extr_intr'}
        else:
            raise ValueError('Invalid value \'' + par_value + '\' for parameter ' + par_name)
    elif par_name == 'checkPoolPoseRobust':
        if isinstance(par_value, float):
            par_value = int(par_value)
        elif not isinstance(par_value, int):
            raise ValueError('Wrong input format for checkPoolPoseRobust')
        pars_out = {'stereoParameters_checkPoolPoseRobust': par_value}
    elif par_name == 'useRANSAC_fewMatches':
        if not isinstance(par_value, bool):
            raise ValueError('Wrong input format for useRANSAC_fewMatches')
        if par_value:
            pars_out = {'stereoParameters_useRANSAC_fewMatches': 'enabled'}
        else:
            pars_out = {'stereoParameters_useRANSAC_fewMatches': 'disabled'}
    elif par_name == 'maxPoolCorrespondences':
        if isinstance(par_value, float):
            par_value = int(par_value)
        elif not isinstance(par_value, int):
            raise ValueError('Wrong input format for maxPoolCorrespondences')
        pars_out = {'stereoParameters_maxPoolCorrespondences': par_value}
    elif par_name == 'maxDist3DPtsZ':
        if isinstance(par_value, int):
            par_value = float(par_value)
        elif not isinstance(par_value, float):
            raise ValueError('Wrong input format for maxDist3DPtsZ')
        pars_out = {'stereoParameters_maxDist3DPtsZ': par_value}
    elif par_name == 'maxRat3DPtsFar':
        if not isinstance(par_value, float):
            raise ValueError('Wrong input format for maxRat3DPtsFar')
        pars_out = {'stereoParameters_maxRat3DPtsFar': par_value}
    elif par_name == 'minStartAggInlRat':
        if not isinstance(par_value, float):
            raise ValueError('Wrong input format for minStartAggInlRat')
        pars_out = {'stereoParameters_minStartAggInlRat': par_value}
    elif par_name == 'minInlierRatSkip':
        if not isinstance(par_value, float):
            raise ValueError('Wrong input format for minInlierRatSkip')
        pars_out = {'stereoParameters_minInlierRatSkip': par_value}
    elif par_name == 'relInlRatThLast':
        if not isinstance(par_value, float):
            raise ValueError('Wrong input format for relInlRatThLast')
        pars_out = {'stereoParameters_relInlRatThLast': par_value}
    elif par_name == 'relInlRatThNew':
        if not isinstance(par_value, float):
            raise ValueError('Wrong input format for relInlRatThNew')
        pars_out = {'stereoParameters_relInlRatThNew': par_value}
    elif par_name == 'relMinInlierRatSkip':
        if not isinstance(par_value, float):
            raise ValueError('Wrong input format for relMinInlierRatSkip')
        pars_out = {'stereoParameters_relMinInlierRatSkip': par_value}
    elif par_name == 'minInlierRatioReInit':
        if not isinstance(par_value, float):
            raise ValueError('Wrong input format for minInlierRatioReInit')
        pars_out = {'stereoParameters_minInlierRatioReInit': par_value}
    elif par_name == 'minNormDistStable':
        if not isinstance(par_value, float):
            raise ValueError('Wrong input format for minNormDistStable')
        pars_out = {'stereoParameters_minNormDistStable': par_value}
    elif par_name == 'absThRankingStable':
        if not isinstance(par_value, float):
            raise ValueError('Wrong input format for absThRankingStable')
        pars_out = {'stereoParameters_absThRankingStable': par_value}
    elif par_name == 'minContStablePoses':
        if isinstance(par_value, float):
            par_value = int(par_value)
        elif not isinstance(par_value, int):
            raise ValueError('Wrong input format for minContStablePoses')
        pars_out = {'stereoParameters_minContStablePoses': par_value}
    elif par_name == 'minPtsDistance':
        if isinstance(par_value, int):
            par_value = float(par_value)
        elif not isinstance(par_value, float):
            raise ValueError('Wrong input format for minPtsDistance')
        pars_out = {'stereoParameters_minPtsDistance': par_value}
    else:
        raise ValueError('Unknown parameter ' + par_name)
    return pars_out


def main():
    parser = argparse.ArgumentParser(description='Execute different test scenarios for the autocalibration based '
                                                 'on name and test number in file '
                                                 'Autocalibration-Parametersweep-Testing.xlsx')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory holding file \'generated_dirs_config.txt\'')
    parser.add_argument('--nrCPUs', type=int, required=False, default=4,
                        help='Number of CPU cores for parallel processing. If a negative value is provided, '
                             'the program tries to find the number of available CPUs on the system - if it fails, '
                             'the absolute value of nrCPUs is used. Default: 4')
    parser.add_argument('--executable', type=str, required=True,
                        help='Executable of the autocalibration SW')
    parser.add_argument('--message_path', type=str, required=True,
                        help='Storing path for text files containing error and normal messages while '
                             'testing. For every different test a '
                             'new directory with the name of option test_name is created. '
                             'Within this directory another directory is created with the name of option test_nr')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Main output path for results of the autocalibration. This directory is also used for '
                             'loading and storing the YML file with found optimal parameters. For every different '
                             'test a new directory with the name of option test_name is created. '
                             'Within this directory another directory is created with the name of option test_nr.')
    parser.add_argument('--test_name', type=str, required=True,
                        help='Name of the main test like \'USAC-testing\' or \'USAC_vs_RANSAC\'')
    parser.add_argument('--test_nr', type=int, required=False,
                        help='Test number within the main test specified by test_name starting with 1')
    args = parser.parse_args()

    ret = choose_test(args.path, args.executable, args.nrCPUs, args.message_path,
                      args.output_path, args.test_name.lower(), args.test_nr)
    sys.exit(ret)


if __name__ == "__main__":
    main()