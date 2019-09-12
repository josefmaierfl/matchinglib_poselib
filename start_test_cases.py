"""
Execute different test scenarios for the autocalibration based on name and test number in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, warnings, time, subprocess as sp

def choose_test(path_ov_file, executable, cpu_cnt, message_path, output_path, test_name, test_nr):
    args = ['--path', path_ov_file, '--nrCPUs', str(cpu_cnt), '--executable', executable,
            '--message_path', message_path]

    #Set to False when performing actual tests:
    testing_test = False

    #Set settings based on best results after testing for further testing
    # Change cfgUSAC parameters 5 & 6 based on result of test_nr 1 of usac-testing
    usac56 = []
    # Change cfgUSAC parameters 1-3 based on result of test_nr 2 of usac-testing
    usac123 = []
    # Change USACInlratFilt parameter based on result of test_nr 2 of usac-testing
    USACInlratFilt = None
    # Change th parameter based on results of test_nr 1 & 2 of usac-testing
    th = None
    # Change refineRT parameters based on result of test_nr 1 of refinement_ba
    refineRT = []
    # Select robust method for testing VFC, GMS, and SOF: If best USAC solution is without GMS or VFC (for estimating
    # some USAC parameters (cfgUSAC digit 1 >1)), try using them for pre-filtering, otherwise use RANSAC
    # and compare with its original results
    robMFilt = None
    # Change BART parameter based on results of test_nr 1 & 2 of refinement_ba
    bart = None
    # Change refineRT_stereo parameters based on result of test_nr 1 of refinement_ba_stereo
    refineRT_stereo = []
    # Change BART_stereo parameter based on results of test_nr 1 & 2 of refinement_ba_stereo
    bart_stereo = None
    # Change minPtsDistance parameter based on result of test_nr 1 of correspondence_pool
    minPtsDistance = None
    # Change maxPoolCorrespondences parameter based on result of test_nr 1 of correspondence_pool
    maxPoolCorrespondences = None
    # Change maxRat3DPtsFar parameter based on result of test_nr 2 of correspondence_pool
    maxRat3DPtsFar = None
    # Change maxDist3DPtsZ parameter based on result of test_nr 2 of correspondence_pool
    maxDist3DPtsZ = None
    # Change relInlRatThLast parameter based on result of test_nr 1 of robustness
    relInlRatThLast = None
    # Change relInlRatThNew parameter based on result of test_nr 1 of robustness
    relInlRatThNew = None
    # Change minInlierRatSkip parameter based on result of test_nr 1 of robustness
    minInlierRatSkip = None
    # Change relMinInlierRatSkip parameter based on result of test_nr 1 of robustness
    relMinInlierRatSkip = None
    # Change minInlierRatioReInit parameter based on result of test_nr 1 of robustness
    minInlierRatioReInit = None
    # Change checkPoolPoseRobust parameter based on result of test_nr 2 of robustness
    checkPoolPoseRobust = None
    # Change minContStablePoses parameter based on result of test_nr 3 of robustness
    minContStablePoses = None
    # Change minNormDistStable parameter based on result of test_nr 3 of robustness
    minNormDistStable = None
    # Change absThRankingStable parameter based on result of test_nr 3 of robustness
    absThRankingStable = None
    # Change useRANSAC_fewMatches to true or false based on result of test_nr 4 of robustness
    useRANSAC_fewMatches = None


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
            if not usac56:
                raise ValueError('Enter best test results for parameters 5 & 6 of usac-testing')
            args += ['--cfgUSAC', '3', '1', '1', '0'] + list(map(str, usac56)) + ['1', '1', '1', '0', '0', '0']
            args += ['--USACInlratFilt', '2']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for usac-testing')
    elif test_name == 'usac_vs_ransac':
        args += ['--refineRT', '0', '0']
        args += ['--RobMethod', 'USAC', 'RANSAC']
        args += ['--th', '0.6', '2.0', '0.2']
        args += ['--useGTCamMat']
        if not usac56 or not usac123:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, usac123)) + ['0'] + list(map(str, usac56))
        if USACInlratFilt is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(USACInlratFilt)]
    elif test_name == 'refinement_ba':
        args += ['--RobMethod', 'USAC']
        if not usac56 or not usac123:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, usac123)) + ['0'] + list(map(str, usac56))
        if USACInlratFilt is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(USACInlratFilt)]
        if th is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(th)]
        if not test_nr:
            raise ValueError('test_nr is required for refinement_ba')
        if test_nr == 1:
            args += ['--refineRT', '0', '0', '1', '1', '1']#, '-13']
            args += ['--BART', '0', '1']
            args += ['--useGTCamMat']
        elif test_nr == 2:
            # if not refineRT:
            #     raise ValueError('Enter best test results for refineRT of refinement_ba')
            # args += ['--refineRT'] + list(map(str, refineRT))
            args += ['--refineRT', '0', '0', '1', '1', '1']
            args += ['--BART', '2']
            args += ['--nr_keypoints', '500']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for refinement_ba')
    elif test_name == 'vfc_gms_sof':
        if robMFilt is None:
            raise ValueError('Enter robust method for testing VFC, GMS, and SOF')
        args += ['--RobMethod', robMFilt]
        if robMFilt == 'USAC':
            if not usac56 or not usac123:
                raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
            args += ['--cfgUSAC'] + list(map(str, usac123)) + ['0'] + list(map(str, usac56))
        else:
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
        args += ['--USACInlratFilt', '0']
        if th is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(th)]
        args += ['--refineRT', '0', '0']
        args += ['--useGTCamMat']
        args += ['--refineVFC']
        args += ['--refineSOF']
        args += ['--refineGMS']
    elif test_name == 'refinement_ba_stereo':
        args += ['--RobMethod', 'USAC']
        if not usac56 or not usac123:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, usac123)) + ['0'] + list(map(str, usac56))
        if USACInlratFilt is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(USACInlratFilt)]
        if th is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(th)]
        if not refineRT:
            raise ValueError('Enter best test results for refineRT of refinement_ba')
        args += ['--refineRT'] + list(map(str, refineRT))
        if bart is None:
            raise ValueError('Enter best test results for BART of refinement_ba')
        args += ['--BART', str(bart)]
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
            if not refineRT_stereo:
                raise ValueError('Enter best test results for refineRT_stereo of refinement_ba_stereo')
            args += ['--refineRT_stereo'] + list(map(str, refineRT_stereo))
            args += ['--BART_stereo', '0', '1']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for refinement_ba_stereo')
    elif test_name == 'correspondence_pool':
        args += ['--RobMethod', 'USAC']
        if not usac56 or not usac123:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, usac123)) + ['0'] + list(map(str, usac56))
        if USACInlratFilt is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(USACInlratFilt)]
        if th is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(th)]
        if not refineRT:
            raise ValueError('Enter best test results for refineRT of refinement_ba')
        args += ['--refineRT'] + list(map(str, refineRT))
        if bart is None:
            raise ValueError('Enter best test results for BART of refinement_ba')
        args += ['--BART', str(bart)]
        args += ['--stereoRef']
        args += ['--minStartAggInlRat', '0.075']
        args += ['--minInlierRatSkip', '0.075']
        args += ['--useGTCamMat']
        if not refineRT_stereo:
            raise ValueError('Enter best test results for refineRT_stereo of refinement_ba_stereo')
        args += ['--refineRT_stereo'] + list(map(str, refineRT_stereo))
        if bart_stereo is None:
            raise ValueError('Enter best test results for BART_stereo of refinement_ba_stereo')
        args += ['--BART_stereo', str(bart_stereo)]
        if not test_nr:
            raise ValueError('test_nr is required for correspondence_pool')
        if test_nr == 1:
            args += ['--minPtsDistance', '1.5', '15.5', '2.0']
            args += ['--maxPoolCorrespondences', '300', '1000', '100', '1000', '2000', '200', '2000', '5000', '500',
                     '5000', '10000', '1000', '10000', '20000', '2000', '20000', '30000', '5000']
        elif test_nr == 2:
            if minPtsDistance is None:
                raise ValueError('Enter best test results for minPtsDistance of correspondence_pool')
            args += ['--minPtsDistance', str(minPtsDistance)]
            if maxPoolCorrespondences is None:
                raise ValueError('Enter best test results for maxPoolCorrespondences of correspondence_pool')
            args += ['--maxPoolCorrespondences', str(maxPoolCorrespondences)]
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
            if minPtsDistance is None:
                raise ValueError('Enter best test results for minPtsDistance of correspondence_pool')
            args += ['--minPtsDistance', str(minPtsDistance)]
            if maxPoolCorrespondences is None:
                raise ValueError('Enter best test results for maxPoolCorrespondences of correspondence_pool')
            args += ['--maxPoolCorrespondences', str(maxPoolCorrespondences)]
            if maxRat3DPtsFar is None:
                raise ValueError('Enter best test results for maxRat3DPtsFar of correspondence_pool')
            args += ['--maxRat3DPtsFar', str(maxRat3DPtsFar)]
            if maxDist3DPtsZ is None:
                raise ValueError('Enter best test results for maxDist3DPtsZ of correspondence_pool')
            args += ['--maxDist3DPtsZ', str(maxDist3DPtsZ)]
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for correspondence_pool')
    elif test_name == 'robustness':
        args += ['--RobMethod', 'USAC']
        if not usac56 or not usac123:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + list(map(str, usac123)) + ['0'] + list(map(str, usac56))
        if USACInlratFilt is None:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(USACInlratFilt)]
        if th is None:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(th)]
        if not refineRT:
            raise ValueError('Enter best test results for refineRT of refinement_ba')
        args += ['--refineRT'] + list(map(str, refineRT))
        if bart is None:
            raise ValueError('Enter best test results for BART of refinement_ba')
        args += ['--BART', str(bart)]
        args += ['--stereoRef']
        args += ['--useGTCamMat']
        if not refineRT_stereo:
            raise ValueError('Enter best test results for refineRT_stereo of refinement_ba_stereo')
        args += ['--refineRT_stereo'] + list(map(str, refineRT_stereo))
        if bart_stereo is None:
            raise ValueError('Enter best test results for BART_stereo of refinement_ba_stereo')
        args += ['--BART_stereo', str(bart_stereo)]
        if minPtsDistance is None:
            raise ValueError('Enter best test results for minPtsDistance of correspondence_pool')
        args += ['--minPtsDistance', str(minPtsDistance)]
        if maxPoolCorrespondences is None:
            raise ValueError('Enter best test results for maxPoolCorrespondences of correspondence_pool')
        args += ['--maxPoolCorrespondences', str(maxPoolCorrespondences)]
        warnings.warn("Warning: Are you sure the selected number of maxPoolCorrespondences is small enough "
                      "to not reach an overall runtime of 0.8s per frame?")
        time.sleep(5.0)
        if maxRat3DPtsFar is None:
            raise ValueError('Enter best test results for maxRat3DPtsFar of correspondence_pool')
        args += ['--maxRat3DPtsFar', str(maxRat3DPtsFar)]
        if maxDist3DPtsZ is None:
            raise ValueError('Enter best test results for maxDist3DPtsZ of correspondence_pool')
        args += ['--maxDist3DPtsZ', str(maxDist3DPtsZ)]
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
            if relInlRatThLast is None:
                raise ValueError('Enter best test results for relInlRatThLast of robustness')
            args += ['--relInlRatThLast', str(relInlRatThLast)]
            if relInlRatThNew is None:
                raise ValueError('Enter best test results for relInlRatThNew of robustness')
            args += ['--relInlRatThNew', str(relInlRatThNew)]
            if minInlierRatSkip is None:
                raise ValueError('Enter best test results for minInlierRatSkip of robustness')
            args += ['--minInlierRatSkip', str(minInlierRatSkip)]
            if relMinInlierRatSkip is None:
                raise ValueError('Enter best test results for relMinInlierRatSkip of robustness')
            args += ['--relMinInlierRatSkip', str(relMinInlierRatSkip)]
            if minInlierRatioReInit is None:
                raise ValueError('Enter best test results for minInlierRatioReInit of robustness')
            args += ['--minInlierRatioReInit', str(minInlierRatioReInit)]
            args += ['--checkPoolPoseRobust', '0', '5', '1']
        elif test_nr == 3:
            warnings.warn("Warning: Are you sure you have selected the SMALL dataset for testing?")
            time.sleep(5.0)
            args += ['--nr_keypoints', '30to500']
            if relInlRatThLast is None:
                raise ValueError('Enter best test results for relInlRatThLast of robustness')
            args += ['--relInlRatThLast', str(relInlRatThLast)]
            if relInlRatThNew is None:
                raise ValueError('Enter best test results for relInlRatThNew of robustness')
            args += ['--relInlRatThNew', str(relInlRatThNew)]
            if minInlierRatSkip is None:
                raise ValueError('Enter best test results for minInlierRatSkip of robustness')
            args += ['--minInlierRatSkip', str(minInlierRatSkip)]
            if relMinInlierRatSkip is None:
                raise ValueError('Enter best test results for relMinInlierRatSkip of robustness')
            args += ['--relMinInlierRatSkip', str(relMinInlierRatSkip)]
            if minInlierRatioReInit is None:
                raise ValueError('Enter best test results for minInlierRatioReInit of robustness')
            args += ['--minInlierRatioReInit', str(minInlierRatioReInit)]
            if checkPoolPoseRobust is None:
                raise ValueError('Enter best test results for checkPoolPoseRobust of robustness')
            args += ['--checkPoolPoseRobust', str(checkPoolPoseRobust)]
            args += ['--minContStablePoses', '3', '5', '1']
            args += ['--minNormDistStable', '0.25', '0.75', '0.1']
            args += ['--absThRankingStable', '0.05', '0.5', '0.075']
        elif test_nr == 4:
            warnings.warn("Warning: Are you sure you have selected the SMALL dataset for testing?")
            time.sleep(5.0)
            args += ['--nr_keypoints', '20to160']
            if relInlRatThLast is None:
                raise ValueError('Enter best test results for relInlRatThLast of robustness')
            args += ['--relInlRatThLast', str(relInlRatThLast)]
            if relInlRatThNew is None:
                raise ValueError('Enter best test results for relInlRatThNew of robustness')
            args += ['--relInlRatThNew', str(relInlRatThNew)]
            if minInlierRatSkip is None:
                raise ValueError('Enter best test results for minInlierRatSkip of robustness')
            args += ['--minInlierRatSkip', str(minInlierRatSkip)]
            if relMinInlierRatSkip is None:
                raise ValueError('Enter best test results for relMinInlierRatSkip of robustness')
            args += ['--relMinInlierRatSkip', str(relMinInlierRatSkip)]
            if minInlierRatioReInit is None:
                raise ValueError('Enter best test results for minInlierRatioReInit of robustness')
            args += ['--minInlierRatioReInit', str(minInlierRatioReInit)]
            if checkPoolPoseRobust is None:
                raise ValueError('Enter best test results for checkPoolPoseRobust of robustness')
            args += ['--checkPoolPoseRobust', str(checkPoolPoseRobust)]
            if minContStablePoses is None:
                raise ValueError('Enter best test results for minContStablePoses of robustness')
            args += ['--minContStablePoses', str(minContStablePoses)]
            if minNormDistStable is None:
                raise ValueError('Enter best test results for minNormDistStable of robustness')
            args += ['--minNormDistStable', str(minNormDistStable)]
            if absThRankingStable is None:
                raise ValueError('Enter best test results for absThRankingStable of robustness')
            args += ['--absThRankingStable', str(absThRankingStable)]
            args += ['--useRANSAC_fewMatches']
        elif test_nr == 5:
            warnings.warn("Warning: Are you sure you have selected the LARGE dataset for testing?")
            time.sleep(5.0)
            if relInlRatThLast is None:
                raise ValueError('Enter best test results for relInlRatThLast of robustness')
            args += ['--relInlRatThLast', str(relInlRatThLast)]
            if relInlRatThNew is None:
                raise ValueError('Enter best test results for relInlRatThNew of robustness')
            args += ['--relInlRatThNew', str(relInlRatThNew)]
            if minInlierRatSkip is None:
                raise ValueError('Enter best test results for minInlierRatSkip of robustness')
            args += ['--minInlierRatSkip', str(minInlierRatSkip)]
            if relMinInlierRatSkip is None:
                raise ValueError('Enter best test results for relMinInlierRatSkip of robustness')
            args += ['--relMinInlierRatSkip', str(relMinInlierRatSkip)]
            if minInlierRatioReInit is None:
                raise ValueError('Enter best test results for minInlierRatioReInit of robustness')
            args += ['--minInlierRatioReInit', str(minInlierRatioReInit)]
            if checkPoolPoseRobust is None:
                raise ValueError('Enter best test results for checkPoolPoseRobust of robustness')
            args += ['--checkPoolPoseRobust', str(checkPoolPoseRobust)]
            if minContStablePoses is None:
                raise ValueError('Enter best test results for minContStablePoses of robustness')
            args += ['--minContStablePoses', str(minContStablePoses)]
            if minNormDistStable is None:
                raise ValueError('Enter best test results for minNormDistStable of robustness')
            args += ['--minNormDistStable', str(minNormDistStable)]
            if absThRankingStable is None:
                raise ValueError('Enter best test results for absThRankingStable of robustness')
            args += ['--absThRankingStable', str(absThRankingStable)]
            if useRANSAC_fewMatches is None:
                raise ValueError('Enter best test result for useRANSAC_fewMatches of robustness')
            if useRANSAC_fewMatches:
                args += ['--useRANSAC_fewMatches']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for robustness')
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
            usac56 = [2, 5]
            if not usac56:
                raise ValueError('Enter best test results for parameters 5 & 6 of usac-testing')
            args += ['--cfgUSAC', '3', '1', '1', '0'] + list(map(str, usac56)) + ['1', '1', '1', '0', '0', '0']
            args += ['--USACInlratFilt', '2']
        elif test_nr == 3:
            args += ['--refineRT', '0', '0']
            args += ['--RobMethod', 'USAC', 'RANSAC']
            args += ['--th', '0.6', '2.0', '0.2']
            args += ['--useGTCamMat']
            usac56 = [2, 5]
            usac123 = [3,1,1]
            if not usac56 or not usac123:
                raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
            args += ['--cfgUSAC'] + list(map(str, usac123)) + ['0'] + list(map(str, usac56))
            USACInlratFilt = 0
            if USACInlratFilt is None:
                raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
            args += ['--USACInlratFilt', str(USACInlratFilt)]
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
            refineRT = [4,2]
            args += ['--refineRT'] + list(map(str, refineRT))
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
            args += ['--useRANSAC_fewMatches']
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
                        help='Storing path for text files containing error and normal messages during the '
                             'generation process of scenes and matches. For every different test a '
                             'new directory with the name of option test_name is created. '
                             'Within this directory another directory is created with the name of option test_nr')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Main output path for results of the autocalibration. For every different test a '
                             'new directory with the name of option test_name is created. '
                             'Within this directory another directory is created with the name of option test_nr')
    parser.add_argument('--test_name', type=str, required=True,
                        help='Name of the main test like \'USAC-testing\' or \'USAC_vs_RANSAC\'')
    parser.add_argument('--test_nr', type=int, required=False,
                        help='Test number within the main test specified by test_name starting with 1')
    args = parser.parse_args()

    return choose_test(args.path, args.executable, args.nrCPUs, args.message_path,
                       args.output_path, args.test_name.lower(), args.test_nr)


if __name__ == "__main__":
    main()