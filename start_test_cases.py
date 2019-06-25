"""
Execute different test scenarios for the autocalibration based on name and test number in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, warnings, time

def choose_test(path_ov_file, executable, cpu_cnt, message_path, output_path, test_name, test_nr):
    args = ['--path', path_ov_file, '--nrCPUs', str(cpu_cnt), '--executable', executable,
            '--message_path', message_path, '--output_path', output_path]

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
            args += ['--depths', 'NMF']
            args += ['--nr_keypoints', '500']
            args += ['--kp_pos_distr', 'equ']
            if not usac56:
                raise ValueError('Enter best test results for parameters 5 & 6 of usac-testing')
            args += ['--cfgUSAC', '3', '1', '1', '0'] + map(str, usac56) + ['1', '1', '1', '0', '0', '0']
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
        args += ['--cfgUSAC'] + map(str, usac123) + ['0'] + map(str, usac56)
        if not USACInlratFilt:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(USACInlratFilt)]
    elif test_name == 'refinement_ba':
        args += ['--RobMethod', 'USAC']
        if not usac56 or not usac123:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + map(str, usac123) + ['0'] + map(str, usac56)
        if not USACInlratFilt:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(USACInlratFilt)]
        if not th:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(th)]
        if not test_nr:
            raise ValueError('test_nr is required for refinement_ba')
        if test_nr == 1:
            args += ['--refineRT', '0', '0', '1', '1', '-13']
            args += ['--BART', '0', '1']
            args += ['--useGTCamMat']
        elif test_nr == 2:
            if not refineRT:
                raise ValueError('Enter best test results for refineRT of refinement_ba')
            args += ['--refineRT'] + map(str, refineRT)
            args += ['--BART', '0', '1']
            args += ['--nr_keypoints', '500']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for refinement_ba')
    elif test_name == 'vfc_gms_sof':
        if not robMFilt:
            raise ValueError('Enter robust method for testing VFC, GMS, and SOF')
        args += ['--RobMethod', robMFilt]
        if robMFilt == 'USAC':
            if not usac56 or not usac123:
                raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
            args += ['--cfgUSAC'] + map(str, usac123) + ['0'] + map(str, usac56)
        else:
            args += ['--cfgUSAC', '3', '1', '1', '0', '2', '5']
        args += ['--USACInlratFilt', '0']
        if not th:
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
        args += ['--cfgUSAC'] + map(str, usac123) + ['0'] + map(str, usac56)
        if not USACInlratFilt:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(USACInlratFilt)]
        if not th:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(th)]
        if not refineRT:
            raise ValueError('Enter best test results for refineRT of refinement_ba')
        args += ['--refineRT'] + map(str, refineRT)
        if not bart:
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
            args += ['--refineRT_stereo', '0', '0', '1', '1', '-13']
            args += ['--BART_stereo', '0', '1']
            args += ['--useGTCamMat']
        elif test_nr == 2:
            if not refineRT_stereo:
                raise ValueError('Enter best test results for refineRT_stereo of refinement_ba_stereo')
            args += ['--refineRT_stereo'] + map(str, refineRT_stereo)
            args += ['--BART_stereo', '0', '1']
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for refinement_ba_stereo')
    elif test_name == 'correspondence_pool':
        args += ['--RobMethod', 'USAC']
        if not usac56 or not usac123:
            raise ValueError('Enter best test results for parameters 1-3 and 5-6 of usac-testing')
        args += ['--cfgUSAC'] + map(str, usac123) + ['0'] + map(str, usac56)
        if not USACInlratFilt:
            raise ValueError('Enter best test result for USACInlratFilt of usac-testing')
        args += ['--USACInlratFilt', str(USACInlratFilt)]
        if not th:
            raise ValueError('Enter best test result for th of usac-testing or usac_vs_ransac')
        args += ['--th', str(th)]
        if not refineRT:
            raise ValueError('Enter best test results for refineRT of refinement_ba')
        args += ['--refineRT'] + map(str, refineRT)
        if not bart:
            raise ValueError('Enter best test results for BART of refinement_ba')
        args += ['--BART', str(bart)]
        args += ['--stereoRef']
        args += ['--minStartAggInlRat', '0.075']
        args += ['--minInlierRatSkip', '0.075']
        if not refineRT_stereo:
            raise ValueError('Enter best test results for refineRT_stereo of refinement_ba_stereo')
        args += ['--refineRT_stereo'] + map(str, refineRT_stereo)
        if not bart_stereo:
            raise ValueError('Enter best test results for BART_stereo of refinement_ba_stereo')
        args += ['--BART_stereo', str(bart_stereo)]
        if not test_nr:
            raise ValueError('test_nr is required for correspondence_pool')
        if test_nr == 1:
            args += ['--minPtsDistance', '1.0', '15.0', '2.0']
            args += ['--maxPoolCorrespondences', '200', '1000', '100', '1000', '2000', '200', '2000', '5000', '500',
                     '5000', '10000', '1000', '10000', '20000', '2000', '20000', '30000', '5000']
        elif test_nr == 2:
            if not minPtsDistance:
                raise ValueError('Enter best test results for minPtsDistance of correspondence_pool')
            args += ['--minPtsDistance', str(minPtsDistance)]
            if not maxPoolCorrespondences:
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
            if not minPtsDistance:
                raise ValueError('Enter best test results for minPtsDistance of correspondence_pool')
            args += ['--minPtsDistance', str(minPtsDistance)]
            if not maxPoolCorrespondences:
                raise ValueError('Enter best test results for maxPoolCorrespondences of correspondence_pool')
            args += ['--maxPoolCorrespondences', str(maxPoolCorrespondences)]
            if not maxRat3DPtsFar:
                raise ValueError('Enter best test results for maxRat3DPtsFar of correspondence_pool')
            args += ['--maxRat3DPtsFar', str(maxRat3DPtsFar)]
            if not maxDist3DPtsZ:
                raise ValueError('Enter best test results for maxDist3DPtsZ of correspondence_pool')
            args += ['--maxDist3DPtsZ', str(maxDist3DPtsZ)]
        else:
            raise ValueError('test_nr ' + str(test_nr) + ' is not supported for correspondence_pool')


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
                        help='Storing path for text files containing error and normal mesages during the '
                             'generation process of scenes and matches')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for results of the autocalibration')
    parser.add_argument('--test_name', type=str, required=True,
                        help='Name of the main test like \'USAC-testing\' or \'USAC_vs_RANSAC\'')
    parser.add_argument('--test_nr', type=int, required=False,
                        help='Test number within the main test specified by test_name starting with 1')
    args = parser.parse_args()

    return choose_test(args.path, args.executable, args.nrCPUs, args.message_path,
                       args.output_path, args.test_name.lower(), args.test_nr)



if __name__ == "__main__":
    main()