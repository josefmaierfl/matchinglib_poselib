"""
Main script file for executing the whole test procedure for testing the autocalibration SW
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np

def main():
    parser = argparse.ArgumentParser(description='Main script file for executing the whole test procedure for '
                                                 'testing the autocalibration SW')
    parser.add_argument('--path', type=str, required=True,
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
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to images')
    parser.add_argument('--store_path', type=str, required=True,
                        help='Storing path for generated scenes and matches')
    parser.add_argument('--load_path', type=str, required=False,
                        help='Optional loading path for generated scenes and matches. '
                             'If not provided, store_path is used.')
    parser.add_argument('--nrCPUs', type=int, required=False, default=-16,
                        help='Number of CPU cores for parallel processing. If a negative value is provided, '
                             'the program tries to find the number of available CPUs on the system - if it fails, '
                             'the absolute value of nrCPUs is used. Default: 4')
    parser.add_argument('--executable', type=str, required=True,
                        help='Executable of the application generating the sequences')
    parser.add_argument('--message_path', type=str, required=True,
                        help='Storing path for text files containing error and normal messages during the '
                             'generation process of scenes and matches')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' holding directories with template scene '
                                                    'configuration files does not exist')
    main_test_names = ['usac-testing', 'usac_vs_ransac', 'refinement_ba', 'vfc_gms_sof',
                       'refinement_ba_stereo', 'correspondence_pool', 'robustness', 'usac_vs_autocalib']
    if args.skip_tests:
        for i in args.skip_tests:
            if i not in main_test_names:
                raise ValueError('Cannot skip test ' + i + ' as it does not exist.')
    scenes_test_names = ['usac-testing', 'correspondence_pool', 'robustness', 'usac_vs_autocalib']
    if args.skip_gen_sc_conf:
        for i in args.skip_gen_sc_conf:
            if i not in scenes_test_names:
                raise ValueError('Cannot skip generation of configuration files for '
                                 'scenes with test name ' + i + ' as it does not exist.')
    if args.skip_crt_sc:
        for i in args.skip_gen_sc_conf:
            if i not in scenes_test_names:
                raise ValueError('Cannot skip creation of scenes with test name ' + i + ' as it does not exist.')
    if args.crt_sc_dirs_file and not os.path.exists(args.crt_sc_dirs_file):
        raise ValueError('File ' + args.crt_sc_dirs_file + ' does not exist.')
    if args.crt_sc_cmds_file and not os.path.exists(args.crt_sc_cmds_file):
        raise ValueError('File ' + args.crt_sc_cmds_file + ' does not exist.')
    if not os.path.exists(args.img_path):
        raise ValueError("Image path does not exist")
    if not os.path.exists(args.store_path):
        raise ValueError("Path for storing sequences does not exist")
    if len(os.listdir(args.store_path)) != 0:
        raise ValueError("Path for storing sequences is not empty")
    if args.load_path:
        if not os.path.exists(args.load_path):
            raise ValueError("Path for loading sequences does not exist")
    if not os.path.exists(args.message_path):
        raise ValueError("Path for storing stdout and stderr does not exist")
    if not os.path.isfile(args.executable):
        raise ValueError('Executable ' + args.executable + ' for generating scenes does not exist')
    elif not os.access(args.executable,os.X_OK):
        raise ValueError('Unable to execute ' + args.executable)
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


    try:
        if args.inlier_range:
            gen_configs(args.path, args.inlier_range, [], args.kpAccRange, args.img_path, args.store_path,
                        args.load_path, args.treatTPasCorrs)
        else:
            gen_configs(args.path, [], args.inlchrate_range, args.kpAccRange, args.img_path, args.store_path,
                        args.load_path, False)
    except FileExistsError:
        print(sys.exc_info()[0])
        return 1
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise
    return 0


if __name__ == "__main__":
    main()