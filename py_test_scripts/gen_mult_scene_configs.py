"""
Loads initial configuration files and builds a folder structure and an overview file for generating
specific configuration and overview files for scenes by varying the used inlier ratio and keypoint accuracy
"""
import sys, re, argparse, os, subprocess as sp, warnings
from shutil import copyfile


def gen_configs(input_path, path_confs_out, inlier_range, inlier_chr, kpAccRange, img_path, store_path, load_path,
                treatTPasCorrs, treatTPasCorrsSel):
    use_load_path = False
    if load_path:
        use_load_path = True
    #Load file names
    files_i = os.listdir(input_path)
    if len(files_i) == 0:
        raise ValueError('No files found.')
    files = []
    for i in files_i:
        fnObj = re.search(r'_initial\.', i, re.I)
        if fnObj:
            files.append(i)
    if len(files) == 0:
        raise ValueError('No files including _init found.')
    ovfile_new = os.path.join(path_confs_out, 'generated_dirs_config.txt')
    if os.path.exists(ovfile_new):
        raise FileExistsError("File containing configuration file names already exists: " + ovfile_new)
    with open(ovfile_new, 'w') as fo:
        for i in files:
            filen = os.path.basename(i)
            fnObj = re.match(r'(.*)_initial\.(.*)', filen, re.I)
            if fnObj:
                base = fnObj.group(1)
                ending = fnObj.group(2)
            else:
                raise ValueError('Cannot extract part of file name')
            pathnew = os.path.join(path_confs_out, base)
            try:
                os.mkdir(pathnew)
            except FileExistsError:
                raise ValueError('Directory ' + pathnew + ' already exists')
            #Copy init file
            try:
                filenew = os.path.join(pathnew, filen)
                copyfile(os.path.join(input_path, i), filenew)
            except IOError:
                raise ValueError('Unable to copy template file')
            #Generate new store path
            store_path_new = os.path.join(store_path, base)
            try:
                os.mkdir(store_path_new)
            except FileExistsError:
                raise ValueError('Directory ' + store_path_new + ' already exists')
            #Generate new config files
            pyfilepath = os.path.dirname(os.path.realpath(__file__))
            if inlier_range:
                pyfilename = os.path.join(pyfilepath, 'gen_scene_configs.py')
            else:
                pyfilename = os.path.join(pyfilepath, 'gen_rob-test_scene_configs.py')
            #pyfilename = 'gen_scene_configs.py'
            if use_load_path:
                load_path_n = load_path
            else:
                load_path_n = store_path_new
            try:
                if inlier_range:
                    cmdline = ['python', pyfilename, '--filename', filenew,
                               '--inlier_range', '%.2f' % inlier_range[0], '%.2f' % inlier_range[1],
                               '%.2f' % inlier_range[2],
                               '--kpAccRange', '%.2f' % kpAccRange[0], '%.2f' % kpAccRange[1],
                               '%.2f' % kpAccRange[2],
                               '--img_path', img_path,
                               '--store_path', store_path_new,
                               '--load_path', load_path_n]
                    if treatTPasCorrs:
                        if treatTPasCorrsSel:
                            fnObj = re.search(treatTPasCorrsSel, filen, re.I)
                            if fnObj:
                                cmdline += ['--treatTPasCorrs']
                        else:
                            cmdline += ['--treatTPasCorrs']
                    retcode = sp.run(cmdline, shell=False, check=True).returncode
                else:
                    cmdline = ['python', pyfilename, '--filename', filenew,
                               '--inlchrate_range', '%.2f' % inlier_chr[0], '%.2f' % inlier_chr[1],
                               '%.2f' % inlier_chr[2]]
                    if len(inlier_chr) > 3:
                        for j in range(3, len(inlier_chr)):
                            cmdline.append('%.2f' % inlier_chr[j])
                    cmdline = cmdline + ['--kpAccRange', '%.2f' % kpAccRange[0], '%.2f' % kpAccRange[1],
                                         '%.2f' % kpAccRange[2],
                                         '--img_path', img_path,
                                         '--store_path', store_path_new,
                                         '--load_path', load_path_n]
                    retcode = sp.run(cmdline, shell=False, check=True).returncode
                if retcode < 0:
                    print("Child was terminated by signal", -retcode, file=sys.stderr)
                else:
                    print("Child returned", retcode, file=sys.stderr)
            except OSError as e:
                print("Execution failed:", e, file=sys.stderr)
            except sp.CalledProcessError as e:
                print("Execution failed:", e, file=sys.stderr)
            #Delete copied template file
            os.remove(filenew)
            #Write directory name holding config files to overview file
            fo.write(pathnew + '\n')
    return 0


def main():
    parser = argparse.ArgumentParser(description='Generate configuration files and overview files for scenes by '
                                                 'varying the used inlier ratio and keypoint accuracy')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory holding template configuration files')
    parser.add_argument('--path_confs_out', type=str, required=False,
                        help='Optional directory for writing configuration files. If not available, '
                             'the directory from argument \'path\' is used.')
    parser.add_argument('--inlier_range', type=float, nargs=3, required=False,
                        help='Range for the inlier ratio. Format: min max step_size')
    parser.add_argument('--inlchrate_range', type=float, nargs='+', required=False,
                        help='Range and additional specific values for the inlier ratio change rate. '
                             'Format: min max step_size v1 v2 ... vn')
    parser.add_argument('--kpAccRange', type=float, nargs=3, required=True,
                        help='Range for the keypoint accuracy. The entered value is devided by 2 ro reach an '
                             'approximate maximum keypoint accuracy based on the given value as the value in '
                             'the configuration file corresponds to the standard deviation. '
                             'Format: min max step_size')
    parser.add_argument('--treatTPasCorrs', type=bool, nargs='?', required=False, default=False, const=True,
                        help='If provided, the number of TP is calculated and written to the config file based on '
                             'the given inlier ratios and the number or range of desired correspondences (TP+FP) '
                             'that is extracted from the filename (*_TP-?*).')
    parser.add_argument('--treatTPasCorrsConfsSel', type=str, required=False,
                        help='If provided and option treatTPasCorrs is provided, one unique string element which '
                             'can be found in one or more initial configuration files. Only for these files, the '
                             'number of TP is calculated as described in the option description of treatTPasCorrs.')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to images')
    parser.add_argument('--store_path', type=str, required=True,
                        help='Storing path for generated scenes and matches')
    parser.add_argument('--load_path', type=str, required=False,
                        help='Optional loading path for generated scenes and matches. '
                             'If not provided, store_path is used.')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' holding template scene configuration files does not exist')
    if not args.path_confs_out:
        args.path_confs_out = args.path
    elif not os.path.exists(args.path_confs_out):
        os.makedirs(args.path_confs_out, exist_ok=True)
    if args.inlier_range:
        if (args.inlier_range[0] > args.inlier_range[1]) or \
                (args.inlier_range[2] > (args.inlier_range[1] - args.inlier_range[0])):
            raise ValueError("Parameters 2-4 (inlier ratio) must have the following format: "
                             "range_min range_max step_size")
        if not round(float((args.inlier_range[1] - args.inlier_range[0]) / args.inlier_range[2]), 6).is_integer():
            raise ValueError("Inlier step size is wrong")
    elif args.inlchrate_range:
        if len(args.inlchrate_range) < 3:
            raise ValueError("Too less argument values for the inlier ratio change rate. "
                             "Format: range_min range_max step_size v1 v2 ... vn")
        if (args.inlchrate_range[0] > args.inlchrate_range[1]) or \
                (args.inlchrate_range[2] > (args.inlchrate_range[1] - args.inlchrate_range[0])):
            raise ValueError("Parameters 2-n (inlier ratio change rate) must have the following format: "
                             "range_min range_max step_size v1 v2 ... vn")
        if not round(float((args.inlchrate_range[1] - args.inlchrate_range[0]) / args.inlchrate_range[2]), 6).is_integer():
            raise ValueError("Inlier change rate step size is wrong")
    else:
        raise ValueError("Inlier change rate or inlier ratio must be provided")
    if (args.kpAccRange[0] > args.kpAccRange[1]) or \
            (args.kpAccRange[2] > (args.kpAccRange[1] - args.kpAccRange[0])):
        raise ValueError("Parameters 5-7 (keypoint accuracy) must have the following format: "
                         "range_min range_max step_size")
    if not round(float((args.kpAccRange[1] - args.kpAccRange[0]) / args.kpAccRange[2]), 6).is_integer():
        raise ValueError("Keypoint accuracy step size is wrong")
    if not os.path.exists(args.img_path):
        raise ValueError("Image path does not exist")
    if not os.path.exists(args.store_path):
        raise ValueError("Path for storing sequences does not exist")
    # if len(os.listdir(args.store_path)) != 0:
    #     raise ValueError("Path for storing sequences is not empty")
    if args.load_path:
        if not os.path.exists(args.load_path):
            raise ValueError("Path for loading sequences does not exist")
    if args.treatTPasCorrsConfsSel and not args.treatTPasCorrs:
        raise ValueError('Specified treatTPasCorrsConfsSel but not treatTPasCorrs')
    try:
        if args.inlier_range:
            ret = gen_configs(args.path, args.path_confs_out, args.inlier_range, [], args.kpAccRange, args.img_path,
                              args.store_path, args.load_path, args.treatTPasCorrs, args.treatTPasCorrsConfsSel)
        else:
            ret = gen_configs(args.path, args.path_confs_out, [], args.inlchrate_range, args.kpAccRange, args.img_path,
                              args.store_path, args.load_path, False, None)
    except FileExistsError:
        warnings.warn(sys.exc_info()[0], UserWarning)
        sys.exit(1)
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise
    sys.exit(ret)


if __name__ == "__main__":
    main()

