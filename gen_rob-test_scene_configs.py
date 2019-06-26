"""
Generate configuration files and overview files for scenes by varying the used inlier ratio change rate and
keypoint accuracy
"""
import sys, re, numpy as np, argparse, os, pandas as pd
#from tabulate import tabulate as tab
#from copy import deepcopy

def gen_configs(input_file_name, inlier_range, inlier_values, kpAccRange, img_path, store_path, load_path):
    path, fname = os.path.split(input_file_name)
    base = ''
    ending = ''
    fnObj = re.match('(.*)_initial\.(.*)', fname, re.I)
    if fnObj:
        base = fnObj.group(1)
        ending = fnObj.group(2)
    else:
        raise ValueError('Filename must include _initial at the end before the file type')

    #nr_inliers = (inlier_range[1] - inlier_range[0])/inlier_range[2] + 1
    #nr_accs = (kpAccRange[1] - kpAccRange[0]) / kpAccRange[2] + 1

    datac = {'conf_file': [], 'img_path': [], 'img_pref': [], 'store_path': [],
             'scene_exists': [], 'load_path': [], 'parSetNr': []}
    #Generate different inlier change rates
    inl_chr = []
    for inl in np.arange(inlier_range[0], inlier_range[1] + inlier_range[2], inlier_range[2]):
        inl_chr.append(inl)
    if inlier_values:
        inl_chr = inl_chr + inlier_values

    cnt = 0
    for inl in inl_chr:
        #fnew = base + '_Inl_%.2f' % inl + '_Acc_%.2f' % kpAccRange[0]
        fnew = 'InlChR_%.2f' % inl + '_Acc_%.2f' % kpAccRange[0]
        fnew = fnew.replace('.', '_') + '.' + ending
        pfnew = os.path.join(path, fnew)
        if os.path.isfile(pfnew):
            raise ValueError('File ' + pfnew + ' already exists.')
        write_config_file(input_file_name, pfnew, inl, kpAccRange[0])
        datac['conf_file'].append(pfnew)
        datac['img_path'].append(img_path)
        datac['img_pref'].append('/')
        datac['store_path'].append(store_path)
        datac['scene_exists'].append(0)
        datac['load_path'].append(load_path)
        datac['parSetNr'].append(cnt)

        for acc in np.arange(kpAccRange[0] + kpAccRange[2], kpAccRange[1] + kpAccRange[2], kpAccRange[2]):
            #fnew = base + '_Inl_%.2f' % inl + '_Acc_%.2f' % acc
            fnew = 'InlChR_%.2f' % inl + '_Acc_%.2f' % acc
            fnew = fnew.replace('.', '_') + '.' + ending
            pfnew = os.path.join(path, fnew)
            if os.path.isfile(pfnew):
                raise ValueError('File ' + pfnew + ' already exists.')
            write_config_file(input_file_name, pfnew, inl, acc)
            datac['conf_file'].append(pfnew)
            datac['img_path'].append(img_path)
            datac['img_pref'].append('/')
            datac['store_path'].append(store_path)
            datac['scene_exists'].append(1)
            datac['load_path'].append(load_path)
            datac['parSetNr'].append(cnt)
        cnt = cnt + 1

    df = pd.DataFrame(data=datac)
    ov_file = os.path.join(path, 'config_files.csv')
    df.to_csv(index=True, sep=';', path_or_buf=ov_file)


def write_config_file(finput, foutput, inlR, acc):
    with open(foutput, 'w') as fo:
        with open(finput, 'r') as fi:
            li = fi.readline()
            founda = 0
            while li:
                lobj = re.search('inlRatChanges:', li)
                aobj = re.search('keypErrDistr:', li)
                if lobj:
                    fo.write('inlRatChanges: ' + str(inlR) + '\n')
                elif aobj:
                    founda = 1
                    fo.write(li)
                elif founda == 1:
                    aobj = re.match('(\s*first:(?:\s*))[0-9.]+', li)
                    if aobj:
                        #fo.write(aobj.group(1) + str(acc) + '\n')
                        fo.write(li)
                        founda = 2
                    else:
                        raise ValueError('Unable to write first line of keypoint accuracy')
                elif founda == 2:
                    aobj = re.match('(\s*second:(?:\s*))[0-9.]+', li)
                    if aobj:
                        fo.write(aobj.group(1) + str(acc) + '\n')
                        founda = 0
                    else:
                        raise ValueError('Unable to write second line of keypoint accuracy')
                else:
                    fo.write(li)
                li = fi.readline()


def main():
    parser = argparse.ArgumentParser(description='Generate configuration files and overview files for scenes by '
                                                 'varying the used inlier ratio and keypoint accuracy')
    parser.add_argument('--filename', type=str, required=True,
                        help='Path and filename of the template configuration file')
    parser.add_argument('--inlchrate_range', type=float, nargs='+', required=True,
                        help='Range and additional specific values for the inlier ratio change rate. '
                             'Format: min max step_size v1 v2 ... vn')
    parser.add_argument('--kpAccRange', type=float, nargs=3, required=True,
                        help='Range for the keypoint accuracy. Format: min max step_size')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to images')
    parser.add_argument('--store_path', type=str, required=True,
                        help='Storing path for generated scenes and matches')
    parser.add_argument('--load_path', type=str, required=False,
                        help='Optional loading path for generated scenes and matches. '
                             'If not provided, store_path is used.')
    args = parser.parse_args()
    if not os.path.isfile(args.filename):
        raise ValueError('File ' + args.filename + ' holding scene configuration file names does not exist')
    if len(args.inlchrate_range) < 3:
        raise ValueError("Too less argument values for the inlier ratio change rate. "
                         "Format: range_min range_max step_size v1 v2 ... vn")
    if (args.inlchrate_range[0] > args.inlchrate_range[1]) or \
            (args.inlchrate_range[2] > (args.inlchrate_range[1] - args.inlchrate_range[0])):
        raise ValueError("Parameters 2-n (inlier ratio change rate) must have the following format: "
                         "range_min range_max step_size v1 v2 ... vn")
    if not round(float((args.inlchrate_range[1] - args.inlchrate_range[0]) / args.inlchrate_range[2]), 6).is_integer():
        raise ValueError("Inlier change rate step size is wrong")
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
    if len(os.listdir(args.store_path)) != 0:
        raise ValueError("Path for storing sequences is not empty")
    if args.load_path:
        if not os.path.exists(args.load_path):
            raise ValueError("Path for loading sequences does not exist")
    else:
        args.load_path = args.store_path
    gen_configs(args.filename, args.inlchrate_range[0:3], args.inlchrate_range[3:], args.kpAccRange, args.img_path,
                args.store_path, args.load_path)


if __name__ == "__main__":
    main()

