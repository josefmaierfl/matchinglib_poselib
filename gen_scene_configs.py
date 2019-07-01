"""
Generate configuration files and overview files for scenes by varying the used inlier ratio and keypoint accuracy
"""
import sys, re, numpy as np, argparse, os, pandas as pd
#from tabulate import tabulate as tab
#from copy import deepcopy

def gen_configs(input_file_name, inlier_range, kpAccRange, img_path, store_path, load_path, treatTPasCorrs):
    path, fname = os.path.split(input_file_name)
    base = ''
    ending = ''
    fnObj = re.match(r'(.*)_initial\.(.*)', fname, re.I)
    if fnObj:
        base = fnObj.group(1)
        ending = fnObj.group(2)
    else:
        raise ValueError('Filename must include _initial at the end before the file type')

    #nr_inliers = (inlier_range[1] - inlier_range[0])/inlier_range[2] + 1
    #nr_accs = (kpAccRange[1] - kpAccRange[0]) / kpAccRange[2] + 1

    datac = {'conf_file': [], 'img_path': [], 'img_pref': [], 'store_path': [],
             'scene_exists': [], 'load_path': [], 'parSetNr': []}
    cnt = 0
    tp = []
    for inl in np.arange(inlier_range[0], inlier_range[1] + inlier_range[2], inlier_range[2]):
        #fnew = base + '_Inl_%.2f' % inl + '_Acc_%.2f' % kpAccRange[0]
        fnew = 'Inl_%.2f' % inl + '_Acc_%.2f' % kpAccRange[0]
        fnew = fnew.replace('.', '_') + '.' + ending
        pfnew = os.path.join(path, fnew)
        if os.path.isfile(pfnew):
            raise ValueError('File ' + pfnew + ' already exists.')
        if treatTPasCorrs:
            tp = getTPRange(fname, inl)
        write_config_file(input_file_name, pfnew, round(inl, 3), round(kpAccRange[0] / 2, 3), tp)
        datac['conf_file'].append(pfnew)
        datac['img_path'].append(img_path)
        datac['img_pref'].append('/')
        datac['store_path'].append(store_path)
        datac['scene_exists'].append(0)
        datac['load_path'].append(load_path)
        datac['parSetNr'].append(cnt)

        for acc in np.arange(kpAccRange[0] + kpAccRange[2], kpAccRange[1] + kpAccRange[2], kpAccRange[2]):
            #fnew = base + '_Inl_%.2f' % inl + '_Acc_%.2f' % acc
            fnew = 'Inl_%.2f' % inl + '_Acc_%.2f' % acc
            fnew = fnew.replace('.', '_') + '.' + ending
            pfnew = os.path.join(path, fnew)
            if os.path.isfile(pfnew):
                raise ValueError('File ' + pfnew + ' already exists.')
            if treatTPasCorrs:
                tp = getTPRange(fname, inl)
            write_config_file(input_file_name, pfnew, round(inl, 3), round(acc / 2, 3), tp)
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
    #cf = pandas.read_csv(input_file_name, delimiter=';')


def getTPRange(filename, inl):
    fnObj = re.match(r'.*_kp-distr-(?:(?:half-img)|(?:1corn)|(?:equ))_depth-(?:F|(?:NM)|(?:NMF))_TP-([0-9to]+).*',
                     filename)
    if fnObj:
        if fnObj.group(1):
            tp_nr = fnObj.group(1)
        else:
            raise ValueError('Unable to extract number of keypoints from ' + filename)
    else:
        raise ValueError('Unable to extract number of keypoints from ' + filename)
    fnObj = re.match(r'(\d+)to(\d+)', tp_nr)
    if fnObj:
        if fnObj.group(1):
            tp_nr1 = int(fnObj.group(1))
        else:
            raise ValueError('Unable to extract number of keypoints from ' + filename)
        if fnObj.group(2):
            tp_nr2 = int(fnObj.group(2))
        else:
            raise ValueError('Unable to extract number of keypoints from ' + filename)
        tp_nr1 = int(round(float(tp_nr1) * inl, 0))
        tp_nr2 = int(round(float(tp_nr2) * inl, 0))
    else:
        tp_nr1 = int(round(float(tp_nr) * inl, 0))
        tp_nr2 = tp_nr1

    return [tp_nr1, tp_nr2]


def write_config_file(finput, foutput, inlR, acc, tp):
    with open(foutput, 'w') as fo:
        with open(finput, 'r') as fi:
            li = fi.readline()
            foundi = 0
            founda = 0
            foundtp = 0
            while li:
                lobj = re.search('inlRatRange:', li)
                aobj = re.search('keypErrDistr:', li)
                if tp:
                    tpobj = re.search('truePosRange:', li)
                else:
                    tpobj = None
                if lobj:
                    foundi = 1
                    fo.write(li)
                elif aobj:
                    founda = 1
                    fo.write(li)
                elif tpobj:
                    foundtp = 1
                    fo.write(li)
                elif foundi == 1:
                    lobj = re.match(r'(\s*first:(?:\s*))[0-9.]+', li)
                    if lobj:
                        fo.write(lobj.group(1) + str(inlR) + '\n')
                        foundi = 2
                    else:
                        raise  ValueError('Unable to write first line of inlier ratio')
                elif foundi == 2:
                    lobj = re.match(r'(\s*second:(?:\s*))[0-9.]+', li)
                    if lobj:
                        fo.write(lobj.group(1) + str(inlR) + '\n')
                        foundi = 0
                    else:
                        raise ValueError('Unable to write second line of inlier ratio')
                elif founda == 1:
                    aobj = re.match(r'(\s*first:(?:\s*))[0-9.]+', li)
                    if aobj:
                        #fo.write(aobj.group(1) + str(acc) + '\n')
                        fo.write(li)
                        founda = 2
                    else:
                        raise ValueError('Unable to write first line of keypoint accuracy')
                elif founda == 2:
                    aobj = re.match(r'(\s*second:(?:\s*))[0-9.]+', li)
                    if aobj:
                        fo.write(aobj.group(1) + str(acc) + '\n')
                        founda = 0
                    else:
                        raise ValueError('Unable to write second line of keypoint accuracy')
                elif foundtp == 1:
                    rpstr = r'\g<1>{}'.format(tp[0])
                    li1 = re.sub(r'(\s*first:\s*)\d+',
                                 rpstr, li)
                    fo.write(li1)
                    foundtp = 2
                elif foundtp == 2:
                    rpstr = r'\g<1>{}'.format(tp[1])
                    li1 = re.sub(r'(\s*second:\s*)\d+',
                                 rpstr, li)
                    fo.write(li1)
                    foundtp = 0
                else:
                    fo.write(li)
                li = fi.readline()


def main():
    parser = argparse.ArgumentParser(description='Generate configuration files and overview files for scenes by '
                                                 'varying the used inlier ratio and keypoint accuracy')
    parser.add_argument('--filename', type=str, required=True,
                        help='Path and filename of the template configuration file')
    parser.add_argument('--inlier_range', type=float, nargs=3, required=True,
                        help='Range for the inlier ratio. Format: min max step_size')
    parser.add_argument('--kpAccRange', type=float, nargs=3, required=True,
                        help='Range for the keypoint accuracy. The entered value is devided by 2 ro reach an '
                             'approximate maximum keypoint accuracy based on the given value as the value in '
                             'the configuration file corresponds to the standard deviation. '
                             'Format: min max step_size')
    parser.add_argument('--treatTPasCorrs', type=bool, nargs='?', required=True, default=False, const=True,
                        help='If provided, the number of TP is calculated and written to the config file based on '
                             'the given inlier ratios and the number or range of desired correspondences (TP+FP) '
                             'that is extracted from the filename (*_TP-?*).')
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
    if (args.inlier_range[0] > args.inlier_range[1]) or \
            (args.inlier_range[2] > (args.inlier_range[1] - args.inlier_range[0])):
        raise ValueError("Parameters 2-4 (inlier ratio) must have the following format: "
                         "range_min range_max step_size")
    if not round(float((args.inlier_range[1] - args.inlier_range[0]) / args.inlier_range[2]), 6).is_integer():
        raise ValueError("Inlier step size is wrong")
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
    gen_configs(args.filename, args.inlier_range, args.kpAccRange, args.img_path, args.store_path, args.load_path,
                args.treatTPasCorrs)


if __name__ == "__main__":
    main()

