"""
Loads test results and calls specific functions for evaluation as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np
import ruamel.yaml as yaml
import modin.pandas as mpd
import pandas as pd

#warnings.simplefilter('ignore', category=UserWarning)

def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)
yaml.SafeLoader.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)

warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

def readOpenCVYaml(file, isstr = False):
    if isstr:
        data = file.split('\n')
    else:
        with open(file, 'r') as fi:
            data = fi.readlines()
    data = [line for line in data if line and line[0] is not '%']
    try:
        data = yaml.load_all("\n".join(data))
        data1 = {}
        for d in data:
            data1.update(d)
        data = data1
    except:
        print('Exception during reading yaml.')
        e = sys.exc_info()
        print(str(e))
        sys.stdout.flush()
        raise BaseException
    return data


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def eval_test(load_path, output_path, test_name, test_nr, cpu_use):
    #Load test results
    res_path = os.path.join(load_path, 'results')
    if not os.path.exists(res_path):
        raise ValueError('No results folder found')
    # Get all folders that contain data
    sub_dirs = [name for name in os.listdir(res_path) if os.path.isdir(os.path.join(res_path, name))]
    sub_dirs = [name for name in sub_dirs if RepresentsInt(name)]
    if len(sub_dirs) == 0:
        raise ValueError('No subdirectories holding data found')
    # data = pd.DataFrame
    data_list = []
    for sf in sub_dirs:
        res_path_it = os.path.join(res_path, sf)
        ov_file = os.path.join(res_path_it, 'allRunsOverview.yaml')
        if not os.path.exists(ov_file):
            raise ValueError('Results overview file allRunsOverview.yaml not found')
        par_data = readOpenCVYaml(ov_file)
        parSetNr = 0
        while True:
            try:
                data_set = par_data['parSetNr' + str(int(parSetNr))]
                parSetNr += 1
            except:
                break
            csvf = os.path.join(res_path_it, data_set['hashTestingPars'])
            if not os.path.exists(csvf):
                raise ValueError('Results file ' + csvf + ' not found')
            csv_data = pd.read_csv(csvf, delimiter=';')
            print('Loaded', csvf, 'with shape', csv_data.shape)
            #csv_data.set_index('Nr')
            addSequInfo_sep = None
            # for idx, row in csv_data.iterrows():
            for row in csv_data.itertuples():
                #tmp = row['addSequInfo'].split('_')
                tmp = row.addSequInfo.split('_')
                tmp = dict([(tmp[x], tmp[x + 1]) for x in range(0,len(tmp), 2)])
                if addSequInfo_sep:
                    for k in addSequInfo_sep.keys():
                        addSequInfo_sep[k].append(tmp[k])
                else:
                    for k in tmp.keys():
                        tmp[k] = [tmp[k]]
                    addSequInfo_sep = tmp
            addSequInfo_df = pd.DataFrame(data=addSequInfo_sep)
            csv_data = pd.concat([csv_data, addSequInfo_df], axis=1, sort=False, join_axes=[csv_data.index])
            csv_data.drop(columns=['addSequInfo'], inplace=True)
            data_set_tmp = merge_dicts(data_set)
            data_set_tmp = pd.DataFrame(data=data_set_tmp, index=[0])
            data_set_repl = pd.DataFrame(np.repeat(data_set_tmp.values, csv_data.shape[0], axis=0))
            data_set_repl.columns = data_set_tmp.columns
            csv_new = pd.concat([csv_data, data_set_repl], axis=1, sort=False, join_axes=[csv_data.index])
            data_list.append(csv_new)
            # if data.empty:
            #     # data = mpd.utils.from_pandas(csv_new)
            #     data = csv_new
            # else:
            #     # data = mpd.concat([data,mpd.utils.from_pandas(csv_new)],
            #     #                   ignore_index=True,
            #     #                   sort=False,
            #     #                   copy=False)
            #     data = pd.concat([data, csv_new], ignore_index=True, sort=False, copy=False)
    data = pd.concat(data_list, ignore_index=True, sort=False, copy=False)
    data_dict = data.to_dict()
    data = mpd.DataFrame(data_dict)
    print('Finished loading data')
    if test_name == 'testing_tests':#'usac-testing':
        if not test_nr:
            raise ValueError('test_nr is required for usac-testing')
        from usac_tests import calcSatisticRt_th
        if test_nr == 1:
            return calcSatisticRt_th(data, output_path)


def merge_dicts(in_dict, mainkey = None):
    tmp = {}
    for i in in_dict.keys():
        if isinstance(in_dict[i], dict):
            tmp1 = merge_dicts(in_dict[i], i)
            for j in tmp1.keys():
                if mainkey is not None:
                    tmp[mainkey + '_' + j] = tmp1[j]
                else:
                    tmp.update({j: tmp1[j]})
        else:
            if mainkey is not None:
                tmp.update({mainkey + '_' + i: in_dict[i]})
            else:
                tmp.update({i: in_dict[i]})
    return tmp


def main():
    parser = argparse.ArgumentParser(description='Loads test results and calls specific functions for evaluation '
                                                 'as specified in file Autocalibration-Parametersweep-Testing.xlsx')
    parser.add_argument('--path', type=str, required=True,
                        help='Main directory holding test results. This directory must hold subdirectories with the '
                             'different test names')
    parser.add_argument('--output_path', type=str, required=False,
                        help='Optional output directory. By default, (if this option is not provided, the data is '
                             'stored in a new directory inside the directory for a specific test.')
    parser.add_argument('--test_name', type=str, required=True,
                        help='Name of the main test like \'USAC-testing\' or \'USAC_vs_RANSAC\'')
    parser.add_argument('--test_nr', type=int, required=False,
                        help='Test number within the main test specified by test_name starting with 1')
    parser.add_argument('--nrCPUs', type=int, required=False, default=-8,
                        help='Number of CPU cores for parallel processing. If a negative value is provided, '
                             'the program tries to find the number of available CPUs on the system - if it fails, '
                             'the absolute value of nrCPUs is used. Default: -8')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise ValueError('Main directory not found')
    test_name = args.test_name.lower()
    load_path = os.path.join(args.path, args.test_name)
    if not os.path.exists(load_path):
        raise ValueError('Specific main test directory ' + load_path + '  not found')
    if args.test_nr:
        load_path = os.path.join(load_path, str(args.test_nr))
        if not os.path.exists(load_path):
            raise ValueError('Specific test directory ' + load_path + '  not found')
    if args.output_path:
        if not os.path.exists(args.output_path):
            raise ValueError('Specified output directory not found')
        output_path =  args.output_path
    else:
        output_path = os.path.join(load_path, 'evals')
        try:
            os.mkdir(output_path)
        except FileExistsError:
            raise ValueError('Directory ' + output_path + ' already exists')
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

    return eval_test(load_path, output_path, test_name, args.test_nr, cpu_use)


if __name__ == "__main__":
    main()