"""
Load configuration files and generate different scenes for testing using multiple CPU cores
"""
import sys, re, statistics as stat, numpy as np, math, argparse, os, pandas, cv2, time, yaml, subprocess as sp
from multiprocessing import Pool
#from tabulate import tabulate as tab
from copy import deepcopy

def genScenes(input_path, executable, nr_cpus, message_path):
    dirs_f = os.path.join(input_path, 'generated_dirs_config.txt')
    if not os.path.exists(dirs_f):
        raise ValueError("Unable to load " + dirs_f)
    dirsc = []
    with open(dirs_f, 'r') as fi:
        #Load directory holding configuration files
        confd = fi.readline()
        while confd:
            if not os.path.exists(confd):
                raise ValueError("Directory " + confd + " does not exist.")
            #Load list of configuration files and settings for generating scenes
            ovcf = os.path.join(confd, 'config_files.csv')
            if not os.path.exists(ovcf):
                raise ValueError("File " + ovcf + " does not exist.")
            dirsc.append(ovcf)
            confd = fi.readline()
    maxd_parallel = int(len(dirsc) / nr_cpus)
    if maxd_parallel == 0:
        maxd_parallel = 1
    nr_used_cpus = int(len(dirsc) / maxd_parallel)
    dirscp = []
    rest = len(dirsc) - nr_used_cpus * maxd_parallel
    cr = 0
    rest_sv = rest
    for i in range(0, len(dirsc) - rest_sv, maxd_parallel):
        if rest > 0:
            dirscp.append(dirsc[(i + cr):(i + cr + maxd_parallel + 1)])
            cr = cr + 1
            rest = rest - 1
        else:
            dirscp.append(dirsc[(i + cr):(i + cr + maxd_parallel)])
    assert len(dirscp) == nr_used_cpus
    cpus_rest = nr_cpus - nr_used_cpus
    if cpus_rest == 0:
        cpus_rest = [1] * nr_used_cpus
    else:
        sub_cpus = nr_cpus / nr_used_cpus
        cpus_rest = [int(sub_cpus)] * nr_used_cpus
        for i in range(0, int(round((sub_cpus - math.floor(sub_cpus)) * nr_used_cpus))):
            cpus_rest[i] = cpus_rest[i] + 1
    work_items = [(dirscp[x], cpus_rest[x], executable, message_path) for x in range(0, nr_used_cpus)]
    with Pool(processes=nr_used_cpus) as pool:
        results = pool.map(processDir, work_items)


def processDir(dirs_list, cpus_rest, executable, message_path):
    for ovcf in dirs_list:
        cf = pandas.read_csv(ovcf, delimiter=';')
        if cf.empty:
            print("File " + ovcf + " is empty.")
            continue
        err_cnt = 0
        #Split entries into tasks for generating initial sequences and matches only
        c_sequ = cf.loc[cf['scene_exists'] == 0]
        c_match = cf.loc[cf['scene_exists'] == 1]

        #Calculate sequences first using multiple CPUs
        maxd_parallel1 = int(c_sequ.shape[0] / cpus_rest)
        if maxd_parallel1 == 0:
            maxd_parallel1 = 1
        nr_used_cpus1 = int(c_sequ.shape[0] / maxd_parallel1)

        #Generate unique path for storing messages
        dirn = os.path.dirname(ovcf)
        sub_dirs = re.findall(r'[\w-]+', dirn)
        if sub_dirs:
            base = sub_dirs[-1]
        else:
            print('Unable to extract last sub-directory name of ', ovcf)
            continue
        mess_new = os.path.join(message_path, base)
        try:
            os.mkdir(mess_new)
        except FileExistsError:
            print('Directory ', mess_new, ' already exists.')
            continue

        cmds = []
        for index, row in c_sequ.iterrows():
            if not os.path.exists(row['conf_file']):
                raise ValueError("Configuration file " + row['conf_file'] + " does not exist.")
            if not os.path.exists(row['store_path']):
                raise ValueError("Save path " + row['store_path'] + " does not exist.")
            cmds.append(([executable,
                         '--img_path', row['img_path'],
                         '--img_pref', row['img_pref'],
                         '--store_path', row['store_path'],
                         '--conf_file', row['conf_file']], int(row['parSetNr']), mess_new))

        with Pool(processes=nr_used_cpus1) as pool:
            results = [pool.apply_async(processSequences, t) for t in cmds]

        for index, row in cf.iterrows():
            #Check if we have to generate the scene and the matches or load the scene and generate only matches
            if not os.path.exists(row['conf_file']):
                raise ValueError("Configuration file " + row['conf_file'] + " does not exist.")
            if not os.path.exists(row['store_path']):
                raise ValueError("Save path " + row['store_path'] + " does not exist.")
            if int(row['scene_exists']) == 0:
                #generate scene and matches
                cmd_l = [executable,
                         '--img_path', row['img_path'],
                         '--img_pref', row['img_pref'],
                         '--store_path', row['store_path'],
                         '--conf_file', row['conf_file']]
            else:
                #get the path for the stored sequence
                ovs_f = os.path.join(row['load_path'], 'sequInfos.yaml')
                if not os.path.exists(ovs_f):
                    raise ValueError("Sequence overview file " + ovs_f + " does not exist.")
                fs_read = cv2.FileStorage(ovs_f, cv2.FILE_STORAGE_READ)
                if not fs_read.isOpened():
                    raise ValueError('Unable to read scenes overview file.')
                fn = fs_read.getNode('parSetNr' + str(int(row['parSetNr']) + err_cnt))
                cv2.FileNode.string()
                if fn.empty():
                    raise ValueError('parSetNr' + str(int(row['parSetNr']) + err_cnt) + ' not found.')
                sub_path = fn.getNode('hashSequencePars').string()
                load_path = os.path.join(row['load_path'], sub_path)
                if not os.path.exists(load_path):
                    raise ValueError("Sequence path " + load_path + " does not exist.")
                cmd_l = [executable,
                         '--img_path', row['img_path'],
                         '--img_pref', row['img_pref'],
                         '--store_path', '*',
                         '--conf_file', row['conf_file'],
                         '--load_folder', load_path]


def processSequences(cmd_l, parSetNr, message_path):
    #Check if we have to wait until other sequence generation processes have finished writing into the overview file
    ov_file = os.path.join(cmd_l[6], 'sequInfos.yaml')
    if parSetNr != 0:
        time.sleep(float(parSetNr * 10))
        cnt = 0
        while not os.path.exists(ov_file) and cnt < 20:
            time.sleep(10)
            cnt = cnt + 1
        if cnt == 20:
            return ['noExe']
        data = readOpenCVYaml(ov_file)
        cnt = 0
        while not data and cnt < 20:
            time.sleep(1)
            data = readOpenCVYaml(ov_file)
            cnt = cnt + 1
        if cnt == 20:
            return ['noExe']
        data_set = data['parSetNr' + str(int(parSetNr - 1))]
        cnt = 0
        while not data_set and cnt < 20:
            time.sleep(10)
            data = readOpenCVYaml(ov_file)
            data_set = data['parSetNr' + str(int(parSetNr - 1))]
            cnt = cnt + 1
        if cnt == 20:
            return ['noExe']
    elif os.path.exists(ov_file):
        raise FileExistsError

    #Generate sequence
    cf_name = os.path.basename(cmd_l[8])
    fnObj = re.match('(.+)\..+', cf_name, re.I)
    if fnObj:
        base = fnObj.group(1)
    else:
        return ['badName']
    err_out = 'stderr_' + base + '.txt'
    mess_out = 'stdout_' + base + '.txt'
    fname_cerr = os.path.join(message_path, err_out)
    fname_mess = os.path.join(message_path, mess_out)
    cerrf = open(fname_cerr, 'w')
    messf = open(fname_mess, 'w')
    cnt3 = 3
    result = None
    while cnt3 > 0:
        try:
            sp.run(cmd_l, stdout=messf, stderr=cerrf, check=True)
            result = [cmd_l, parSetNr]
            cnt3 = 0
        except sp.CalledProcessError as e:
            cnt3 = cnt3 - 1
            print('Failed to generate scene!')
            if abs(e.returncode) != 2:
                raise ChildProcessError
            if cnt3 > 0:
                print('Trying again!')
                cerrf.close()
                messf.close()
                fname_cerr_err = os.path.join(message_path, 'err_' + str(cnt3) + '_' + err_out)
                fname_mess_err = os.path.join(message_path, 'err_' + str(cnt3) + '_' + mess_out)
                os.rename(fname_cerr, fname_cerr_err)
                os.rename(fname_mess, fname_mess_err)
                cerrf = open(fname_cerr, 'w')
                messf = open(fname_mess, 'w')
                continue
            err_filen = 'errorInfo_' + base + '.txt'
            fname_err = os.path.join(message_path, err_filen)
            if e.cmd or e.stderr or e.stdout:
                with open(fname_err, 'w') as fo1:
                    if e.cmd:
                        for line in e.cmd:
                            fo1.write(line + ' ')
                        fo1.write('\n\n')
                    if e.stderr:
                        for line in e.stderr:
                            fo1.write(line + ' ')
                        fo1.write('\n\n')
                    if e.stdout:
                        for line in e.stdout:
                            fo1.write(line + ' ')
            cerrf.close()
            messf.close()
            raise ChildProcessError
    cerrf.close()
    messf.close()
    return result


def readOpenCVYaml(file):
    with open(file, 'r') as fi:
        data = fi.readlines()
    data = [line for line in data if line[0] is not '%']
    data = yaml.safe_load("\n".join(data))
    return data

def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)


def main():
    parser = argparse.ArgumentParser(description='Generate multiple scenes and matches from configuration files')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory holding file \'generated_dirs_config.txt\'')
    parser.add_argument('--nrCPUs', type=int, nargs=1, required=False, default=4,
                        help='Number of CPU cores for parallel processing. If a negative value is provided, '
                             'the program tries to find the number of available CPUs on the system - if it fails, '
                             'the absolute value of nrCPUs is used. Default: 4')
    parser.add_argument('--executable', type=str, required=True,
                        help='Executable of the application generating the sequences')
    parser.add_argument('--message_path', type=str, required=True,
                        help='Storing path for text files containing error and normal mesages during the '
                             'generation process of scenes and matches')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' does not exist')
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

    genScenes(args.path, args.executable, cpu_use, args.message_path)

if __name__ == "__main__":
    main()


