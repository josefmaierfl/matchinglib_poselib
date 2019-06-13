"""
Execute autocalibration using different parameters and all generated sequences within a given folder.
"""
import sys, re, statistics as stat, numpy as np, math, argparse, os, pandas, cv2, time, subprocess as sp
import ruamel.yaml as yaml
import multiprocessing
import warnings
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
#from tabulate import tabulate as tab
from copy import deepcopy

def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)
yaml.SafeLoader.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)

warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)

def readOpenCVYaml(file):
    with open(file, 'r') as fi:
        data = fi.readlines()
    data = [line for line in data if line[0] is not '%']
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

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def autocalib():


def main():
    parser = argparse.ArgumentParser(description='Execute autocalibration using different parameters and all '
                                                 'generated sequences within a given folder.')
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
    parser.add_argument('--inlier_ratios', type=float, nargs='+', required=False,
                        help='Use only the given set of inlier ratios. If not provided, all available are used.')
    parser.add_argument('--kp_accs', type=float, nargs='+', required=False,
                        help='Use only the given set of keypoint accuracies. If not provided, all available are used.')
    parser.add_argument('--depths', type=str, nargs='+', required=False,
                        help='Use only the given set of depth distributions. If not provided, all available are used.')
    parser.add_argument('--kp_pos_distr', type=str, nargs='+', required=False,
                        help='Use only the given set of keypoint position distributions. '
                             'If not provided, all available are used.')
    parser.add_argument('--refineVFC', type=bool, required=False, nargs='?', default=False, const=True,
                        help='If provided, refinement of the matches with VFC is used.')
    parser.add_argument('--refineSOF', type=bool, required=False, nargs='?', default=False, const=True,
                        help='If provided, refinement of the matches with SOF is used.')
    parser.add_argument('--refineGMS', type=bool, required=False, nargs='?', default=False, const=True,
                        help='If provided, refinement of the matches with GMS is used.')
    parser.add_argument('--refineRT', type=int, nargs='+', required=True,
                        help='The first 2 values are reserved for providing specific values. If another 2 values are '
                             'provided (value range 0-1), they specify, if the 2 options should be varied. '
                             'If another 2 values are provided, option values that should not be used during '
                             'the parameter sweep can be entered. If the last value or 1 additional value is equal '
                             '-13, refineRT is not disabled if BART is disabled and vice versa. '
                             'If more than 2 values are present, the first 2 can be used to pass to option '
                             'refineRT_stereo, if refineRT_stereo is not provided.')
    parser.add_argument('--BART', type=int, nargs='+', required=False, default=[0],
                        help='If provided, the first value specifies the option BART and the optional second value '
                             '(range 0-1) specifies if a parameter sweep should be performed. '
                             'If 2 values are present, the first one can be used to pass to option '
                             'BART_stereo, if BART_stereo is not provided.')
    parser.add_argument('--RobMethod', type=str, nargs='+', required=False, default=['USAC'],
                        help='If provided, the list of strings specify the used robust estimation methods.')
    parser.add_argument('--cfgUSAC', type=int, nargs='+', required=True,
                        help='The first 6 values are reserved for providing specific values. If another 6 values are '
                             'provided (value range 0-1), they specify, if the 6 options should be varied. '
                             'If another 6 values are provided, option values that should not be used during '
                             'the parameter sweep can be entered. '
                             'If more than 6 values are present, the first 6 are ignored.')
    parser.add_argument('--USACInlratFilt', type=int, required=True,
                        help='If the specified value is >=0 and <2, the given filter (0 = GMS, 1 = VFC) is used. '
                             'For a value >=2, a parameter sweep is performed on the possible inputs.')
    parser.add_argument('--th', type=float, nargs='+', required=True,
                        help='If only 1 value is provided, it specifies a specific inlier threshold. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--stereoRef', type=bool, nargs='?', required=False, default=False, const=True,
                        help='If provided, correspondence aggregation is performed. Otherwise, the extrinsics are '
                             'only calculated based on the data of a single stereo frame.')
    parser.add_argument('--refineRT_stereo', type=int, nargs='+', required=False,
                        help='The first 2 values are reserved for providing specific values. If another 2 values are '
                             'provided (value range 0-1), they specify, if the 2 options should be varied. '
                             'If another 2 values are provided, option values that should not be used during '
                             'the parameter sweep can be entered. If the last value or 1 additional value is equal '
                             '-13, refineRT is not disabled if BART is disabled and vice versa. '
                             'If more than 2 values are present, the first 2 are ignored. '
                             'If the argument is not provided, the values from refineRT are used.')
    parser.add_argument('--BART_stereo', type=int, nargs='+', required=False,
                        help='If provided, the first value specifies the option BART and the optional second value '
                             '(range 0-1) specifies if a parameter sweep should be performed. '
                             'If 2 values are present, the first one is ignored. '
                             'If the argument is not provided, the values from BART are used.')
    parser.add_argument('--minStartAggInlRat', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, it specifies a specific minimum inlier ratio. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--relInlRatThLast', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, it specifies a specific Maximum relative change between '
                             'the inlier ratio with the last E and the new robustly estimated E. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--relInlRatThNew', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--minInlierRatSkip', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--relMinInlierRatSkip', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--maxSkipPairs', type=int, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--minInlierRatioReInit', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--minPtsDistance', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--maxPoolCorrespondences', type=int, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 * n values '
                             'are provided, ranges can be specified: '
                             'Format: min1 max1 step_size1 min2 max2 step_size2 ... minn maxn step_sizen')
    parser.add_argument('--minContStablePoses', type=int, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--absThRankingStable', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--useRANSAC_fewMatches', type=bool, nargs='?', required=False, default=False, const=True,
                        help='If provided, correspondence aggregation is performed. Otherwise, the extrinsics are '
                             'only calculated based on the data of a single stereo frame.')
    parser.add_argument('--checkPoolPoseRobust', type=int, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--minNormDistStable', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--raiseSkipCnt', type=int, nargs=2, required=False, default=[1, 6],
                        help='2 values have to be provided. No parameter sweep possible.')
    parser.add_argument('--maxRat3DPtsFar', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--maxDist3DPtsZ', type=float, nargs='+', required=False,
                        help='If only 1 value is provided, this specific value is used. If 3 values '
                             'are provided, a range can be specified: Format: min max step_size')
    parser.add_argument('--useGTCamMat', type=bool, required=False, nargs='?', default=False, const=True,
                        help='If provided, the GT camera matrices are always used and the distorted ones are ignored.')
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
    if not os.path.exists(args.output_path):
        raise ValueError('Directory ' + args.output_path + ' does not exist')

    #Build command lines
    cmds = []
    if args.refineVFC:
        cmds.append(['--refineVFC'])
    if args.refineSOF:
        cmds.append(['--refineSOF'])
    if args.refineGMS:
        cmds.append(['--refineGMS'])
    if len(args.refineRT) < 2 or len(args.refineRT) > 7:
        raise ValueError('Wrong number of arguments for refineRT')
    if len(args.refineRT) == 2:
        if args.refineRT[0] < 0 or args.refineRT[0] > 6:
            raise ValueError('First value for refineRT out of range')
        if args.refineRT[1] < 0 or args.refineRT[1] > 2:
            raise ValueError('Second value for refineRT out of range')
        if cmds:
            for it in cmds:
                it.append('--refineRT')
                it.append(''.join(map(str, args.refineRT)))
        else:
            cmds.append(['--refineRT', ''.join(map(str, args.refineRT))])
    elif len(args.refineRT) > 3:
        missdig = [-1, -1]
        bartrefRTdis = False
        if args.refineRT[-1] == -13:
            bartrefRTdis = True
        if len(args.refineRT) > 5:
            if args.refineRT[5] >= 0 and args.refineRT[5] < 3:
                missdig[1] = args.refineRT[5]
        elif len(args.refineRT) > 4:
            if args.refineRT[4] >= 0 and args.refineRT[5] < 7:
                missdig[0] = args.refineRT[4]
        if args.refineRT[2] > 0:
            refineRT_tmp = [[i] for i in range(0,7) if i != missdig[0]]
        else:
            if args.refineRT[0] < 0 or args.refineRT[0] > 6:
                raise ValueError('First value for refineRT out of range')
            refineRT_tmp = [[args.refineRT[0]]]
        if args.refineRT[3] > 0:
            refineRT_tmp1 = deepcopy(refineRT_tmp)
            refineRT_tmp = []
            for i in range(0, 3):
                if i != missdig[1]:
                    for j in refineRT_tmp1:
                        refineRT_tmp.append([j[0], i])
        else:
            if args.refineRT[1] < 0 or args.refineRT[1] > 2:
                raise ValueError('Second value for refineRT out of range')
            for j in refineRT_tmp:
                j.append(args.refineRT[1])


    return autocalib(args.path, args.executable, cpu_use, args.message_path)

if __name__ == "__main__":
    main()
