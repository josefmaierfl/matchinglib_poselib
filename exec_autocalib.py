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
                             'refineRT_stereo, if refineRT_stereo is not provided. '
                             'For sub-parameters where no sweep should be performed, the corresponding value '
                             'within the first 2 values is selected.')
    parser.add_argument('--BART', type=int, nargs='+', required=False, default=[0],
                        help='If provided, the first value specifies the option BART and the optional second value '
                             ' (its value does not matter) specifies if a parameter sweep should be performed. '
                             'If 2 values are present, the first one can be used to pass to option '
                             'BART_stereo, if BART_stereo is not provided.')
    parser.add_argument('--RobMethod', type=str, nargs='+', required=False, default=['USAC'],
                        help='If provided, the list of strings specify the used robust estimation methods.')
    parser.add_argument('--cfgUSAC', type=int, nargs='+', required=True,
                        help='The first 6 values are reserved for providing specific values. If another 6 values are '
                             'provided (value range 0-1), they specify, if the 6 options should be varied. '
                             'If another 6 values are provided, option values that should not be used during '
                             'the parameter sweep can be entered. '
                             'For sub-parameters where no sweep should be performed, the corresponding value '
                             'within the first 6 values is selected.')
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
                        help='For stereo refinement: If provided, RANSAC for robust estimation '
                             'if less than 100 matches are available is used.')
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
    bartrefRTdis = False
    refineRT_tmp = None
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
        itdig2 = True
        if args.refineRT[-1] == -13:
            bartrefRTdis = True
        if len(args.refineRT) > 5:
            if args.refineRT[5] >= 0 and args.refineRT[5] < 3:
                missdig[1] = args.refineRT[5]
            if args.refineRT[4] >= 0 and args.refineRT[4] < 7:
                missdig[0] = args.refineRT[4]
        elif len(args.refineRT) > 4:
            if args.refineRT[4] >= 0 and args.refineRT[4] < 7:
                missdig[0] = args.refineRT[4]
        if args.refineRT[2] > 0:
            refineRT_tmp = [[i] for i in range(0,7) if i != missdig[0]]
        else:
            if args.refineRT[0] < 0 or args.refineRT[0] > 6:
                raise ValueError('First value for refineRT out of range')
            refineRT_tmp = [[args.refineRT[0]]]
            if args.refineRT[0] == 0:
                itdig2 = False
        if args.refineRT[3] > 0 and itdig2:
            refineRT_tmp1 = deepcopy(refineRT_tmp)
            refineRT_tmp = [[0, 0]]
            for i in range(0, 3):
                if i != missdig[1]:
                    for j in refineRT_tmp1:
                        if j[0] != 0:
                            refineRT_tmp.append([j[0], i])
        elif not itdig2:
            refineRT_tmp[0].append(0)
        else:
            if args.refineRT[1] < 0 or args.refineRT[1] > 2:
                raise ValueError('Second value for refineRT out of range')
            for j in refineRT_tmp:
                j.append(args.refineRT[1])
    else:
        raise ValueError('Wrong number of arguments for refineRT')

    if len(args.BART) > 2:
        raise ValueError('Wrong number of arguments for BART')
    ref_bart = []
    if len(args.BART) == 1:
        if args.BART[0] < 0 or args.BART[0] > 2:
            raise ValueError('Value for BART out of range')
        if args.BART[0] == 0 and bartrefRTdis:
            raise ValueError('Bundle adjustment is disabled but it was specified in refineRT that it is not allowed to '
                             'be disabled if refineRT is disabled.')
        if refineRT_tmp:
            for it in refineRT_tmp:
                ref_bart.append(['--refineRT', ''.join(map(str, it)), '--BART', str(args.BART[0])])
        else:
            ref_bart = [['--BART', str(args.BART[0])]]
    else:
        if not refineRT_tmp:
            ref_bart = [['--BART', str(i)] for i in range(0, 3)]
        else:
            for it in refineRT_tmp:
                for i in range(0, 3):
                    if not bartrefRTdis or not (i == 0 and it[0] == 0 and bartrefRTdis):
                        ref_bart.append(['--refineRT', ''.join(map(str, it)), '--BART', str(i)])
    if not cmds:
        cmds = ref_bart
    else:
        cmds_rb = deepcopy(cmds)
        cmds = []
        for it in cmds_rb:
            for it1 in ref_bart:
                cmds.append(it + it1)

    if len(args.RobMethod) == 1:
        if args.RobMethod[0] != 'USAC' and args.RobMethod[0] != 'RANSAC':
            raise ValueError('Wrong argument for RobMethod')
        for it in cmds:
            it.extend(['--RobMethod', args.RobMethod[0]])
    else:
        cmds_rob = deepcopy(cmds)
        cmds = []
        for it in cmds_rob:
            for it1 in args.RobMethod:
                if it1 != 'USAC' and it1 != 'RANSAC':
                    raise ValueError('Wrong arguments for RobMethod')
                cmds.append(it + ['--RobMethod', it1])

    if len(args.cfgUSAC) < 6 or len(args.cfgUSAC) > 18:
        raise ValueError('Wrong number of arguments for cfgUSAC')
    cfgUSAC_tmp = None
    if len(args.cfgUSAC) == 6:
        if args.cfgUSAC[0] < 0 or args.cfgUSAC[0] > 3:
            raise ValueError('First value for cfgUSAC out of range')
        if args.cfgUSAC[1] < 0 or args.cfgUSAC[1] > 1:
            raise ValueError('Second value for cfgUSAC out of range')
        if args.cfgUSAC[2] < 0 or args.cfgUSAC[2] > 1:
            raise ValueError('Third value for cfgUSAC out of range')
        if args.cfgUSAC[3] < 0 or args.cfgUSAC[3] > 2:
            raise ValueError('4th value for cfgUSAC out of range')
        if args.cfgUSAC[4] < 0 or args.cfgUSAC[4] > 2:
            raise ValueError('5th value for cfgUSAC out of range')
        if args.cfgUSAC[5] < 0 or args.cfgUSAC[5] > 7:
            raise ValueError('6th value for cfgUSAC out of range')
        for it in cmds:
            it.extend(['--cfgUSAC', ''.join(map(str, args.cfgUSAC))])
    elif len(args.cfgUSAC) > 6:
        missdig = [-1] * 6
        if len(args.cfgUSAC) == 18:
            if args.cfgUSAC[12] >= 0 and args.cfgUSAC[12] < 4:
                missdig[0] = args.cfgUSAC[12]
            if args.cfgUSAC[13] >= 0 and args.cfgUSAC[13] < 2:
                missdig[1] = args.cfgUSAC[13]
            if args.cfgUSAC[14] >= 0 and args.cfgUSAC[14] < 2:
                missdig[2] = args.cfgUSAC[14]
            if args.cfgUSAC[15] >= 0 and args.cfgUSAC[15] < 3:
                missdig[3] = args.cfgUSAC[15]
            if args.cfgUSAC[16] >= 0 and args.cfgUSAC[16] < 3:
                missdig[4] = args.cfgUSAC[16]
            if args.cfgUSAC[17] >= 0 and args.cfgUSAC[17] < 8:
                missdig[5] = args.cfgUSAC[17]
        elif len(args.cfgUSAC) != 12:
            raise ValueError('Wrong number of arguments for cfgUSAC')
        if args.cfgUSAC[6] > 0:
            cfgUSAC_tmp = [[i] for i in range(0,4) if i != missdig[0]]
        else:
            if args.cfgUSAC[0] < 0 or args.cfgUSAC[0] > 3:
                raise ValueError('First value for cfgUSAC out of range')
            cfgUSAC_tmp = [[args.cfgUSAC[0]]]
        cfgUSAC_tmp = appUSAC(cfgUSAC_tmp, args.cfgUSAC, 7, 'Second', 1, missdig)
        cfgUSAC_tmp = appUSAC(cfgUSAC_tmp, args.cfgUSAC, 8, 'Third', 1, missdig)
        cfgUSAC_tmp = appUSAC(cfgUSAC_tmp, args.cfgUSAC, 9, '4th', 2, missdig)
        cfgUSAC_tmp = appUSAC(cfgUSAC_tmp, args.cfgUSAC, 10, '5th', 2, missdig)
        cfgUSAC_tmp = appUSAC(cfgUSAC_tmp, args.cfgUSAC, 11, '6th', 7, missdig)
        cmds_usac = deepcopy(cmds)
        cmds = []
        for it in cmds_usac:
            for it1 in cfgUSAC_tmp:
                cmds.append(it + ['--cfgUSAC', ''.join(map(str, it1))])
    else:
        raise ValueError('Wrong number of arguments for cfgUSAC')

    if args.USACInlratFilt < 0:
        raise ValueError('Wrong number of arguments for USACInlratFilt')
    if args.USACInlratFilt < 2:
        for it in cmds:
            if args.USACInlratFilt == 0:
                if '--refineGMS' in it:
                    raise ValueError('Cannot use GMS filter within USAC if the filter was used on the input data')
                if args.USACInlratFilt == 1:
                    if '--refineVFC' in it:
                        raise ValueError('Cannot use VFC filter within USAC if the filter was used on the input data')
            it.extend(['--USACInlratFilt', str(args.USACInlratFilt)])
    else:
        cmds_usac = deepcopy(cmds)
        cmds = []
        for it in cmds_usac:
            if '--refineGMS' in it:
                raise ValueError('Cannot use GMS filter within USAC if the filter was used on the input data')
            if '--refineVFC' in it:
                raise ValueError('Cannot use VFC filter within USAC if the filter was used on the input data')
            for i in range(2):
                cmds.append(it + ['--USACInlratFilt', str(i)])

    cmds = appRange(cmds, args.th, 'th')

    if args.stereoRef:
        for it in cmds:
            it.append('--stereoRef')

    bartrefRTdis = False
    refineRT_tmp = None
    if not args.refineRT_stereo and args.stereoRef:
        if args.refineRT[0] < 0 or args.refineRT[0] > 6:
            raise ValueError('First value for refineRT out of range')
        if args.refineRT[1] < 0 or args.refineRT[1] > 2:
            raise ValueError('Second value for refineRT out of range')
        for it in cmds:
            it.append('--refineRT_stereo')
            it.append(''.join(map(str, args.refineRT[0:2])))
    elif args.refineRT_stereo:
        if len(args.refineRT_stereo) < 2 or len(args.refineRT_stereo) > 7:
            raise ValueError('Wrong number of arguments for refineRT_stereo')
        if len(args.refineRT_stereo) == 2:
            if args.refineRT_stereo[0] < 0 or args.refineRT_stereo[0] > 6:
                raise ValueError('First value for refineRT_stereo out of range')
            if args.refineRT_stereo[1] < 0 or args.refineRT_stereo[1] > 2:
                raise ValueError('Second value for refineRT_stereo out of range')
            for it in cmds:
                it.append('--refineRT_stereo')
                it.append(''.join(map(str, args.refineRT_stereo)))
        elif len(args.refineRT_stereo) > 3:
            missdig = [-1, -1]
            itdig2 = True
            if args.refineRT_stereo[-1] == -13:
                bartrefRTdis = True
            if len(args.refineRT_stereo) > 5:
                if args.refineRT_stereo[5] >= 0 and args.refineRT_stereo[5] < 3:
                    missdig[1] = args.refineRT_stereo[5]
                if args.refineRT_stereo[4] >= 0 and args.refineRT_stereo[4] < 7:
                    missdig[0] = args.refineRT_stereo[4]
            elif len(args.refineRT_stereo) > 4:
                if args.refineRT_stereo[4] >= 0 and args.refineRT_stereo[4] < 7:
                    missdig[0] = args.refineRT_stereo[4]
            if args.refineRT_stereo[2] > 0:
                refineRT_tmp = [[i] for i in range(0, 7) if i != missdig[0]]
            else:
                if args.refineRT_stereo[0] < 0 or args.refineRT_stereo[0] > 6:
                    raise ValueError('First value for refineRT_stereo out of range')
                refineRT_tmp = [[args.refineRT_stereo[0]]]
                if args.refineRT_stereo[0] == 0:
                    itdig2 = False
            if args.refineRT_stereo[3] > 0 and itdig2:
                refineRT_tmp1 = deepcopy(refineRT_tmp)
                refineRT_tmp = [[0, 0]]
                for i in range(0, 3):
                    if i != missdig[1]:
                        for j in refineRT_tmp1:
                            if j[0] != 0:
                                refineRT_tmp.append([j[0], i])
            elif not itdig2:
                refineRT_tmp[0].append(0)
            else:
                if args.refineRT_stereo[1] < 0 or args.refineRT_stereo[1] > 2:
                    raise ValueError('Second value for refineRT_stereo out of range')
                for j in refineRT_tmp:
                    j.append(args.refineRT_stereo[1])
        else:
            raise ValueError('Wrong number of arguments for refineRT_stereo')

    ref_bart = []
    if not args.BART_stereo and args.stereoRef:
        if args.BART[0] < 0 or args.BART[0] > 2:
            raise ValueError('Value for BART out of range')
        if refineRT_tmp:
            for it in refineRT_tmp:
                ref_bart.append(['--refineRT_stereo', ''.join(map(str, it)), '--BART_stereo', str(args.BART[0])])
        else:
            ref_bart = [['--BART', str(args.BART[0])]]
    elif args.BART_stereo:
        if len(args.BART_stereo) > 2:
            raise ValueError('Wrong number of arguments for BART_stereo')
        if len(args.BART_stereo) == 1:
            if args.BART_stereo[0] < 0 or args.BART_stereo[0] > 2:
                raise ValueError('Value for BART_stereo out of range')
            if args.BART_stereo[0] == 0 and bartrefRTdis:
                raise ValueError('Bundle adjustment is disabled but it was specified in refineRT_stereo that '
                                 'it is not allowed to be disabled if refineRT_stereo is disabled.')
            if refineRT_tmp:
                for it in refineRT_tmp:
                    ref_bart.append(['--refineRT_stereo',
                                     ''.join(map(str, it)), '--BART_stereo', str(args.BART_stereo[0])])
            else:
                ref_bart = [['--BART_stereo', str(args.BART_stereo[0])]]
        else:
            if not refineRT_tmp:
                ref_bart = [['--BART_stereo', str(i)] for i in range(0, 3)]
            else:
                for it in refineRT_tmp:
                    for i in range(0, 3):
                        if not bartrefRTdis or not (i == 0 and it[0] == 0 and bartrefRTdis):
                            ref_bart.append(['--refineRT_stereo', ''.join(map(str, it)), '--BART_stereo', str(i)])
    cmds_rb = deepcopy(cmds)
    cmds = []
    for it in cmds_rb:
        for it1 in ref_bart:
            cmds.append(it + it1)

    if args.minStartAggInlRat:
        cmds = appRange(cmds, args.minStartAggInlRat, 'minStartAggInlRat')

    if args.relInlRatThLast:
        cmds = appRange(cmds, args.relInlRatThLast, 'relInlRatThLast')

    if args.relInlRatThNew:
        cmds = appRange(cmds, args.relInlRatThNew, 'relInlRatThNew')

    if args.minInlierRatSkip:
        cmds = appRange(cmds, args.minInlierRatSkip, 'minInlierRatSkip')

    if args.relMinInlierRatSkip:
        cmds = appRange(cmds, args.relMinInlierRatSkip, 'relMinInlierRatSkip')

    if args.maxSkipPairs:
        cmds = appRange(cmds, args.maxSkipPairs, 'maxSkipPairs')

    if args.minInlierRatioReInit:
        cmds = appRange(cmds, args.minInlierRatioReInit, 'minInlierRatioReInit')

    if args.minPtsDistance:
        cmds = appRange(cmds, args.minPtsDistance, 'minPtsDistance')

    if args.maxPoolCorrespondences:
        cmds = appMultRanges(cmds, args.maxPoolCorrespondences, 'maxPoolCorrespondences')

    if args.minContStablePoses:
        cmds = appRange(cmds, args.minContStablePoses, 'minContStablePoses')

    if args.absThRankingStable:
        cmds = appRange(cmds, args.absThRankingStable, 'absThRankingStable')

    if args.useRANSAC_fewMatches:
        for it in cmds:
            it.append('--useRANSAC_fewMatches')

    if args.checkPoolPoseRobust:
        cmds = appRange(cmds, args.checkPoolPoseRobust, 'checkPoolPoseRobust')

    if args.minNormDistStable:
        cmds = appRange(cmds, args.minNormDistStable, 'minNormDistStable')

    if args.raiseSkipCnt[0] < 0 or args.raiseSkipCnt[0] > 9:
        raise ValueError('First value for raiseSkipCnt out of range')
    if args.raiseSkipCnt[1] < 0 or args.raiseSkipCnt[1] > 9:
        raise ValueError('Second value for raiseSkipCnt out of range')
    for it in cmds:
        it.extend(['--raiseSkipCnt', ''.join(args.raiseSkipCnt)])

    if args.maxRat3DPtsFar:
        cmds = appRange(cmds, args.maxRat3DPtsFar, 'maxRat3DPtsFar')

    if args.maxDist3DPtsZ:
        cmds = appRange(cmds, args.maxDist3DPtsZ, 'maxDist3DPtsZ')

    if args.useGTCamMat:
        for it in cmds:
            it.append('--useGTCamMat')


    return autocalib(args.path, args.executable, cpu_use, args.message_path)

def appUSAC(reslist, cfgUSAC, idx, errstr, maxv, missdig):
    if cfgUSAC[idx] > 0:
        cfgUSAC_tmp1 = deepcopy(reslist)
        reslist = []
        for i in range(maxv + 1):
            if i != missdig[idx - 6]:
                for it in cfgUSAC_tmp1:
                    reslist.append(it + [i])
    else:
        if cfgUSAC[idx - 6] < 0 or cfgUSAC[idx - 6] > maxv:
            raise ValueError(errstr + ' value for cfgUSAC out of range')
        for it in reslist:
            it.append(cfgUSAC[idx - 6])
    return reslist

def appRange(reslist, inlist, str_name):
    if len(inlist) == 1:
        for it in reslist:
            it.extend(['--' + str_name, str(inlist[0])])
    elif len(inlist) == 3:
        if (inlist[0] > inlist[1]) or \
            (inlist[2] > (inlist[1] - inlist[0])):
            raise ValueError("Parameters 1-3 (option " + str_name + ") must have the following format: "
                             "range_min range_max step_size")
        ints = False
        if not (isinstance(inlist[0], int) and isinstance(inlist[1], int) and isinstance(inlist[2], int)):
            if not float((inlist[1] - inlist[0]) / inlist[2]).is_integer():
                raise ValueError("Option " + str_name + " step size is wrong")
        else:
            ints = True
        cmds_th = deepcopy(reslist)
        reslist = []
        for it in cmds_th:
            if not ints:
                for th in np.arange(inlist[0], inlist[1] + inlist[2], inlist[2]):
                    reslist.append(it + ['--' + str_name, str(th)])
            else:
                for th in range(inlist[0], inlist[1] + inlist[2], inlist[2]):
                    reslist.append(it + ['--' + str_name, str(th)])
    else:
        raise ValueError('Wrong number of arguments for ' + str_name)
    return reslist

def appMultRanges(reslist, inlist, str_name):
    if len(inlist) == 1:
        for it in reslist:
            it.extend(['--' + str_name, str(inlist[0])])
    elif len(inlist) % 3 == 0:
        corrs_tmp = []
        ints = False
        for idx in range(0, len(inlist), 3):
            if (inlist[idx] > inlist[idx + 1]) or \
                    (inlist[idx + 2] > (inlist[idx + 1] - inlist[idx])):
                raise ValueError("Parameters 1-3 (option " + str_name + ") must have the following format: "
                                                                        "range_min range_max step_size")
            if not (isinstance(inlist[idx], int) and isinstance(inlist[idx + 1], int) and
                    isinstance(inlist[idx + 2], int)):
                if not float((inlist[idx + 1] - inlist[idx]) / inlist[idx + 2]).is_integer():
                    raise ValueError("Option " + str_name + " step size is wrong")
            else:
                ints = True
            first = inlist[idx]
            if corrs_tmp:
                if corrs_tmp[-1] == first:
                    first = inlist[idx] + inlist[idx + 2]
            if not ints:
                for th in np.arange(first, inlist[idx + 1] + inlist[idx + 2], inlist[idx + 2]):
                    corrs_tmp.append(th)
            else:
                for th in range(first, inlist[idx + 1] + inlist[idx + 2], inlist[idx + 2]):
                    corrs_tmp.append(th)
        cmds_th = deepcopy(reslist)
        reslist = []
        for it in cmds_th:
            for th in corrs_tmp:
                reslist.append(it + ['--' + str_name, str(th)])
    else:
        raise ValueError('Wrong number of arguments for ' + str_name)
    return reslist

if __name__ == "__main__":
    main()
