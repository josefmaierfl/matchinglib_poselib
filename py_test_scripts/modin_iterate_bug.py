import numpy as np
import modin.pandas as pd
# import pandas as pd

tdata = '/home/maierj/work/Sequence_Test/py_test/test1/17369781811722364928.txt'
# num_pts = int(5000)
# addInfo = ['a', 'b', 'c', '12', '18', 'aa', 'a123', 'qw34rf']
# addInfoKeys = ['k1', 'k2', 'k3']
# create_info = []
# for i in range(0, num_pts):
#     create_info += ['_'.join([addInfoKeys[j] +
#                               '_' +
#                               addInfo[a] for j, a in enumerate(np.random.randint(0, len(addInfo), len(addInfoKeys)))])]
# data = {'Nr': list(range(0, num_pts)),
#         'err1_1': [0.3, 0.5, 0.7, 0.4, 0.6] * int(num_pts/5),
#         'err2': 1000 + np.abs(np.random.randn(num_pts) * 10),
#         'err3': 10 + np.random.randn(num_pts) * 5,
#         'err4': -1000 + np.abs(np.random.randn(num_pts)),
#         'err5': [0.3, 0.5, 0.7, 0.4, 0.6] * int(num_pts/5),
#         'err6': np.abs(np.random.randn(num_pts) * 100),
#         'err7': -10000 + np.random.randn(num_pts) * 100,
#         'err8': 20000 + np.random.randn(num_pts),
#         'err9': -450 + np.random.randn(num_pts),
#         'par1': np.random.randint(0, 3, num_pts),
#         'par2': np.random.randint(0, 7, num_pts),
#         'par3': np.random.randint(8, 10, num_pts),
#         'th': np.tile(np.arange(0.4, 0.9, 0.1), int(num_pts/5)),
#         'th2': np.tile(np.arange(0.05, 0.45, 0.1), int(num_pts/4)),
#         'useless': [1, 1, 2, 3] * int(num_pts/4),
#         'Filter_a(0,0)': [0] * 10 + [1] * int(num_pts - 10),
#         'Filter_a(0,1)': [0] * 10 + [0] * int(num_pts - 10),
#         'Filter_a(0,2)': [float(0)] * 10 + [0.1] * int(num_pts - 10),
#         'Filter_a(1,0)': [0] * 10 + [1] * int(num_pts - 10),
#         'Filter_a(1,1)': [0] * 10 + [1] * int(num_pts - 10),
#         'Filter_a(1,2)': [0] * 10 + [0] * int(num_pts - 10),
#         'Filter_a(2,0)': [float(0)] * 10 + [0.1] * int(num_pts - 10),
#         'Filter_a(2,1)': [0] * 10 + [0] * int(num_pts - 10),
#         'Filter_a(2,2)': [0] * 10 + [0] * int(num_pts - 10),
#         'addInfo': create_info}
# data = pd.DataFrame(data)
# data.to_csv(index=False, sep=';', path_or_buf=tdata, header=True)
# data_types = {'Nr': np.int64,
#               'R_diffAll': np.float64, 'R_diff_roll_deg': np.float64, 'R_diff_pitch_deg': np.float64,
#               'R_diff_yaw_deg': np.float64, 't_angDiff_deg': np.float64, 't_distDiff': np.float64,
#               't_diff_tx': np.float64, 't_diff_ty': np.float64, 't_diff_tz': np.float64,
#               'R_mostLikely_diffAll': np.float64, 'R_mostLikely_diff_roll_deg': np.float64,
#               'R_mostLikely_diff_pitch_deg': np.float64, 'R_mostLikely_diff_yaw_deg': np.float64,
#               't_mostLikely_angDiff_deg': np.float64, 't_mostLikely_distDiff': np.float64,
#               't_mostLikely_diff_tx': np.float64, 't_mostLikely_diff_ty': np.float64,
#               't_mostLikely_diff_tz': np.float64, 'R_out(0,0)': np.float64, 'R_out(0,1)': np.float64,
#               'R_out(0,2)': np.float64, 'R_out(1,0)': np.float64, 'R_out(1,1)': np.float64, 'R_out(1,2)': np.float64,
#               'R_out(2,0)': np.float64, 'R_out(2,1)': np.float64, 'R_out(2,2)': np.float64, 't_out(0,0)': np.float64,
#               't_out(1,0)': np.float64, 't_out(2,0)': np.float64, 'R_mostLikely(0,0)': np.float64,
#               'R_mostLikely(0,1)': np.float64, 'R_mostLikely(0,2)': np.float64, 'R_mostLikely(1,0)': np.float64,
#               'R_mostLikely(1,1)': np.float64, 'R_mostLikely(1,2)': np.float64, 'R_mostLikely(2,0)': np.float64,
#               'R_mostLikely(2,1)': np.float64, 'R_mostLikely(2,2)': np.float64, 't_mostLikely(0,0)': np.float64,
#               't_mostLikely(1,0)': np.float64, 't_mostLikely(2,0)': np.float64, 'R_GT(0,0)': np.float64,
#               'R_GT(0,1)': np.float64, 'R_GT(0,2)': np.float64, 'R_GT(1,0)': np.float64, 'R_GT(1,1)': np.float64,
#               'R_GT(1,2)': np.float64, 'R_GT(2,0)': np.float64, 'R_GT(2,1)': np.float64, 'R_GT(2,2)': np.float64,
#               't_GT(0,0)': np.float64, 't_GT(1,0)': np.float64, 't_GT(2,0)': np.float64, 'poseIsStable': np.float64,
#               'mostLikelyPose_stable': np.float64, 'K1_fxDiff': np.float64, 'K1_fyDiff': np.float64,
#               'K1_fxyDiffNorm': np.float64, 'K1_cxDiff': np.float64, 'K1_cyDiff': np.float64,
#               'K1_cxyDiffNorm': np.float64, 'K1_cxyfxfyNorm': np.float64, 'K2_fxDiff': np.float64,
#               'K2_fyDiff': np.float64, 'K2_fxyDiffNorm': np.float64, 'K2_cxDiff': np.float64, 'K2_cyDiff': np.float64,
#               'K2_cxyDiffNorm': np.float64, 'K2_cxyfxfyNorm': np.float64, 'K1(0,0)': np.float64, 'K1(0,1)': np.float64,
#               'K1(0,2)': np.float64, 'K1(1,0)': np.float64, 'K1(1,1)': np.float64, 'K1(1,2)': np.float64,
#               'K1(2,0)': np.float64, 'K1(2,1)': np.float64, 'K1(2,2)': np.float64, 'K2(0,0)': np.float64,
#               'K2(0,1)': np.float64, 'K2(0,2)': np.float64, 'K2(1,0)': np.float64, 'K2(1,1)': np.float64,
#               'K2(1,2)': np.float64, 'K2(2,0)': np.float64, 'K2(2,1)': np.float64, 'K2(2,2)': np.float64,
#               'K1_GT(0,0)': np.float64, 'K1_GT(0,1)': np.float64, 'K1_GT(0,2)': np.float64, 'K1_GT(1,0)': np.float64,
#               'K1_GT(1,1)': np.float64, 'K1_GT(1,2)': np.float64, 'K1_GT(2,0)': np.float64, 'K1_GT(2,1)': np.float64,
#               'K1_GT(2,2)': np.float64, 'K2_GT(0,0)': np.float64, 'K2_GT(0,1)': np.float64, 'K2_GT(0,2)': np.float64,
#               'K2_GT(1,0)': np.float64, 'K2_GT(1,1)': np.float64, 'K2_GT(1,2)': np.float64, 'K2_GT(2,0)': np.float64,
#               'K2_GT(2,1)': np.float64, 'K2_GT(2,2)': np.float64, 'K1_degenerate(0,0)': np.float64,
#               'K1_degenerate(0,1)': np.float64, 'K1_degenerate(0,2)': np.float64, 'K1_degenerate(1,0)': np.float64,
#               'K1_degenerate(1,1)': np.float64, 'K1_degenerate(1,2)': np.float64, 'K1_degenerate(2,0)': np.float64,
#               'K1_degenerate(2,1)': np.float64, 'K1_degenerate(2,2)': np.float64, 'K2_degenerate(0,0)': np.float64,
#               'K2_degenerate(0,1)': np.float64, 'K2_degenerate(0,2)': np.float64, 'K2_degenerate(1,0)': np.float64,
#               'K2_degenerate(1,1)': np.float64, 'K2_degenerate(1,2)': np.float64, 'K2_degenerate(2,0)': np.float64,
#               'K2_degenerate(2,1)': np.float64, 'K2_degenerate(2,2)': np.float64, 'inlRat_estimated': np.float64,
#               'inlRat_GT': np.float64, 'nrCorrs_filtered': np.int64, 'nrCorrs_estimated': np.int64,
#               'nrCorrs_GT': np.int64, 'filtering_us': np.int64, 'robEstimationAndRef_us': np.int64,
#               'linRefinement_us': np.int64, 'bundleAdjust_us': np.int64, 'stereoRefine_us': np.int64,
#               'addSequInfo': str}

csv_data = pd.read_csv(tdata, delimiter=';')#, dtype=data_types)
addSequInfo_sep = None
for row in csv_data.itertuples():
    tmp = row.addInfo.split('_')
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
csv_data.drop(columns=['addInfo'], inplace=True)
# data_set_tmp = merge_dicts(data_set)
# data_set_tmp = pd.DataFrame(data=data_set_tmp, index=[0])
# data_set_repl = pd.DataFrame(np.repeat(data_set_tmp.values, csv_data.shape[0], axis=0))
# data_set_repl.columns = data_set_tmp.columns
# csv_new = pd.concat([csv_data, data_set_repl], axis=1, sort=False, join_axes=[csv_data.index])