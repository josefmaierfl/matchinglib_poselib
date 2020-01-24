"""
Reads results from evaluation and tries to find the optimal parameters for autocalib
"""
import os, warnings
import pandas as pd
import numpy as np
from copy import deepcopy

def read_paramter_file(eval_path, main_pars):
    par_file = os.path.join(eval_path, 'resulting_best_parameters.yaml')
    if not os.path.exists(par_file):
        warnings.warn('Unable to locate file ' + par_file, UserWarning)
        return None
    from usac_eval import readYaml
    try:
        data = readYaml(par_file)
    except BaseException:
        warnings.warn('Unable to read parameter file from evaluation', UserWarning)
        return None
    for i in main_pars:
        if not any([i == b for b in data.keys()]):
            warnings.warn('Unsufficient parameters found in ' + par_file)
            return None
    return data


def get_USAC_pars56(eval_path, par_name):
    # Possible par_name: USAC_parameters_estimator, USAC_parameters_refinealg
    main_pars = ['USAC_opt_refine_ops_th', 'USAC_opt_refine_ops_inlrat', 'USAC_opt_refine_ops_inlrat_th',
                 'USAC_opt_refine_min_time']
    data = read_paramter_file(eval_path, main_pars)
    if data is None:
        return None
    res = [data[main_pars[0]]['Algorithms'][par_name],
           data[main_pars[1]]['Algorithms'][par_name],
           data[main_pars[2]]['Algorithms'][par_name],
           data[main_pars[3]][par_name]]
    if all(a == res[0] for a in res):
        return res[0]
    else:
        same = [res[3] == a for a in res[:3]]
        if not any(same):
            return None
        from itertools import compress
        same_neg = [not a for a in same]
        same.append(False)
        same_neg.append(False)
        filtered = list(compress(main_pars, same))
        if len(filtered) == 1 and filtered[0] == main_pars[0]:
            return None
        elif len(filtered) == 2 and filtered[0] == main_pars[1] and filtered[1] == main_pars[2]:
            return res[1]
        filtered_neg = list(compress(main_pars, same_neg))
        b_vals = []
        b_vals_neg = []
        for i in filtered:
            if i == 'USAC_opt_refine_ops_th' or i == 'USAC_opt_refine_ops_inlrat':
                b_vals.append(data[i]['b_best_val'])
            else:
                b_vals.append(data[i]['b_min'])
        for i in filtered_neg:
            if i == 'USAC_opt_refine_ops_th' or i == 'USAC_opt_refine_ops_inlrat':
                b_vals_neg.append(data[i]['b_best_val'])
            else:
                b_vals_neg.append(data[i]['b_min'])
        b_mean = sum(b_vals) / len(b_vals)
        b_mean_neg = sum(b_vals_neg) / len(b_vals_neg)
        diff = abs((b_mean - b_mean_neg) / max(b_mean, b_mean_neg))
        if (len(filtered) == 2 and diff < 0.05) or (len(filtered) == 1 and diff < 0.02):
            for idx, i in enumerate(main_pars):
                if i == filtered[0]:
                    return res[idx]
    return None


def get_USAC_pars123(eval_path, par_name):
    # Possible par_name: USAC_parameters_automaticSprtInit, USAC_parameters_automaticProsacParameters,
    # USAC_parameters_prevalidateSample, USAC_parameters_USACInlratFilt
    main_pars = ['USAC_opt_search_ops_th', 'USAC_opt_search_ops_inlrat', 'USAC_opt_search_ops_kpAccSd_th',
                 'USAC_opt_search_ops_inlrat_th', 'USAC_opt_search_min_time', 'USAC_opt_search_min_time_inlrat_th',
                 'USAC_opt_search_min_time_kpAccSd_inlrat_th', 'USAC_opt_search_min_inlrat_diff']
    data = read_paramter_file(eval_path, main_pars)
    if data is None:
        return None
    res = [data[main_pars[0]]['Algorithms'][par_name],
           data[main_pars[1]]['Algorithms'][par_name],
           data[main_pars[2]]['Algorithms'][par_name],
           data[main_pars[3]]['Algorithms'][par_name],
           data[main_pars[4]][par_name],
           data[main_pars[5]]['Algorithm'][par_name],
           data[main_pars[6]]['res1']['Algorithm'][par_name],
           data[main_pars[6]]['res2']['Algorithm'][par_name],
           data[main_pars[7]]['Algorithm'][par_name]]
    if all(a == res[0] for a in res):
        return res[0]
    else:
        index = main_pars[:7] + main_pars[6:]
        index2 = [np.NaN] * 6 + ['res1', 'res2', np.NaN]
        fieldType = ['err'] * 4 + ['time'] * 4 + ['diff']
        err_val = [data[main_pars[0]]['b_best_val'],
                   data[main_pars[1]]['b_best_val'],
                   data[main_pars[2]]['b_min'],
                   data[main_pars[3]]['b_min']] + [np.NaN] * 5
        time = [np.NaN] * 5 + [data[main_pars[5]]['Time_us'],
                               data[main_pars[6]]['res1']['Time_us'],
                               data[main_pars[6]]['res2']['Time_us']] + [np.NaN]
        weight = [0.4] + [1.0] * 5 + [0.3] * 2 + [0.5]
        idiff = [np.NaN] * 8 + [data[main_pars[7]]['inlRatDiff']]
        d1 = {'res_name': index, 'index2': index2, 'fieldType': fieldType,
              'err_val': err_val, 'time': time, 'idiff': idiff, 'alg_name': res, 'weight': weight}
        df = pd.DataFrame(data=d1)
        tc = df.loc[df['fieldType'] == 'time']['alg_name'].value_counts()
        tc.name = tc.name + '_cnt'
        tc.index.name = 'alg_name'
        tc_df = tc.reset_index()
        tc2 = tc.loc[tc.gt(1)]
        if tc2.empty:
            return None
        bc = df.loc[df['fieldType'] == 'err']['alg_name'].value_counts()
        bc.name = bc.name + '_cnt'
        bc.index.name = 'alg_name'
        bc2 = bc.loc[bc.gt(1)]
        if bc2.empty:
            return None
        same = tc_df['alg_name'].isin(bc.index.values.tolist())
        df2 = df.loc[df['alg_name'].isin(tc_df.loc[same, 'alg_name'].tolist())].groupby(['fieldType', 'alg_name'])
        df_sum = df2['weight'].sum().reset_index()
        df_sum = df_sum.loc[((df_sum['weight'] > 1) & (df_sum['fieldType'] == 'time')) |
                            ((df_sum['weight'] > 1) & (df_sum['fieldType'] == 'err')) | (df_sum['fieldType'] == 'diff')]
        if df_sum.empty:
            return None
        if df_sum.shape[0] == 1:
            return str(df_sum.loc[:, 'alg_name'].iloc[0])
        df_sum2 = df_sum.groupby(['alg_name'])['weight'].sum().sort_values(ascending=False)
        if df_sum2.iloc[0] == df_sum2.iloc[1]:
            df_sum2 = df_sum2.loc[(df_sum2 == df_sum2.iloc[0])]
            df3 = df.loc[df['alg_name'].isin(df_sum2.index.values.tolist())]
            df3 = df3.loc[(df3['fieldType'] == 'err')]
            df3_sum = df3.groupby('alg_name').apply(lambda x: np.average(x.err_val, weights=x.weight)).sort_values()
            return str(df3_sum.index.values.tolist()[0])
        else:
            return str(df_sum2.index.values.tolist()[0])


def get_th(eval_path1, eval_path2):
    main_pars1 = ['USAC_opt_refine_ops_th', 'USAC_opt_refine_ops_inlrat_th']
    main_pars2 = ['USAC_opt_search_ops_th', 'USAC_opt_search_ops_kpAccSd_th',
                  'USAC_opt_search_ops_inlrat_th', 'USAC_opt_search_min_time_inlrat_th',
                  'USAC_opt_search_min_time_kpAccSd_inlrat_th', 'USAC_opt_search_min_inlrat_diff']
    data1 = read_paramter_file(eval_path1, main_pars1)
    if data1 is None:
        return None
    data2 = read_paramter_file(eval_path2, main_pars2)
    if data2 is None:
        return None
    # Results from smallest errors
    res1 = [data1[main_pars1[0]]['th_best'],
            data1[main_pars1[0]]['th_best_mean'],
            data1[main_pars1[1]]['th'],
            data2[main_pars2[0]]['th_best'],
            data2[main_pars2[0]]['th_best_mean'],
            data2[main_pars2[1]]['th'],
            data2[main_pars2[2]]['th'],
            data2[main_pars2[5]]['th']]
    res1 = [float(a) for a in res1]
    # Results from best timing
    res2 = [data2[main_pars2[3]]['th'],
            data2[main_pars2[4]]['res1']['th'],
            data2[main_pars2[4]]['res2']['th']]
    res2 = [float(a) for a in res2]
    s_err = pd.Series(res1)
    ses = s_err.describe()
    if ses['max'] - ses['min'] > 0.5:
        return None
    s_t = pd.Series(res2)
    sts = s_t.describe()
    if sts[r'50%'] < ses['min'] or sts[r'50%'] > ses['max']:
        return None
    if ses['std'] > 0.2:
        s_err1 = s_err.loc[((s_err >= ses[r'25%']) & (s_err <= ses[r'75%']))]
        ses1 = s_err1.describe()
        if ses1['std'] > 0.15:
            return None
        s_err1 = s_err1.append(pd.Series(sts[r'50%']))
        ses1 = s_err1.describe()
        return float(ses1['mean'])
    s_err = s_err.append(pd.Series(sts[r'50%']))
    ses = s_err.describe()
    return float(ses['mean'])


def get_robMFilt(eval_path, par_name):
    # par_name should be RobMethod
    main_pars = ['usac_vs_autoc_stabRT_inlrat', 'usac_vs_autoc_stabRT_depth']
    data = read_paramter_file(eval_path, main_pars)
    if data is None:
        return None
    res = [data[main_pars[0]]['Algorithms'][par_name],
           data[main_pars[1]]['Algorithms'][par_name]]
    if all(a == res[0] for a in res):
        if res[0] != 'USAC':
            warnings.warn('RANSAC is found to be better in terms of accuracy compared to USAC.')
        return res[0]
    else:
        b0 = data[main_pars[0]]['b_best_val']
        b1 = data[main_pars[1]]['b_best_val']
        diff = abs((b0 - b1) / max(b0, b1))
        if diff > 0.1:
            return None
        if np.isclose(b0, b1):
            return 'USAC'
        if b0 > b1:
            return res[1]
        else:
            return res[0]


def get_refinement_ba(eval_path1, eval_path2, par_name):
    # Possible par_name: refineMethod_algorithm, refineMethod_costFunction, BART
    main_pars1 = ['refineRT_BA_opts_inlrat', 'refinement_ba_best_comb_scenes', 'refineRT_BA_min_time',
                  'refineRT_BA_opts_kpAccSd']
    main_pars2 = ['refineRT_opts_for_BA2_inlrat', 'refinement_best_comb_for_BA2_scenes',
                  'refineRT_opts_for_BA2_K_inlrat', 'refinement_best_comb_for_BA2_K_scenes']
    data1 = read_paramter_file(eval_path1, main_pars1)
    if data1 is None:
        return None
    res = [data1[a]['Algorithms'][par_name] for a in main_pars1]
    fieldType = ['err'] * 2 + ['time'] + ['err']
    err_val = [data1[main_pars1[0]]['b_best_val'],
               data1[main_pars1[1]]['b_min']] + [np.NaN] + [data1[main_pars1[3]]['b_best_val']]
    time = [np.NaN] * 2 + [data1[main_pars1[2]]['t_min']] + [np.NaN]
    kerr = [np.NaN] * 4
    weight = [1.0] * 2 + [0.75, 1.0]
    index = deepcopy(main_pars1)
    minw = 2.0
    if par_name != 'BART':
        data2 = read_paramter_file(eval_path2, main_pars2)
        if data2 is None:
            return None
        res += [data2[a]['Algorithms'][par_name] for a in main_pars2]
        fieldType += ['err'] * 2 + ['kerr'] * 2
        err_val += [data2[main_pars2[0]]['b_best_val'],
                    data2[main_pars2[1]]['b_min']] + [np.NaN] * 2
        time += [np.NaN] * 4
        kerr += [np.NaN] * 2 + [data2[main_pars2[2]]['ke_best_val'], data2[main_pars2[3]]['ke_min']]
        weight += [0.7] * 2 + [0.4] * 2
        index += main_pars2
        minw = 2.5
    if all(a == res[0] for a in res):
        return res[0]
    else:
        d1 = {'res_name': index, 'fieldType': fieldType,
              'err_val': err_val, 'time': time, 'kerr': kerr, 'alg_name': res, 'weight': weight}
        df = pd.DataFrame(data=d1)
        df_err = df.loc[df['fieldType'] == 'err'].copy(deep=True)
        addw = df_err['alg_name'].isin(df.loc[(df['fieldType'] == 'time'), 'alg_name'].tolist())
        if not addw.any():
            minw *= 1.15
        else:
            df_err.loc[addw, 'weight'] += 0.3
        if par_name != 'BART':
            addw = df_err['alg_name'].isin(df.loc[(df['fieldType'] == 'kerr'), 'alg_name'].tolist())
            if not addw.any():
                minw *= 1.1
            else:
                bc = df.loc[(df['fieldType'] == 'kerr'), 'alg_name'].value_counts()
                df_err.loc[addw, 'weight'] += 0.25 * float(bc.iloc[0])
        dfe_sum = df_err.groupby(['alg_name'])['weight'].sum().sort_values(ascending=False)
        if dfe_sum.iloc[0] < minw:
            return None
        if dfe_sum.shape[0] == 1:
            return str(dfe_sum.index.values.tolist()[0])
        if dfe_sum.iloc[0] == dfe_sum.iloc[1]:
            df_sum2 = dfe_sum.loc[(dfe_sum == dfe_sum.iloc[0])]
            df3 = df_err.loc[df_err['alg_name'].isin(df_sum2.index.values.tolist())]
            df3_sum = df3.groupby('alg_name').apply(lambda x: np.average(x.err_val, weights=x.weight)).sort_values()
            return str(df3_sum.index.values.tolist()[0])
        else:
            return str(dfe_sum.index.values.tolist()[0])


def get_refinement_ba_stereo(eval_path1, eval_path2, par_name):
    # Possible par_name: stereoParameters_refineMethod_CorrPool_algorithm,
    # stereoParameters_refineMethod_CorrPool_costFunction, stereoParameters_BART_CorrPool
    main_pars1 = ['refRT_stereo_BA_opts_inlrat', 'ref_stereo_ba_best_comb_scenes', 'refRT_BA_stereo_min_time']
    main_pars2 = ['refRT_stereo_opts_for_BA2_inlrat', 'ref_stereo_best_comb_for_BA2_scenes',
                  'refRT_stereo_opts_for_BA2_K_inlrat', 'ref_stereo_best_comb_for_BA2_K_scenes']
    data1 = read_paramter_file(eval_path1, main_pars1)
    if data1 is None:
        return None
    res = [str(data1[main_pars1[0]]['Algorithms'][par_name]),
           str(data1[main_pars1[1]]['Algorithms'][par_name]),
           str(data1[main_pars1[2]]['Algorithm'][par_name])]
    fieldType = ['err'] * 2 + ['time']
    err_val = [data1[main_pars1[0]]['b_best_val'],
               data1[main_pars1[1]]['b_min']] + [np.NaN]
    time = [np.NaN] * 2 + [data1[main_pars1[2]]['min_mean_time']]
    kerr = [np.NaN] * 3
    weight = [1.0] * 2 + [0.75]
    index = deepcopy(main_pars1)
    minw = 1.39
    if par_name != 'stereoParameters_BART_CorrPool':
        data2 = read_paramter_file(eval_path2, main_pars2)
        if data2 is None:
            return None
        res += [str(data2[a]['Algorithms'][par_name]) for a in main_pars2]
        fieldType += ['err'] * 2 + ['kerr'] * 2
        err_val += [data2[main_pars2[0]]['b_best_val'],
                    data2[main_pars2[1]]['b_min']] + [np.NaN] * 2
        time += [np.NaN] * 4
        kerr += [np.NaN] * 2 + [data2[main_pars2[2]]['ke_best_val'], data2[main_pars2[3]]['ke_min']]
        weight += [0.7] * 2 + [0.4] * 2
        index += main_pars2
        minw = 2.5
    if all(a == res[0] for a in res):
        return res[0]
    else:
        d1 = {'res_name': index, 'fieldType': fieldType,
              'err_val': err_val, 'time': time, 'kerr': kerr, 'alg_name': res, 'weight': weight}
        df = pd.DataFrame(data=d1)
        df_err = df.loc[df['fieldType'] == 'err'].copy(deep=True)
        addw = df_err['alg_name'].isin(df.loc[(df['fieldType'] == 'time'), 'alg_name'].tolist())
        if not addw.any():
            minw *= 1.15
        else:
            df_err.loc[addw, 'weight'] += 0.4
        if par_name != 'stereoParameters_BART_CorrPool':
            addw = df_err['alg_name'].isin(df.loc[(df['fieldType'] == 'kerr'), 'alg_name'].tolist())
            if not addw.any():
                minw *= 1.15
            else:
                bc = df.loc[(df['fieldType'] == 'kerr'), 'alg_name'].value_counts()
                df_err.loc[addw, 'weight'] += 0.25 * float(bc.iloc[0])
        dfe_sum = df_err.groupby(['alg_name'])['weight'].sum().sort_values(ascending=False)
        if dfe_sum.iloc[0] < minw:
            return None
        if dfe_sum.shape[0] == 1:
            return str(dfe_sum.index.values.tolist()[0])
        if dfe_sum.iloc[0] == dfe_sum.iloc[1]:
            df_sum2 = dfe_sum.loc[(dfe_sum == dfe_sum.iloc[0])]
            df3 = df_err.loc[df_err['alg_name'].isin(df_sum2.index.values.tolist())]
            df3_sum = df3.groupby('alg_name').apply(lambda x: np.average(x.err_val, weights=x.weight)).sort_values()
            return str(df3_sum.index.values.tolist()[0])
        else:
            return str(dfe_sum.index.values.tolist()[0])


def get_corrpool_size(eval_path):
    # Possible par_name: stereoParameters_maxPoolCorrespondences
    main_pars = ['corrpool_size_pts_dist_inlrat', 'corrpool_size_pts_dist_best_comb_scenes',
                 'corrpool_size_pts_dist_end_frames_best_comb_scenes', 'corrpool_size_converge',
                 'corrpool_size_converge_mean']
    data = read_paramter_file(eval_path, main_pars)
    if data is None:
        return None
    res = [float(data[main_pars[0]]['Algorithms']['stereoParameters_maxPoolCorrespondences']),
           float(data[main_pars[1]]['Algorithms']['stereoParameters_maxPoolCorrespondences']),
           float(data[main_pars[2]]['Algorithms']['stereoParameters_maxPoolCorrespondences'])]
    res1 = [float(data[main_pars[3]]['mean_conv_pool_size']),
            float(data[main_pars[4]]['mean_conv_pool_size'])]
    res_dist = [float(data[main_pars[3]]['Algorithm']['stereoParameters_minPtsDistance']),
                float(data[main_pars[4]]['Algorithm']['stereoParameters_minPtsDistance'])]
    b_err = [data[main_pars[0]]['b_best_val'],
             data[main_pars[1]]['b_min'],
             data[main_pars[2]]['b_min']]
    err1 = [abs(data[main_pars[3]]['mean_R_error']) + abs(data[main_pars[3]]['mean_t_error']),
            abs(data[main_pars[4]]['mean_R_error']) + abs(data[main_pars[4]]['mean_t_error'])]
    mi = min(res)
    ma = max(res)
    if np.isclose(mi, ma):
        m = mi
    else:
        m = mi + (mi + ma) / 2
    mm = [min(0.75 * m, mi), max(1.2 * m, ma)]
    in_range = []
    for idx, i in enumerate(res1):
        if (i > mm[0]) and (i < mm[1]) and i < 25000.0:
            in_range.append(idx)
    if not in_range:
        return None
    elif len(in_range) == 1:
        m = res1[in_range[0]]
    else:
        if any((a / max(res1)) < 0.75 for a in res1):
            return None
        if np.isclose(res_dist[0], res_dist[1]) or np.isclose(err1[0], err1[1]):
            m = sum(res1) / len(res1)
        else:
            w = [1.3 - a / max(err1) for a in err1]
            m = np.average(np.array(res1), weights=np.array(w))
    d = [abs(a - m) for a in res]
    w = [(1.5 - a / max(d)) for a in d]
    w1 = [1.5 - a / max(b_err) for a in b_err]
    w = [a * b for a, b in zip(w, w1)]
    w = [0.1 * a / max(w) for a in w]
    return int(np.average(np.array([m] + res), weights=np.array([1.0] + w)))


def get_min_pt_dist(eval_path, maxPoolCorrs):
    if maxPoolCorrs is None:
        return None
    # Possible par_name: stereoParameters_minPtsDistance
    main_pars = ['corrpool_size_pts_dist_inlrat', 'corrpool_size_pts_dist_best_comb_scenes',
                 'corrpool_size_pts_dist_end_frames_best_comb_scenes', 'corrpool_size_converge',
                 'corrpool_size_converge_mean']
    data = read_paramter_file(eval_path, main_pars)
    if data is None:
        return None
    res = [float(data[main_pars[0]]['Algorithms']['stereoParameters_minPtsDistance']),
           float(data[main_pars[1]]['Algorithms']['stereoParameters_minPtsDistance']),
           float(data[main_pars[2]]['Algorithms']['stereoParameters_minPtsDistance']),
           float(data[main_pars[3]]['Algorithm']['stereoParameters_minPtsDistance']),
           float(data[main_pars[4]]['Algorithm']['stereoParameters_minPtsDistance'])]
    res1 = [float(data[main_pars[0]]['Algorithms']['stereoParameters_maxPoolCorrespondences']),
            float(data[main_pars[1]]['Algorithms']['stereoParameters_maxPoolCorrespondences']),
            float(data[main_pars[2]]['Algorithms']['stereoParameters_maxPoolCorrespondences']),
            float(data[main_pars[3]]['mean_conv_pool_size']),
            float(data[main_pars[4]]['mean_conv_pool_size'])]
    b_err = [data[main_pars[0]]['b_best_val'],
             data[main_pars[1]]['b_min'],
             data[main_pars[2]]['b_min']]
    if np.allclose(np.array(res), res[0]):
        return res[0]
    s_d = pd.Series(res)
    sds = s_d.describe()
    if sds['max'] - sds['min'] > 8:
        return None
    if sds['std'] > 4:
        return None
    w = [(1.5 - a / max(b_err)) for a in b_err]
    w = [a / max(w) for a in w] + [1.0, 1.0]
    d = [abs(a - maxPoolCorrs) for a in res1]
    w1 = [(1.3 - a / max(d)) for a in d]
    w1 = [0.33 * a / max(w1) for a in w1]
    w = [a * b for a, b in zip(w, w1)]
    return round(float(np.average(np.array(res), weights=np.array(w))), 3)


def get_corrpool_1(eval_path):
    pool_size = get_corrpool_size(eval_path)
    if pool_size is None:
        return None
    if pool_size > 10000:
        warnings.warn('Estimated a really large optimal correspondence pool size of ' + str(pool_size), UserWarning)
    pts_dist = get_min_pt_dist(eval_path, pool_size)
    if pts_dist is None:
        return None
    return {'stereoParameters_maxPoolCorrespondences': pool_size, 'stereoParameters_minPtsDistance': pts_dist}


def check_usac56_comb_exists(eval_path):
    main_pars = ['USAC_opt_refine_ops_th', 'USAC_opt_refine_ops_inlrat', 'USAC_opt_refine_ops_inlrat_th',
                 'USAC_opt_refine_min_time']
    data = read_paramter_file(eval_path, main_pars)
    if data is None:
        return None
    res = [data[main_pars[0]]['Algorithms'],
           data[main_pars[1]]['Algorithms'],
           data[main_pars[2]]['Algorithms'],
           data[main_pars[3]]]
    return res


def check_refinement_ba_comb_exists(eval_path1):
    main_pars1 = ['refineRT_BA_opts_inlrat', 'refinement_ba_best_comb_scenes', 'refineRT_BA_min_time',
                  'refineRT_BA_opts_kpAccSd']
    data1 = read_paramter_file(eval_path1, main_pars1)
    if data1 is None:
        return None
    res = [data1[a]['Algorithms'] for a in main_pars1]
    return res


def check_refinement_ba_stereo_comb_exists(eval_path1, eval_path2):
    main_pars1 = ['refRT_stereo_BA_opts_inlrat', 'ref_stereo_ba_best_comb_scenes', 'refRT_BA_stereo_min_time']
    main_pars2 = ['refRT_stereo_opts_for_BA2_inlrat', 'ref_stereo_best_comb_for_BA2_scenes',
                  'refRT_stereo_opts_for_BA2_K_inlrat', 'ref_stereo_best_comb_for_BA2_K_scenes']
    data1 = read_paramter_file(eval_path1, main_pars1)
    if data1 is None:
        return None
    res = [data1[main_pars1[0]]['Algorithms'],
           data1[main_pars1[1]]['Algorithms'],
           data1[main_pars1[2]]['Algorithm']]
    for i in res:
        for j in i.keys():
            i[j] = str(i[j])
    data2 = read_paramter_file(eval_path2, main_pars2)
    if data2 is None:
        return None
    res += [data2[a]['Algorithms'] for a in main_pars2]
    return res


def check_usac123_comb_exists(eval_path):
    main_pars = ['USAC_opt_search_ops_th', 'USAC_opt_search_ops_inlrat', 'USAC_opt_search_ops_kpAccSd_th',
                 'USAC_opt_search_ops_inlrat_th', 'USAC_opt_search_min_time', 'USAC_opt_search_min_time_inlrat_th',
                 'USAC_opt_search_min_time_kpAccSd_inlrat_th', 'USAC_opt_search_min_inlrat_diff']
    data = read_paramter_file(eval_path, main_pars)
    if data is None:
        return None
    res = [data[main_pars[0]]['Algorithms'],
           data[main_pars[1]]['Algorithms'],
           data[main_pars[2]]['Algorithms'],
           data[main_pars[3]]['Algorithms'],
           data[main_pars[4]],
           data[main_pars[5]]['Algorithm'],
           data[main_pars[6]]['res1']['Algorithm'],
           data[main_pars[6]]['res2']['Algorithm'],
           data[main_pars[7]]['Algorithm']]
    return res


def check_comb_exists(eval_path, par_names, func_name, eval_path2=None, skip_par_name=None):
    if not isinstance(par_names, dict):
        raise ValueError('Parameter names musts be in dict format')
    if skip_par_name is not None and not isinstance(skip_par_name, list):
        raise ValueError('Skip parameter names musts be in list format')
    par_names1 = deepcopy(par_names)
    if skip_par_name is not None:
        for key in skip_par_name:
            par_names1.pop(key, None)
    if len(par_names1.keys()) == 1:
        return True
    if eval_path2 is not None:
        combs = func_name(eval_path, eval_path2)
    else:
        combs = func_name(eval_path)
    for i in par_names1.keys():
        for j in combs:
            if not any(a == i for a in j.keys()):
                return False #raise ValueError('Parameter names read from file do not match given names.')
    for i in combs:
        if all(i[a] == par_names1[a] for a in par_names1.keys()):
            return True
    return False


def main():
    path = '/home/maierj/work/Sequence_Test/py_test/correspondence_pool/1'
    path2 = '/home/maierj/work/Sequence_Test/py_test/refinement_ba_stereo/2'
    pars = ['USAC_parameters_estimator', 'USAC_parameters_refinealg', 'USAC_parameters_USACInlratFilt']
    # skip_cons_par = ['USAC_parameters_USACInlratFilt']
    # rets = dict.fromkeys(pars)
    # for i in pars:
    #     rets[i] = get_refinement_ba_stereo(path, path2, i)
    # ret = check_comb_exists(path, rets, check_refinement_ba_stereo_comb_exists, path2, skip_cons_par)
    # th = get_th(path, path2)
    # path = '/home/maierj/work/Sequence_Test/py_test/usac_vs_autocalib/1'
    # ret = get_robMFilt(path, 'stereoRef')
    ret = get_corrpool_1(path)


if __name__ == '__main__':
    main()