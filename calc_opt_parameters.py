"""
Reads results from evaluation and tries to find the optimal parameters for autocalib
"""
import os, warnings
import pandas as pd
import numpy as np


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


def check_comb_exists(eval_path, par_names, func_name):
    if not isinstance(par_names, dict):
        raise ValueError('Parameter names musts be in dict format')
    if len(par_names.keys()) == 1:
        return True
    combs = func_name(eval_path)
    for i in par_names.keys():
        for j in combs:
            if not any(a == i for a in j.keys()):
                raise ValueError('Parameter names read from file do not match given names.')
    for i in combs:
        if all(i[a] == par_names[a] for a in par_names.keys()):
            return True
    return False


def main():
    path = '/home/maierj/work/Sequence_Test/py_test/usac-testing/1'
    ret = get_USAC_pars56(path, 'USAC_parameters_refinealg')

if __name__ == '__main__':
    main()