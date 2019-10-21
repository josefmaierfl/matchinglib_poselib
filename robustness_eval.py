"""
Calculates the best performing parameter values with given testing data as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np, math
# import modin.pandas as pd
import pandas as pd
#from jinja2 import Template as ji
import jinja2 as ji
import ruamel.yaml as yaml
from usac_eval import ji_env, get_time_fixed_kp, insert_opt_lbreak, prepare_io

def get_rt_change_type(**keywords):
    if 'data_seperators' not in keywords:
        raise ValueError('it_parameters missing!')

    df_grp = keywords['data'].groupby('data_seperators')
    grp_keys = df_grp.groups.keys()
    df_list = []
    for grp in grp_keys:
        tmp = df_grp.get_group(grp)
        nr_min = tmp['Nr'].min()
        nr_max = tmp['Nr'].max()
        rng1 = nr_max - nr_min + 1
        tmp1 = tmp['Nr'].loc[(tmp['Nr'] == nr_min)]
        tmp1_it = tmp1.iteritems()
        idx_prev, _ = next(tmp1_it)
        indexes = {'first': [], 'last': []}
        for idx, _ in tmp1_it:
            diff = idx - idx_prev
            indexes['first'].append(idx_prev)
            if diff != rng1:
                tmp2 = tmp['Nr'].iloc[((tmp.index > idx_prev) & (tmp.index < idx))]
                tmp2_it = tmp2.iteritems()
                idx1_prev, nr_prev = next(tmp2_it)
                for idx1, nr in tmp2_it:
                    if nr_prev > nr:
                        indexes['last'].append(idx1_prev + 1)
                        indexes['first'].append(idx1)
                    idx1_prev = idx1
            indexes['last'].append(idx)
            idx_prev = idx
        indexes['first'].append(tmp1.index[-1])
        diff = tmp.index[-1] - tmp1.index[-1] + 1
        if diff != rng1:
            tmp2 = tmp['Nr'].iloc[(tmp.index > tmp1.index[-1])]
            tmp2_it = tmp2.iteritems()
            idx1_prev, nr_prev = next(tmp2_it)
            for idx1, nr in tmp2_it:
                if nr_prev > nr:
                    indexes['last'].append(idx1_prev + 1)
                    indexes['first'].append(idx1)
                idx1_prev = idx1
        indexes['last'].append(tmp.index[-1] + 1)

        for first, last in zip(indexes['first'], indexes['last']):
            tmp1 = tmp.iloc[((tmp.index >= first) & (tmp.index < last))]
            hlp = (tmp1['R_GT_n_diffAll'] + tmp1['t_GT_n_angDiff']).fillna(0).round(decimals=6)
            cnt = np.count_nonzero(hlp.to_numpy())
            frac = cnt / tmp1.shape[0]
            if frac > 0.5:
                rxc = tmp1['R_GT_n_diff_roll_deg'].fillna(0).abs().sum() > 1e-3
                ryc = tmp1['R_GT_n_diff_pitch_deg'].fillna(0).abs().sum() > 1e-3
                rzc = tmp1['R_GT_n_diff_yaw_deg'].fillna(0).abs().sum() > 1e-3
                txc = tmp1['t_GT_n_elemDiff_tx'].fillna(0).abs().sum() > 1e-4
                tyc = tmp1['t_GT_n_elemDiff_ty'].fillna(0).abs().sum() > 1e-4
                tzc = tmp1['t_GT_n_elemDiff_tz'].fillna(0).abs().sum() > 1e-4
                if rxc and ryc and rzc and txc and tyc and tzc:
                    tmp1['rt_change_type'] = pd.Series(['crt'] * int(tmp1.shape[0]))
                elif rxc and ryc and rzc:
                    tmp1['rt_change_type'] = pd.Series(['cra'] * int(tmp1.shape[0]))
                elif txc and tyc and tzc:
                    tmp1['rt_change_type'] = pd.Series(['cta'] * int(tmp1.shape[0]))
                elif rxc:
                    tmp1['rt_change_type'] = pd.Series(['crx'] * int(tmp1.shape[0]))
                elif ryc:
                    tmp1['rt_change_type'] = pd.Series(['cry'] * int(tmp1.shape[0]))
                elif rzc:
                    tmp1['rt_change_type'] = pd.Series(['crz'] * int(tmp1.shape[0]))
                elif txc:
                    tmp1['rt_change_type'] = pd.Series(['ctx'] * int(tmp1.shape[0]))
                elif tyc:
                    tmp1['rt_change_type'] = pd.Series(['cty'] * int(tmp1.shape[0]))
                elif tzc:
                    tmp1['rt_change_type'] = pd.Series(['ctz'] * int(tmp1.shape[0]))
                else:
                    tmp1['rt_change_type'] = pd.Series(['nv'] * int(tmp1.shape[0]))# no variation
            else:
                rxc = tmp1['R_GT_n_diff_roll_deg'].fillna(0).abs().sum() > 1e-3
                ryc = tmp1['R_GT_n_diff_pitch_deg'].fillna(0).abs().sum() > 1e-3
                rzc = tmp1['R_GT_n_diff_yaw_deg'].fillna(0).abs().sum() > 1e-3
                txc = tmp1['t_GT_n_elemDiff_tx'].fillna(0).abs().sum() > 1e-4
                tyc = tmp1['t_GT_n_elemDiff_ty'].fillna(0).abs().sum() > 1e-4
                tzc = tmp1['t_GT_n_elemDiff_tz'].fillna(0).abs().sum() > 1e-4
                if rxc and ryc and rzc and txc and tyc and tzc:
                    tmp1['rt_change_type'] = pd.Series(['jrt'] * int(tmp1.shape[0]))
                elif rxc and ryc and rzc:
                    tmp1['rt_change_type'] = pd.Series(['jra'] * int(tmp1.shape[0]))
                elif txc and tyc and tzc:
                    tmp1['rt_change_type'] = pd.Series(['jta'] * int(tmp1.shape[0]))
                elif rxc:
                    tmp1['rt_change_type'] = pd.Series(['jrx'] * int(tmp1.shape[0]))
                elif ryc:
                    tmp1['rt_change_type'] = pd.Series(['jry'] * int(tmp1.shape[0]))
                elif rzc:
                    tmp1['rt_change_type'] = pd.Series(['jrz'] * int(tmp1.shape[0]))
                elif txc:
                    tmp1['rt_change_type'] = pd.Series(['jtx'] * int(tmp1.shape[0]))
                elif tyc:
                    tmp1['rt_change_type'] = pd.Series(['jty'] * int(tmp1.shape[0]))
                elif tzc:
                    tmp1['rt_change_type'] = pd.Series(['jtz'] * int(tmp1.shape[0]))
                else:
                    tmp1['rt_change_type'] = pd.Series(['nv'] * int(tmp1.shape[0]))# no variation
            df_list.append(tmp1)
    df_new = pd.concat(df_list, axis=0, ignore_index=False)
    if 'filter_scene' in keywords:
        df_new = df_new.loc[df_new['rt_change_type'].str.contains(keywords['filter_scene'], regex=False)]
    return df_new

