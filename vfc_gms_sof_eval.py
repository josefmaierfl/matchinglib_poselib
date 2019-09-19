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
from usac_eval import ji_env, get_time_fixed_kp, insert_opt_lbreak, prepare_io, check_par_file_exists, NoAliasDumper

def get_min_inlrat_diff_no_fig(**keywords):
    if 'res_par_name' not in keywords:
        raise ValueError('Missing parameter res_par_name')
    if 'err_type' not in keywords:
        raise ValueError('Missing parameter err_type')
    if len(keywords) < 2:
        raise ValueError('Wrong number of arguments for function pars_calc_single_fig_partitions')
    if 'data' not in keywords:
        raise ValueError('Missing data argument of function pars_calc_single_fig_partitions')
    if len(keywords['eval_columns']) != 1:
        raise ValueError('Wrong number of eval_columns')
    tmp = keywords['data'][keywords['eval_columns'][0]].loc[:, ['mean']].abs()
    tmp = tmp.loc[tmp.idxmin()]

    main_parameter_name = keywords['res_par_name']  # 'USAC_opt_search_min_inlrat_diff'
    # Check if file and parameters exist
    ppar_file, res = check_par_file_exists(main_parameter_name, keywords['res_folder'], 0)

    with open(ppar_file, 'a') as fo:
        # Write parameters
        alg_comb_bestl = tmp.index.values[0]
        if len(keywords['it_parameters']) != len(alg_comb_bestl):
            raise ValueError('Nr of search algorithms does not match')
        alg_w = {}
        for i, val in enumerate(keywords['it_parameters']):
            alg_w[val] = alg_comb_bestl[i]
        yaml.dump({main_parameter_name: {'Algorithm': alg_w,
                                         keywords['err_type']: float(tmp['mean'].values[0])}},
                  stream=fo, Dumper=NoAliasDumper, default_flow_style=False)

    return res