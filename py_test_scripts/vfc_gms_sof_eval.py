"""
Released under the MIT License - https://opensource.org/licenses/MIT

Copyright (c) 2020 Josef Maier

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)

Description: Calculates the best performing parameter values with given testing data as specified in file
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

    import eval_mutex as em
    em.init_lock()
    em.acquire_lock()
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
    em.release_lock()

    return res