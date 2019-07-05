"""
Evaluates results from the autocalibration present in a pandas DataFrame as specified in file
Autocalibration-Parametersweep-Testing.xlsx
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np
import ruamel.yaml as yaml
#import modin.pandas as pd
import pandas as pd

def calcSatisticRt_th(data, store_path):
    #Select columns we need
    df = data[['R_diffAll', 'R_diff_roll_deg', 'R_diff_pitch_deg', 'R_diff_yaw_deg',
               't_angDiff_deg', 't_distDiff', 't_diff_tx', 't_diff_ty', 't_diff_tz',
               'USAC_parameters_estimator', 'USAC_parameters_refinealg', 'th']]
    #Group by USAC parameters 5&6 and calculate the statistic
    stats = df.groupby(['USAC_parameters_estimator', 'USAC_parameters_refinealg', 'th']).describe()
    errvalnames = stats.columns.values # Includes statistic name and error value names
    grp_names = stats.index.names #As used when generating the groups
    # holds the grouped names/entries within the group names excluding the last entry th
    grp_values = list(dict.fromkeys([i[0:2] for i in stats.index.values]))
    for it in errvalnames:
        if it[-1] != 'count':
            for grp in grp_values:
                tmp = stats.loc[:, it].loc[grp]
                print(tmp)





    #print(stats.groups('USAC_parameters_estimator').get_group(1).Index('mean'))
    sub = stats['R_diffAll']['mean']#['USAC_parameters_estimator']#['R_diffAll']['mean']#.get_group(1)
    a = sub.loc[0, 0, 0.4]
    print(sub.loc[sub.index.levels[0][1]])
    print('ending')
    # print(sub)
    # print('Eval')
    #a=[i[0:2] for i in grp_levels]
    #list(dict.fromkeys(a))
    # stats.index.levels
    # stats.index.names
    # stats.columns.levels#Exclude 'count'
    #a=stats.loc[:,errvalnames[18]].loc[grp_values[5]]
    #a=stats.loc[:, it].loc[grp_values[0][0:2]]


#Only for testing
def main():
    data = {'R_diffAll': [0.3, 0.5, 0.7, 0.4, 0.6] * 20,
            'R_diff_roll_deg': np.abs(np.random.randn(100) * 1000),
            'R_diff_pitch_deg': np.random.randn(100) * 10,
            'R_diff_yaw_deg': np.abs(np.random.randn(100)),
            't_angDiff_deg': [0.3, 0.5, 0.7, 0.4, 0.6] * 20,
            't_distDiff': np.abs(np.random.randn(100) * 100),
            't_diff_tx': np.random.randn(100),
            't_diff_ty': np.random.randn(100),
            't_diff_tz': np.random.randn(100),
            'USAC_parameters_estimator': np.random.randint(0, 3, 100),
            'USAC_parameters_refinealg': np.random.randint(0, 7, 100),
            'th': np.tile(np.arange(0.4, 0.9, 0.1), 20),
            'useless': [1, 1, 2, 3] * 25}
    data = pd.DataFrame(data)
    calcSatisticRt_th(data, '/home/maierj/work/Sequence_Test/py_test')


if __name__ == "__main__":
    main()