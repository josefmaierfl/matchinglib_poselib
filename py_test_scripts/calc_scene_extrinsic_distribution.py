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

Description: Read stereo configurations from a generated scene and calculate statistics on their parameters
"""
import sys, re, numpy as np, argparse, os, pandas as pd, subprocess as sp, cv2
import ruamel.yaml as yaml
from scipy.spatial.transform import Rotation as R


class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True

    def increase_indent(self, flow=False, sequence=None, indentless=False):
        return super(NoAliasDumper, self).increase_indent(flow, False)


# A yaml representer is for dumping structs into a yaml node.
# So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data
def opencv_matrix_representer(dumper, mat):
    if mat.ndim > 1:
        mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    else:
        mapping = {'rows': mat.shape[0], 'cols': 1, 'dt': 'd', 'data': mat.tolist()}
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)
yaml.add_representer(np.ndarray, opencv_matrix_representer)


def read_single_extrinsics(store_path, parSetNr):
    scene_ovf = os.path.join(store_path, 'sequInfos.yaml')
    if not os.path.exists(scene_ovf):
        raise ValueError('Overview file for scenes does not exist.')
    # Read stereo configuration path
    fs_read = cv2.FileStorage(scene_ovf, cv2.FILE_STORAGE_READ)
    if not fs_read.isOpened():
        raise ValueError('Unable to read yaml overview file.')
    fn = fs_read.getNode('parSetNr' + str(parSetNr))
    sub_path = fn.getNode('hashSequencePars').string()
    fs_read.release()
    sequ_path = os.path.join(store_path, sub_path)
    if not os.path.exists(sequ_path):
        raise ValueError('Unable to find sequence sub-path.')
    # Read stereo configurations
    sequ_par_fn = os.path.join(sequ_path, 'sequPars.yaml.gz')
    if not os.path.exists(sequ_par_fn):
        raise ValueError('Scene parameter file does not exist.')
    fs_read = cv2.FileStorage(sequ_par_fn, cv2.FILE_STORAGE_READ)
    if not fs_read.isOpened():
        raise ValueError('Unable to read scene parameter file.')
    fn = fs_read.getNode('R_stereo')
    if not fn.isSeq():
        raise ValueError('R_stereo is no sequence.')
    r_stereo = []
    n = fn.size()
    for i in range(0, n):
        r_stereo.append(np.asarray(fn.at(i).mat()))
    fn = fs_read.getNode('t_stereo')
    if not fn.isSeq():
        raise ValueError('t_stereo is no sequence.')
    t_stereo = []
    n = fn.size()
    for i in range(0, n):
        t_stereo.append(np.asarray(fn.at(i).mat()))
    fs_read.release()

    rt_pars = {'rx': [], 'ry': [], 'rz': [], 'tx': [], 'ty': [], 'tz': []}
    for i, j in zip(r_stereo, t_stereo):
        r = R.from_matrix(i)
        ang = r.as_euler('YZX', degrees=True)
        rt_pars['rx'].append(ang[2])
        rt_pars['ry'].append(ang[0])
        rt_pars['rz'].append(ang[1])
        rt_pars['tx'].append(j[0][0])
        rt_pars['ty'].append(j[1][0])
        rt_pars['tz'].append(j[2][0])

    df = pd.DataFrame(rt_pars)
    return df


def calc_single_extr_stats(out_path, store_path, parSetNr):
    df = read_single_extrinsics(store_path, parSetNr)
    stats = df.describe()
    file_name = 'rt_stats_parSetNr' + str(parSetNr)
    file_path = os.path.join(out_path, file_name + '.csv')
    cnt = 1
    file_name_init = file_name
    while os.path.exists(file_path):
        file_name = file_name_init + '_' + str(int(cnt))
        file_path = os.path.join(out_path, file_name + '.csv')
        cnt += 1
    with open(file_path, 'w') as f:
        stats.to_csv(index=True, sep=';', path_or_buf=f, header=True, na_rep='nan')


def main():
    parser = argparse.ArgumentParser(description='Extracts poses from a generated scene and calculates their statistics')
    parser.add_argument('--path_out', type=str, required=True,
                        help='Directory for writing statistics')
    parser.add_argument('--in_path', type=str, required=True,
                        help='Storing path for generated scenes and matches')
    parser.add_argument('--parSetNr', type=int, required=True,
                        help='Specify a specific parSetNr for extracting R&t only from this dataset')
    args = parser.parse_args()
    if not os.path.exists(args.path_out):
        raise ValueError('Directory ' + args.path + ' for storing statistics does not exist')
    if not os.path.exists(args.in_path):
        raise ValueError("Path of stored sequences does not exist")
    if len(os.listdir(args.in_path)) == 0:
        raise ValueError("Path with stored sequences is empty")
    calc_single_extr_stats(args.path_out, args.in_path, args.parSetNr)


if __name__ == "__main__":
    main()
