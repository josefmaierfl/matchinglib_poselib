"""
Reads configuration files and changes the option corrsPerRegion to use multiple matrices
"""
import sys, re, numpy as np, argparse, os, pandas as pd
import ruamel.yaml as yaml

def change_configs(input_path):
    # Load file names
    files_i = os.listdir(input_path)
    if len(files_i) == 0:
        raise ValueError('No files found.')
    files = []
    for i in files_i:
        fnObj = re.search('_initial\.', i, re.I)
        if fnObj:
            files.append(i)
    if len(files) == 0:
        raise ValueError('No files including _init found.')
    for i in files:
        tmp = i.replace('.yaml', '_tmp.yaml')
        pfnew = os.path.join(input_path, tmp)
        pfold = os.path.join(input_path, i)
        if os.path.isfile(pfnew):
            raise ValueError('File ' + pfnew + ' already exists.')
        write_config_file(pfold, pfnew)
        os.remove(pfold)
        os.rename(pfnew, pfold)

def write_config_file(finput, foutput):
    # Write first part of config file
    with open(finput, 'r') as fi:
        with open(foutput, 'w') as fo:
            li2 = fi.readline()
            while li2:
                lobj = re.search('corrsPerRegion:', li2)
                if lobj:
                    break
                else:
                    fo.write(li2)
                li2 = fi.readline()

    #Write corrsPerRegion
    mult_corrsPerRegion = [np.ndarray(shape=(3,3),
                                      dtype=float,
                                      order='C',
                                      buffer=
                                      np.array([[0.6, 0.2, 0.05],
                                                [0.6, 0.2, 0.05],
                                                [0.6, 0.2, 0.05]])),
                           np.ndarray(shape=(3, 3),
                                      dtype=float,
                                      order='C',
                                      buffer=
                                      np.array([[0.1, 0.8, 0.1],
                                                [0.1, 0.8, 0.1],
                                                [0.1, 0.8, 0.1]])),
                           np.ndarray(shape=(3, 3),
                                      dtype=float,
                                      order='C',
                                      buffer=
                                      np.array([[0.05, 0.2, 0.6],
                                                [0.05, 0.2, 0.6],
                                                [0.05, 0.2, 0.6]])),
                           np.ndarray(shape=(3, 3),
                                      dtype=float,
                                      order='C',
                                      buffer=
                                      np.array([[0.08, 0.2, 0.2],
                                                [0.06, 0.08, 0.2],
                                                [0.04, 0.06, 0.08]])),
                           np.ndarray(shape=(3, 3),
                                      dtype=float,
                                      order='C',
                                      buffer=
                                      np.array([[0.04, 0.06, 0.08],
                                                [0.06, 0.08, 0.2],
                                                [0.08, 0.2, 0.2]])),
                           np.ndarray(shape=(3, 3),
                                      dtype=float,
                                      order='C',
                                      buffer=
                                      np.array([[0.08, 0.06, 0.04],
                                                [0.2, 0.08, 0.06],
                                                [0.2, 0.2, 0.08]])),
                           np.ndarray(shape=(3, 3),
                                      dtype=float,
                                      order='C',
                                      buffer=
                                      np.array([[0.2, 0.2, 0.08],
                                                [0.2, 0.08, 0.06],
                                                [0.08, 0.06, 0.04]]))]
    with open(foutput, 'a') as fo:
        # Write mat
        yaml.dump({'corrsPerRegion': mult_corrsPerRegion}, stream=fo, Dumper=NoAliasDumper)

        with open(finput, 'r') as fi:
            li1 = fi.readline()
            line1_found = False
            line2_found = False
            pt = -1
            pos_list = []
            list_len = 4
            while li1:
                lobj = re.search('corrsPerRegion:', li1)
                if lobj:
                    line1_found = True
                elif line1_found:
                    lobj = re.search('corrsPerRegRepRate:', li1)
                    if lobj:
                        fi.seek(pos_list[(list_len + pt - 3) % list_len])
                        li1 = fi.readline()
                        fo.write(li1)
                        line1_found = False
                        line2_found = True
                    else:
                        if len(pos_list) < list_len:
                            pos_list.append(fi.tell())
                            pt = pt + 1
                        else:
                            pt = (pt + 1) % list_len
                            pos_list[pt] = fi.tell()
                elif line2_found:
                    lobj = re.search('corrsPerRegRepRate:', li1)
                    if lobj:
                        fo.write('corrsPerRegRepRate: 10\n')
                    else:
                        fo.write(li1)
                li1 = fi.readline()


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


def main():
    parser = argparse.ArgumentParser(description='Changes the option corrsPerRegion in multiple configuration '
                                                 'files')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory holding template configuration files')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' holding template scene configuration files does not exist')
    change_configs(args.path)


if __name__ == "__main__":
    main()
