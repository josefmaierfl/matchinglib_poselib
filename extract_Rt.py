"""
Copies all stereo configurations from a generated scene into a config file
"""
import sys, re, numpy as np, argparse, os, pandas as pd, subprocess as sp, cv2#, yaml
import ruamel.yaml as yaml
from copy import deepcopy

def update_extrinsics(input_path, store_path):
    # Load configuration file names
    files_i = os.listdir(input_path)
    if len(files_i) == 0:
        raise ValueError('No files found.')
    files_config = []
    for i in files_i:
        fnObj = re.search('_initial\.', i, re.I)
        if fnObj:
            files_config.append(i)
    if len(files_config) == 0:
        raise ValueError('No files including _init found.')
    files_config.sort()

    #Check if file with generated scenes (command lines) exist
    file_cmd = ''
    for i in files_i:
        fnObj = re.match('(generated_scenes_index\.txt)', i, re.I)
        if fnObj:
            file_cmd = i
            break
    if not file_cmd:
        raise ValueError('No file holding executed command lines found.')
    file_cmd_full = os.path.join(input_path, file_cmd)
    with open(file_cmd_full, 'r') as cfi:
        li = cfi.readline()
        while li:
            cfiObj = re.match('.*--conf_file\s+(.+);(parSetNr\d+)', li)
            if cfiObj:
                conf_file = cfiObj.group(1)
                parsetNr = cfiObj.group(2)
                conf_file2 = os.path.basename(conf_file)
            else:
                raise ValueError('Unable to extract configuration file name and parSet number.')
            if conf_file2 not in files_config:
                raise ValueError('Cannot find matching configuration file in given directory.')
            scene_ovf = os.path.join(store_path, 'sequInfos.yaml')
            if not os.path.exists(scene_ovf):
                raise ValueError('Overview file for scenes does not exist.')

            #Read stereo configuration path
            fs_read = cv2.FileStorage(scene_ovf, cv2.FILE_STORAGE_READ)
            if not fs_read.isOpened():
                raise ValueError('Unable to read yaml overview file.')
            fn = fs_read.getNode(parsetNr)
            sub_path = fn.getNode('hashSequencePars').string()
            fs_read.release()
            sequ_path = os.path.join(store_path, sub_path)
            if not os.path.exists(sequ_path):
                raise ValueError('Unable to find sequence sub-path.')

            #Read stereo configurations
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
                r_stereo.append(fn.at(i).mat())
            fn = fs_read.getNode('t_stereo')
            if not fn.isSeq():
                raise ValueError('t_stereo is no sequence.')
            t_stereo = []
            n = fn.size()
            for i in range(0, n):
                t_stereo.append(fn.at(i).mat())
            K1 = fs_read.getNode('K1').mat()
            K2 = fs_read.getNode('K2').mat()
            fs_read.release()

            #Write first part of config file
            if not os.path.exists(conf_file):
                raise ValueError('Configuration file does not exist.')
            cfiObj = re.match('(.*)(\.\w{3,4})',conf_file)
            if cfiObj:
                b_name = cfiObj.group(1)
                name_end = cfiObj.group(2)
            else:
                raise ValueError('Unable to extract config file base name.')
            conf_file_new = b_name + '_tmp' + name_end
            with open(conf_file, 'r') as fi:
                with open(conf_file_new, 'w') as fo:
                    li2 = fi.readline()
                    while li2:
                        lobj = re.search('useSpecificCamPars:', li2)
                        lobj2 = re.search('specificCamPars:', li2)
                        if lobj:
                            fo.write('useSpecificCamPars: 1\n')
                        elif lobj2:
                            break
                        else:
                            fo.write(li2)
                        li2 = fi.readline()

            with open(conf_file_new, 'a') as fo:
                # Write stereo camera parameters
                sequ_new = []
                for i, j in zip(r_stereo, t_stereo):
                    node1 = {'R': i,
                             't': j,
                             'K1': K1,
                             'K2': K2
                    }
                    sequ_new.append(node1)
                yaml.dump({'specificCamPars': sequ_new}, stream=fo, Dumper=NoAliasDumper)#, default_flow_style=False)

                #Write last part of yaml file
                with open(conf_file, 'r') as fi:
                    li1 = fi.readline()
                    line1_found = False
                    line2_found = False
                    pt = -1
                    pos_list = []
                    list_len = 3
                    while li1:
                        lobj = re.search('specificCamPars:', li1)
                        if lobj:
                            line1_found = True
                        elif line1_found:
                            lobj = re.search('imageOverlap:', li1)
                            if lobj:
                                fi.seek(pos_list[(list_len + pt - 2) % list_len])
                                li1 = fi.readline()
                                fo.write(li1)
                                line1_found = False
                                line2_found = True
                            else:
                                if(len(pos_list) < list_len):
                                    pos_list.append(fi.tell())
                                    pt = pt + 1
                                else:
                                    pt = (pt + 1) % list_len
                                    pos_list[pt] = fi.tell()
                        elif line2_found:
                            fo.write(li1)
                        li1 = fi.readline()
            os.remove(conf_file)
            os.rename(conf_file_new, conf_file)
            li = cfi.readline()

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
    parser = argparse.ArgumentParser(description='Copies all stereo configurations from a generated scene into a config file')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory holding template configuration files')
    parser.add_argument('--store_path', type=str, required=True,
                        help='Storing path for generated scenes and matches')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' holding template scene configuration files does not exist')
    if not os.path.exists(args.store_path):
        raise ValueError("Path of stored sequences does not exist")
    if len(os.listdir(args.store_path)) == 0:
        raise ValueError("Path with stored sequences is empty")
    update_extrinsics(args.path, args.store_path)


if __name__ == "__main__":
    main()
