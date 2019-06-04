"""
Loads initial configuration files and creates scenes which can be further used to extract information
that is needed for the final configuration files
"""
import sys, re, numpy as np, argparse, os, pandas as pd, subprocess as sp
from copy import deepcopy

def gen_scenes(test_app, input_path, img_path, store_path, message_path):
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
    files.sort()
    ovfile_new = os.path.join(input_path, 'generated_scenes_index.txt')

    cmd_line = [test_app, '--img_path', img_path, '--img_pref', '/', '--store_path', store_path, '--conf_file']

    cnt = 0
    cnt2 = 0
    with open(ovfile_new, 'w') as fo:
        for i in files:
            cmd_line_full = deepcopy(cmd_line)
            cmd_line_full.append(os.path.join(input_path, i))
            fnObj = re.match('(.*)_initial\..*', i, re.I)
            if fnObj:
                base = fnObj.group(1)
            else:
                base = str(cnt)
            err_out = 'stderr_' + base + '.txt'
            mess_out = 'stdout_' + base + '.txt'
            fname_cerr = os.path.join(message_path, err_out)
            fname_mess = os.path.join(message_path, mess_out)
            cerrf = open(fname_cerr, 'w')
            messf = open(fname_mess, 'w')
            cnt3 = 3
            while cnt3 > 0:
                try:
                    sp.run(cmd_line_full, stdout=messf, stderr=cerrf, check=True)
                    fo.write(' '.join(map(str, cmd_line_full)) + ';parSetNr' + str(cnt2) + '\n')
                    cnt2 = cnt2 + 1
                    cnt3 = 0
                except sp.CalledProcessError as e:
                    cnt3 = cnt3 - 1
                    print('Failed to generate scene!')
                    if cnt3 > 0:
                        print('Trying again!')
                        cerrf.close()
                        messf.close()
                        fname_cerr_err = os.path.join(message_path, 'err_' + str(cnt3) + '_' + err_out)
                        fname_mess_err = os.path.join(message_path, 'err_' + str(cnt3) + '_' + mess_out)
                        os.rename(fname_cerr, fname_cerr_err)
                        os.rename(fname_mess, fname_mess_err)
                        cerrf = open(fname_cerr, 'w')
                        messf = open(fname_mess, 'w')
                        continue
                    err_filen = 'errorInfo_' + base + '.txt'
                    fname_err = os.path.join(message_path, err_filen)
                    if e.cmd or e.stderr or e.stdout:
                        with open(fname_err, 'w') as fo1:
                            if e.cmd:
                                for line in e.cmd:
                                    fo1.write(line + ' ')
                                fo1.write('\n\n')
                            if e.stderr:
                                for line in e.stderr:
                                    fo1.write(line + ' ')
                                fo1.write('\n\n')
                            if e.stdout:
                                for line in e.stdout:
                                    fo1.write(line + ' ')
            cerrf.close()
            messf.close()
            cnt = cnt + 1

def main():
    parser = argparse.ArgumentParser(description='Execute generateVirtualSequence for multiple configuration files')
    parser.add_argument('--executable', type=str, required=True,
                        help='Executable of the application generating the sequences')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory holding template configuration files')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to images')
    parser.add_argument('--store_path', type=str, required=True,
                        help='Storing path for generated scenes and matches')
    parser.add_argument('--message_path', type=str, required=True,
                        help='Storing path for text files containing error and normal mesages during the '
                             'generation process of scenes and matches')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' holding template scene configuration files does not exist')
    if not os.path.exists(args.img_path):
        raise ValueError("Image path does not exist")
    if not os.path.exists(args.store_path):
        raise ValueError("Path for storing sequences does not exist")
    if len(os.listdir(args.store_path)) != 0:
        raise ValueError("Path for storing sequences is not empty")
    if not os.path.isfile(args.executable):
        raise ValueError('Executable ' + args.executable + ' for generating scenes does not exist')
    elif not os.access(args.executable,os.X_OK):
        raise ValueError('Unable to execute ' + args.executable)
    if not os.path.exists(args.message_path):
        raise ValueError("Path for storing stdout and stderr does not exist")
    gen_scenes(args.executable, args.path, args.img_path, args.store_path, args.message_path)


if __name__ == "__main__":
    main()

