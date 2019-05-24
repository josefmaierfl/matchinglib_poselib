"""
Loads initial configuration files and creates scenes which can be further used to extract information
that is needed for the final configuration files
"""
import sys, re, numpy as np, argparse, os, pandas as pd, subprocess as sp

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
    ovfile_new = os.path.join(input_path, 'generated_scenes_index.txt')

    cmd_line = [test_app, '--img_path', img_path, '--img_pref', '/', '--store_path', store_path, '--conf_file']

    cnt = 0
    for i in files:
        cmd_line_full = cmd_line
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
        try:
            sp.run(cmd_line_full, stdout=messf, stderr=cerrf, check=True)
        except sp.CalledProcessError as e:
            err_filen = 'errorInfo_' + base + '.txt'
            fname_err = os.path.join(message_path, err_filen)
            with open(fname_err, 'w') as fo:
                for line in e.cmd:
                    fo.write(line)
                fo.write('\n\n')
                for line in e.stderr:
                    fo.write(line)
                fo.write('\n\n')
                for line in e.stdout:
                    fo.write(line)
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
                        help='Storing path for generated scenes and matches')
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

