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

Description: Reads configuration files and changes the option truePosRange based on the information in the file name
"""
import sys, re, numpy as np, argparse, os, pandas as pd


def change_configs(input_path):
    # Load file names
    files_i = os.listdir(input_path)
    if len(files_i) == 0:
        raise ValueError('No files found.')
    files = []
    for i in files_i:
        fnObj = re.search(r'_initial\.', i, re.I)
        if fnObj:
            files.append(i)
    if len(files) == 0:
        raise ValueError('No files including _init found.')
    files2 = {'file': [], 'tp_nr1': [], 'tp_nr2': []}
    for i in files:
        fnObj = re.match(r'.*_kp-distr-(?:(?:half-img)|(?:1corn)|(?:equ))_depth-(?:F|(?:NM)|(?:NMF))_TP-([0-9to]+).*', i)
        if fnObj:
            if fnObj.group(1):
                tp_nr = fnObj.group(1)
            else:
                raise ValueError('Unable to extract number of keypoints from ' + i)
        else:
            raise ValueError('Unable to extract number of keypoints from ' + i)
        fnObj = re.match(r'(\d+)to(\d+)', tp_nr)
        if fnObj:
            if fnObj.group(1):
                tp_nr1 = fnObj.group(1)
            else:
                raise ValueError('Unable to extract number of keypoints from ' + i)
            if fnObj.group(2):
                tp_nr2 = fnObj.group(2)
            else:
                raise ValueError('Unable to extract number of keypoints from ' + i)
        else:
            tp_nr1 = tp_nr
            tp_nr2 = tp_nr
        files2['file'].append(i)
        files2['tp_nr1'].append(tp_nr1)
        files2['tp_nr2'].append(tp_nr2)
    if len(files2['file']) == 0:
        raise ValueError('No files that includes number of TP found.')
    for i, file in enumerate(files2['file']):
        tmp = file.replace('.yaml', '_tmp.yaml')
        pfnew = os.path.join(input_path, tmp)
        pfold = os.path.join(input_path, file)
        if os.path.isfile(pfnew):
            raise ValueError('File ' + pfnew + ' already exists.')
        write_config_file(pfold, pfnew, [files2['tp_nr1'][i], files2['tp_nr2'][i]])
        os.remove(pfold)
        os.rename(pfnew, pfold)


def write_config_file(finput, foutput, tp_range):
    with open(foutput, 'w') as fo:
        with open(finput, 'r') as fi:
            li = fi.readline()
            line1_found = False
            line2_found = False
            while li:
                lobj = re.search('truePosRange:', li)
                if lobj:
                    line1_found = True
                    fo.write(li)
                elif line1_found:
                    line1_found = False
                    line2_found = True
                    rpstr = r'\g<1>{}'.format(tp_range[0])
                    li1 = re.sub(r'(\s*first:\s*)\d+',
                                 rpstr, li)
                    fo.write(li1)
                elif line2_found:
                    line2_found = False
                    rpstr = r'\g<1>{}'.format(tp_range[1])
                    li1 = re.sub(r'(\s*second:\s*)\d+',
                                 rpstr, li)
                    fo.write(li1)
                else:
                    fo.write(li)
                li = fi.readline()


def main():
    parser = argparse.ArgumentParser(description='Reads configuration files and changes the option truePosRange '
                                                 'based on the information in the file name')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory holding template configuration files')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' holding template scene configuration files does not exist')
    change_configs(args.path)


if __name__ == "__main__":
    main()
