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

Description: Reads configuration files and changes the option acceptBadStereoPars to 1
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
    with open(foutput, 'w') as fo:
        with open(finput, 'r') as fi:
            li = fi.readline()
            while li:
                #lobj = re.search('acceptBadStereoPars:', li)
                lobj = re.search('imageOverlap:', li)
                if lobj:
                    #fo.write('acceptBadStereoPars: 1\n')
                    fo.write('imageOverlap: 8.5000000000000004e-01\n')
                else:
                    fo.write(li)
                li = fi.readline()


def main():
    parser = argparse.ArgumentParser(description='Changes the option acceptBadStereoPars in multiple configuration '
                                                 'files')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory holding template configuration files')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' holding template scene configuration files does not exist')
    change_configs(args.path)


if __name__ == "__main__":
    main()
