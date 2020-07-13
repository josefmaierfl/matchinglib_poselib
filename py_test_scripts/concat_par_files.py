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
"""
import sys, re, argparse, os
import ruamel.yaml as yaml
import multiprocessing
import warnings

warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


def readOpenCVYaml(file, isstr = False):
    if isstr:
        data = file.split('\n')
    else:
        with open(file, 'r') as fi:
            data = fi.readlines()
    data = [line for line in data if line and line[0] is not '%']
    try:
        data = yaml.load_all("\n".join(data))
        data1 = {}
        for d in data:
            data1.update(d)
        data = data1
    except:
        print('Exception during reading yaml.')
        e = sys.exc_info()
        print(str(e))
        sys.stdout.flush()
        raise BaseException
    return data


def is_parset_file(name):
    if name.find('parSetNr_', 0, 9) == 0:
        return True
    return False


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    # alist.sort(key=natural_keys) sorts in human order
    # http://nedbatchelder.com/blog/200712/human_sorting.html
    # (See Toothy's implementation in the comments)
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def concat_files(path, cpu_cnt):
    sub_paths = [os.path.join(path, a) for a in os.listdir(path) if RepresentsInt(a) and not os.path.isfile(os.path.join(path, a))]

    from usac_eval import NoAliasDumper
    for sp in sub_paths:
        parset_path = os.path.join(sp, 'pars')
        concat_yaml = {}
        if os.path.exists(parset_path):
            print()
            print('Concatenating parameter files...')
            sub_files = [os.path.join(parset_path, name) for name in os.listdir(parset_path) if is_parset_file(name)]
            sub_files = sorted(sub_files, key=natural_keys)
            cnt_dot = 0
            res1 = 0
            with multiprocessing.Pool(processes=cpu_cnt) as pool:
                results = [pool.apply_async(readOpenCVYaml, (t, )) for t in sub_files]
                for idx, r in enumerate(results):
                    while 1:
                        sys.stdout.flush()
                        try:
                            concat_yaml['parSetNr' + str(idx)] = r.get(2.0)
                            break
                        except multiprocessing.TimeoutError:
                            if cnt_dot >= 90:
                                print()
                                cnt_dot = 0
                            sys.stdout.write('.')
                            cnt_dot = cnt_dot + 1
                        except ChildProcessError:
                            res1 = 1
                            pool.terminate()
                            break
                        except:
                            res1 = 2
                            pool.terminate()
                            break
                    if res1:
                        break
            if res1:
                return res1
            _, fext = os.path.splitext(sub_files[0])
            yaml_file = os.path.join(sp, 'allRunsOverview' + fext)
            with open(yaml_file, 'w') as fo:
                yaml.dump(concat_yaml, stream=fo, Dumper=NoAliasDumper, default_flow_style=False)
    return 0


def main():
    parser = argparse.ArgumentParser(description='Concats parameter yaml files of found tests')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory holding numbered directories with results')
    parser.add_argument('--nrCPUs', type=int, required=False, default=4,
                        help='Number of CPU cores for parallel processing. If a negative value is provided, '
                             'the program tries to find the number of available CPUs on the system - if it fails, '
                             'the absolute value of nrCPUs is used. Default: 4')
    args = parser.parse_args()
    if args.nrCPUs > 72 or args.nrCPUs == 0:
        raise ValueError("Unable to use " + str(args.nrCPUs) + " CPU cores.")
    av_cpus = os.cpu_count()
    if av_cpus:
        if args.nrCPUs < 0:
            cpu_use = av_cpus
        elif args.nrCPUs > av_cpus:
            print('Demanded ' + str(args.nrCPUs) + ' but only ' + str(av_cpus) + ' CPUs are available. Using '
                  + str(av_cpus) + ' CPUs.')
            cpu_use = av_cpus
        else:
            cpu_use = args.nrCPUs
    elif args.nrCPUs < 0:
        print('Unable to determine # of CPUs. Using ' + str(abs(args.nrCPUs)) + ' CPUs.')
        cpu_use = abs(args.nrCPUs)
    else:
        cpu_use = args.nrCPUs

    ret = concat_files(args.path, cpu_use)
    sys.exit(ret)


if __name__ == "__main__":
    main()
