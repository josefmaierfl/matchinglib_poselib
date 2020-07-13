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
import sys, os, warnings, re, argparse
from copy import deepcopy
import shutil, tempfile
import stat


def get_tex_folders(path):
    sub_dirs = [os.path.join(path, a) for a in os.listdir(path)]
    ss_dirs = []
    notex = True
    for i in sub_dirs:
        if os.path.isdir(i):
            ss_dir = get_tex_folders(i)
            if ss_dir:
                ss_dirs += ss_dir
        elif notex and os.path.isfile(i):
            (_, ext) = os.path.splitext(i)
            if ext == '.tex':
                notex = False
                ss_dirs += [path]
    return ss_dirs


def recompile(path, pout, cpu_use):
    tex_dirs = get_tex_folders(path)

    ret = 0
    for td in tex_dirs:
        tmp_path = tempfile.mkdtemp()
        dest = copytree(td, tmp_path,
                        ignore=shutil.ignore_patterns('*.pdf', '*.txt', '*.yaml'),
                        ignore_dangling_symlinks=True)
        pdf_info = {'tex_dir': dest,
                    'tex_fname': [],
                    'pdf_fname': [],
                    'text': [],
                    'build_gloss': False,
                    'mk_idx': False,
                    'ext': False}
        rel_path = os.path.relpath(td, start=path)
        outp = os.path.join(pout, rel_path)
        if os.path.basename(outp) == 'tex':
            pdf_path = os.path.join(os.path.dirname(outp), 'pdf')
        else:
            pdf_path = os.path.join(outp, 'pdf')
        os.makedirs(pdf_path, exist_ok=True)
        for f in os.listdir(dest):
            f = os.path.join(dest, f)
            if os.path.isfile(f):
                (f_name, ext) = os.path.splitext(f)
                if ext == '.tex':
                    basename = os.path.basename(f_name)
                    pdf_info['tex_fname'].append(basename + '_na' + ext)
                    text = []
                    with open(f, 'r') as fi:
                        li = fi.readline()
                        founda = False
                        while li:
                            if not founda:
                                aobj = re.search(r'\\author\{.*\}', li)
                                if aobj:
                                    founda = True
                                    li = fi.readline()
                                    continue
                            if not pdf_info['build_gloss']:
                                gobj = re.search(r'\\printunsrtglossary', li)
                                if gobj:
                                    pdf_info['build_gloss'] = True
                            if not pdf_info['mk_idx']:
                                iobj = re.search(r'\\listoffigures', li)
                                if iobj:
                                    pdf_info['mk_idx'] = True
                                else:
                                    iobj = re.search(r'\\tableofcontents', li)
                                    if iobj:
                                        pdf_info['mk_idx'] = True
                            if not pdf_info['ext']:
                                eobj = re.search(r'\\usepgfplotslibrary\{external\}', li)
                                if eobj:
                                    pdf_info['ext'] = True
                            text.append(li)
                            li = fi.readline()
                    pdf_info['text'].append(''.join(text))
                    pdf_info['pdf_fname'].append(os.path.join(pdf_path, basename + '.pdf'))
        from statistics_and_plot import compile_tex
        ret += compile_tex(pdf_info['text'],
                           pdf_info['tex_dir'],
                           pdf_info['tex_fname'],
                           make_fig_index=pdf_info['mk_idx'],
                           out_pdf_filen=pdf_info['pdf_fname'],
                           figs_externalize=pdf_info['ext'],
                           build_glossary=pdf_info['build_gloss'],
                           nr_cpus=cpu_use)
        shutil.rmtree(tmp_path)
    return ret


def inspect_dir(path, includes):
    ignores = []
    ign_cnt = 0
    folder_use = []
    if os.path.isfile(path):
        _, ext = os.path.splitext(path)
        if all(ext.lower() != a for a in includes):
            ignores = [ext]
            ign_cnt = 1
    else:
        content = os.listdir(path)
        for i in content:
            i_folder = os.path.join(path, i)
            folders, ign, cnt = inspect_dir(i_folder)
            is_dir = os.path.isdir(i_folder)
            if (is_dir and cnt == len(os.listdir(i_folder))) or (not is_dir and cnt):
                ign_cnt += 1
            elif is_dir:
                if folders:
                    folder_use += folders
                else:
                    folder_use.append(i_folder)
            ignores += ign
    return folder_use, ignores, ign_cnt


def copy_pdfs(path, pout):
    folder_use, ignores, ign_cnt = inspect_dir(path, ['.pdf'])
    ignores = list(dict.fromkeys(ignores))
    ignores = ['*' + ig for ig in ignores]
    main_folder = os.path.commonpath(folder_use)
    base = os.path.basename(main_folder)
    main2 = os.path.join(pout, base)
    try:
        os.mkdir(main2)
    except FileExistsError:
        pass
    for fo in folder_use:
        rel_path = os.path.relpath(fo, start=main_folder)
        abs2 = os.path.normpath(os.path.join(main2, rel_path))
        os.makedirs(abs2, exist_ok=True)
        ign = deepcopy(ignores)
        ad_ignores = [os.path.join(fo, a) for a in os.listdir(fo)]
        ad_ignores = [os.path.basename(a) for a in ad_ignores if os.path.isdir(a)]
        if ad_ignores:
            ign += ad_ignores
        ign = set(ign)
        copytree(fo, abs2, symlinks=False, ignore_dangling_symlinks=True,
                 ignore=shutil.ignore_patterns(*ign))


def copytree(src, dst, symlinks=False, ignore=None, ignore_dangling_symlinks=False):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    errors = []
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        try:
            if symlinks and os.path.islink(s):
                linkto = os.readlink(s)
                # ignore dangling symlink if the flag is on
                if not os.path.exists(linkto) and ignore_dangling_symlinks:
                    continue
                # otherwise let the copy occurs. copy2 will raise an error
                if os.path.lexists(d):
                    os.remove(d)
                if os.path.isdir(s):
                    copytree(s, d, symlinks, ignore)
                else:
                    shutil.copy2(s, d)
            elif os.path.isdir(s):
                copytree(s, d, symlinks, ignore)
            elif not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
                # catch the Error from the recursive copytree so that we can
                # continue with other files
        except shutil.Error as err:
            errors.extend(err.args[0])
        except OSError as why:
            errors.append((s, d, str(why)))
    try:
        shutil.copystat(src, dst)
    except OSError as why:
        # Copying file access times may fail on Windows
        if getattr(why, 'winerror', None) is None:
            errors.append((src, dst, str(why)))
    if errors:
        raise shutil.Error(errors)
    return dst


def main():
    parser = argparse.ArgumentParser(description='Loads all tex-files within a given directory including all '
                                                 'sub-directories, deletes author information and builds PDFs. If '
                                                 'option --copy_only is specified, all and only PDF documents are '
                                                 'copied to the destination folder while preserving the folder '
                                                 'structure.')
    parser.add_argument('--path', type=str, required=True,
                        help='Main path for loading tex-files or copying PDFs')
    parser.add_argument('--pout', type=str, required=True,
                        help='Main output path for storing generated/copied PDFs')
    parser.add_argument('--nrCPUs', type=int, required=False, default=-6,
                        help='Number of CPU cores for parallel processing. If a negative value is provided, '
                             'the program tries to find the number of available CPUs on the system - if it fails, '
                             'the absolute value of nrCPUs is used. Default: -6')
    parser.add_argument('--copy_only', type=bool, nargs='?', required=False, default=False, const=True,
                        help='All and only PDF documents are copied to the destination folder while preserving the '
                             'folder structure. No compilation of tex-files.')
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
    if not os.path.exists(args.path) or not os.path.isdir(args.path):
        raise ValueError("Input path does not exist/must be a directry")
    if not os.path.exists(args.pout):
        os.makedirs(args.pout, exist_ok=True)
    if len(os.listdir(args.pout)) != 0:
        warnings.warn('Directory ' + args.pout + ' is not empty.', UserWarning)
        print('Resume (y/n)?')
        ui = input('Resume (y/n)? ')
        if ui != 'y':
            sys.exit(1)

    if not args.copy_only:
        ret = recompile(args.path, args.pout, cpu_use)
        sys.exit(ret)
    else:
        copy_pdfs(args.path, args.pout)
        sys.exit(0)


if __name__ == "__main__":
    main()
