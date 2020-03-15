import sys, os, subprocess as sp, warnings, math, re, argparse
from copy import deepcopy
import shutil, tempfile
import multiprocessing


def get_tex_folders(path):
    sub_dirs = os.listdir(path)
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
        dest = shutil.copytree(td, tmp_path,
                               ignore=shutil.ignore_patterns('*.pdf'),
                               ignore_dangling_symlinks=True,
                               dirs_exist_ok=True)
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
                    pdf_info['tex_fname'].append(''.join(text))
                    pdf_info['pdf_fname'].append(os.path.join(pdf_path, basename + '.pdf'))
        from statistics_and_plot import compile_tex
        ret += compile_tex(pdf_info['tex_fname'],
                           pdf_info['tex_dir'],
                           pdf_info['tex_fname'],
                           make_fig_index=pdf_info['mk_idx'],
                           out_pdf_filen=pdf_info['pdf_fname'],
                           figs_externalize=pdf_info['ext'],
                           build_glossary=pdf_info['build_gloss'],
                           nr_cpus=cpu_use)
        shutil.rmtree(tmp_path)
    return ret


def inspect_dir(path):
    ignores = []
    ign_cnt = 0
    folder_use = []
    if os.path.isfile(path):
        _, ext = os.path.splitext(path)
        if ext.lower() != '.pdf':
            ignores = [ext]
            ign_cnt = 1
    else:
        content = os.listdir(path)
        for i in content:
            i_folder = os.path.join(path, i)
            folders, ign, cnt = inspect_dir(i_folder)
            is_dir = os.path.isdir(i_folder)
            if is_dir and  cnt == len(os.listdir(i_folder)):
                ign_cnt += 1
            elif is_dir:
                if folders:
                    folder_use += folders
                else:
                    folder_use.append(i_folder)
            ignores += ign
    return folder_use, ignores, ign_cnt


def copy_pdfs(path, pout):
    folder_use, ignores, ign_cnt = inspect_dir(path)
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
        ad_ignores = [a for a in ad_ignores if os.path.isdir(a)]
        if ad_ignores:
            ign += ad_ignores
        ign = set(ign)
        shutil.copytree(fo, abs2, symlinks=False, ignore_dangling_symlinks=True,
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns(*ign))


def main():
    parser = argparse.ArgumentParser(description='Loads all tex-files within a given directory including all '
                                                 'sub-directories, deletes author information and builds PDFs. If '
                                                 'option --copy_only is specified, all and only PDF documents are '
                                                 'copied to the destination folder while preserving the folder '
                                                 'structure.')
    parser.add_argument('--path', type=str, required=True,
                        help='Main path for loading tex-files')
    parser.add_argument('--pout', type=str, required=True,
                        help='Main output path for storing generated PDFs')
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
