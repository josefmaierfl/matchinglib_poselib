"""
Main script file for executing the whole test procedure for testing the autocalibration SW
"""
import sys, re, argparse, os, subprocess as sp, warnings, numpy as np

def main():
    parser = argparse.ArgumentParser(description='Main script file for executing the whole test procedure for '
                                                 'testing the autocalibration SW')
    parser.add_argument('--path', type=str, required=True,
                        help='Directory holding directories with template configuration files')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to images')
    parser.add_argument('--store_path', type=str, required=True,
                        help='Storing path for generated scenes and matches')
    parser.add_argument('--load_path', type=str, required=False,
                        help='Optional loading path for generated scenes and matches. '
                             'If not provided, store_path is used.')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise ValueError('Directory ' + args.path + ' holding directories with template scene '
                                                    'configuration files does not exist')
    if not os.path.exists(args.img_path):
        raise ValueError("Image path does not exist")
    if not os.path.exists(args.store_path):
        raise ValueError("Path for storing sequences does not exist")
    if len(os.listdir(args.store_path)) != 0:
        raise ValueError("Path for storing sequences is not empty")
    if args.load_path:
        if not os.path.exists(args.load_path):
            raise ValueError("Path for loading sequences does not exist")
    #else:
        #args.load_path = args.store_path
    try:
        if args.inlier_range:
            gen_configs(args.path, args.inlier_range, [], args.kpAccRange, args.img_path, args.store_path,
                        args.load_path, args.treatTPasCorrs)
        else:
            gen_configs(args.path, [], args.inlchrate_range, args.kpAccRange, args.img_path, args.store_path,
                        args.load_path, False)
    except FileExistsError:
        print(sys.exc_info()[0])
        return 1
    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise
    return 0


if __name__ == "__main__":
    main()