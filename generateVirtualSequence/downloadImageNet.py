"""
Script for downloading images from ImageNet randomly or only for specific categories
"""
import sys, re, argparse, os, numpy as np
import math
import requests
from bs4 import BeautifulSoup as Soup
import urllib.request
from urllib.parse import urlparse
import multiprocessing
import cv2
import timeit


regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)


def check_url(url):
    return re.match(regex, url) is not None


def requestServerReleaseVersion():
    serverVersionURL = 'http://www.image-net.org/api/text/imagenet.check_latest_version.php'
    vers_p = requests.get(serverVersionURL)
    return vers_p.content.decode('utf-8')


def obtainServerMetaData(path_store):
    StructureFileURL = 'http://www.image-net.org/api/xml/structure_released.xml'
    releaseStatusURL = 'http://www.image-net.org/api/xml/ReleaseStatus.xml'
    relstructfile = os.path.join(path_store, 'structure_released.xml')
    downloaded = False
    if not os.path.exists(relstructfile):
        structfdl = requests.get(StructureFileURL)
        with open(relstructfile, 'wb') as f:
            f.write(structfdl.content)
        downloaded = True
    rel_v = requestServerReleaseVersion()
    relstatf = os.path.join(path_store, 'ReleaseStatus.xml')
    if os.path.exists(relstatf):
        with open(relstatf, 'r') as f:
            soup = Soup(f, 'html.parser')
            clientReleaseVersion = soup.releasestatus.releasedata.string
        if(rel_v != clientReleaseVersion):
            relstatxml = requests.get(releaseStatusURL)
            with open(relstatf, 'wb') as f:
                f.write(relstatxml.content)
            if not downloaded:
                structfdl = requests.get(StructureFileURL)
                with open(relstructfile, 'wb') as f:
                    f.write(structfdl.content)
    else:
        relstatxml = requests.get(releaseStatusURL)
        with open(relstatf, 'wb') as f:
            f.write(relstatxml.content)
    return rel_v


def getSynSets(path_store, wnids, keywords):
    relstructfile = os.path.join(path_store, 'structure_released.xml')
    wnids_found = []
    with open(relstructfile, 'r') as f:
        soup = Soup(f, 'html.parser')
        tmp = soup.imagenetstructure.synset
        if wnids is None and keywords is None:
            for i in tmp.find_all('synset', recursive=False):
                wnids_found.append(i.get('wnid'))
        else:
            if wnids:
                tmp1 = tmp.find_all(wnid=wnids, recursive=True)
                if tmp1:
                    for i in tmp1:
                        wnids_found.append(i.get('wnid'))
            if keywords:
                kwds = r'(?:^|\s)' + r'(?:$|\s|,)|(?:^|\s)'.join(keywords) + r'(?:$|\s|,)'
                tmp1 = tmp.find_all(words=re.compile(kwds))
                if tmp1:
                    for i in tmp1:
                        wnids_found.append(i.get('wnid'))
                    wnids_found = list(dict.fromkeys(wnids_found))
                    del_ids = []
                    for i, wnid in enumerate(wnids_found):
                        tmp1 = tmp.find(wnid=wnid)
                        rest_ids = wnids_found[:i] + wnids_found[(i + 1):]
                        tmp2 = tmp1.find_all(wnid=rest_ids, recursive=True)
                        if tmp2:
                            del_ids += [a.get('wnid') for a in tmp2]
                    if del_ids:
                        del_ids = list(dict.fromkeys(del_ids))
                        for i in del_ids:
                            wnids_found.remove(i)
            if len(wnids_found) == 0:
                print('Given WNIDs and/or keywords not found on imagenet', sys.stderr)
                sys.exit(1)
        #Check for empty wnid URL files
        wnids_new = []
        getChildsIfParetEmpty(path_store, wnids_found, tmp, wnids_new)
    return wnids_new


def getChildsIfParetEmpty(path_store, wnids, xmlstruct, wnids_new):
    for wnid in wnids:
        urls = getImageXMLs(path_store, [wnid])
        if urls:
            wnids_new.append(wnid)
        else:
            if hasattr(xmlstruct, 'contents'):
                tmp = xmlstruct.find(wnid=wnid)
                if hasattr(tmp, 'contents'):
                    child_ids = []
                    for elem in tmp.contents:
                        if hasattr(elem, 'attrs'):
                            child_ids.append(elem.get('wnid'))
                    getChildsIfParetEmpty(path_store, child_ids, tmp, wnids_new)


def getImageXMLs(path_store, wnids):
    link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='
    linklist = {}
    for wnid in wnids:
        link1 = link + wnid
        imgsfile = os.path.join(path_store, wnid + '.txt')
        downloaded = False
        if not os.path.exists(imgsfile):
            imglinks = requests.get(link1)
            with open(imgsfile, 'wb') as f:
                f.write(imglinks.content)
            downloaded = True
            split_urls = imglinks.content.decode('utf-8').splitlines()
        if not downloaded:
            with open(imgsfile, 'r') as f:
                split_urls = f.readlines()
        split_urls1 = []
        for url in split_urls:
            if len(url) > 10 and check_url(url.strip()):
                split_urls1.append(url.strip())
        if len(split_urls1) > 0:
            linklist[wnid] = split_urls1
    return linklist


def downloadImgs(path_store, linklist, nr_imgs):
    id_cnt = len(linklist)
    nr = int(math.ceil(nr_imgs / id_cnt))
    nr_list = [nr] * id_cnt
    if id_cnt * nr < nr_imgs:
        nr_list[-1] += nr_imgs - id_cnt * nr
    cnt = 0
    for idx, i in enumerate(linklist):
        lll = len(linklist[i])
        if lll < nr_list[idx]:
            cnt += nr_list[idx] - lll
            nr_list[idx] = lll
        elif cnt > 0:
            new_cnt = nr_list[idx] + cnt
            if lll < new_cnt:
                diff = lll - nr_list[idx]
                cnt -= diff
                nr_list[idx] = lll
            else:
                nr_list[idx] = new_cnt
                cnt = 0

    nr_act_imgs = 0
    cnt = 0
    ts = []
    t_out = 8
    for idx, elem in enumerate(linklist.items()):
        path_id = os.path.join(path_store, elem[0])
        if not os.path.exists(path_id):
            os.mkdir(path_id)
        nr_imgs = len(os.listdir(path_id))
        if nr_imgs < nr_list[idx]:
            nr_links = len(elem[1])
            if cnt > 0:
                if nr_list[idx] + cnt <= nr_links:
                    nr_list[idx] += cnt
                    cnt = 0
                else:
                    cnt -= nr_links - nr_list[idx]
                    nr_list[idx] = nr_links
            diff = nr_list[idx] - nr_imgs
            d_idxs = np.arange(nr_links)
            np.random.shuffle(d_idxs)
            i = 0
            i2 = 0
            while i < diff and i2 < nr_links:
                link = elem[1][d_idxs[i2]]
                iname = os.path.basename(urlparse(link).path)
                fname = os.path.join(path_id, iname)
                if os.path.exists(fname):
                    nr_act_imgs += 1
                    i2 += 1
                    i += 1
                    continue
                start = timeit.default_timer()
                p = multiprocessing.Process(target=get_file, args=(link, fname))
                p.start()
                p.join(t_out)#Wait only for a few seconds
                if p.is_alive():
                    p.terminate()
                    p.join()
                    i2 += 1
                    if os.path.exists(fname):
                        os.remove(fname)
                    continue
                if p.exitcode != 0:
                    i2 += 1
                    if os.path.exists(fname):
                        os.remove(fname)
                    continue
                endt = timeit.default_timer()
                dt = endt - start
                if not os.path.exists(fname):
                    i2 += 1
                    continue
                fsize = os.path.getsize(fname)
                if fsize < 14000:
                    os.remove(fname)
                    i2 += 1
                    continue
                ts.append(fsize / (1048576 * dt))
                t_out = get_download_speed(ts, t_out)
                if not check_img(fname):
                    os.remove(fname)
                    i2 += 1
                    continue
                i += 1
                i2 += 1
                nr_act_imgs += 1
            nr_imgs = len(os.listdir(path_id))
            if nr_imgs == 0:
                os.removedirs(path_id)
                cnt += nr_list[idx]
            elif nr_imgs < nr_list[idx]:
                cnt += nr_list[idx] - nr_imgs
        else:
            nr_act_imgs += nr_imgs

    return nr_act_imgs != 0 and nr_act_imgs > 0.75 * nr_imgs


def get_file(link, fname):
    try:
        urllib.request.urlretrieve(link, fname)
    except:
        if os.path.exists(fname):
            os.remove(fname)
        sys.exit(1)
    sys.exit(0)


def check_img(filename):
    try:
        img = cv2.imread(filename)
    except:
        return False
    if not isinstance(img, np.ndarray):
        return False
    if img.shape[0] < 50 or img.shape[1] < 50:
        return False
    img = np.reshape(img, -1)
    var = np.var(img)
    return var > 200


def get_download_speed(t_vec, init_t):
    if len(t_vec) < 10:
        return init_t
    t1 = np.mean(t_vec)
    t1 *= 1.25
    t5m = 5 / t1
    t5m = max(min(15, t5m), 2)
    return t5m


def main():
    parser = argparse.ArgumentParser(description='Script for downloading images from ImageNet randomly or only for specific categories')
    parser.add_argument('--path_store', type=str, required=True,
                        help='Directory for storing images')
    parser.add_argument('--wnids', type=str, required=False, nargs='+', default=None,
                        help='If provided, only images using the provided ImageNet IDs (wnid) are downloaded. '
                             'Otherwise, random IDs are used until the desired number of images are downloaded.')
    parser.add_argument('--keywords', type=str, required=False, nargs='+', default=None,
                        help='If provided, the database is searched for these keywords and corresponding images are '
                             'downloaded.')
    parser.add_argument('--nr_imgs', type=int, required=True,
                        help='Number of images to download.')
    args = parser.parse_args()
    if not os.path.exists(args.path_store):
        os.mkdir(args.path_store)
    if args.nr_imgs < 1:
        raise ValueError('Number of images to download must be larger 0.')
    xmlfileFolder = os.path.join(args.path_store, 'xml_files')
    if not os.path.exists(xmlfileFolder):
        os.mkdir(xmlfileFolder)

    rel_v = obtainServerMetaData(xmlfileFolder)
    # args.wnids = ['n09478210', 'n09227839', 'abcf']
    # args.keywords = ['plant']
    wnids_found = getSynSets(xmlfileFolder, args.wnids, args.keywords)
    linklist = getImageXMLs(xmlfileFolder, wnids_found)
    ret = downloadImgs(args.path_store, linklist, args.nr_imgs)
    if not ret:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()