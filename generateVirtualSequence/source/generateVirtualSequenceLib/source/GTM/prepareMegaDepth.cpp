//
// Created by maierj on 5/11/20.
//
#include "GTM/prepareMegaDepth.h"
#include "GTM/colmapBase.h"

using namespace colmap;
using namespace std;

bool convertMegaDepthData(const megaDepthFolders& folders){
    colmapBase cb;
    if(!cb.prepareColMapData(folders)){
        return false;
    }
    if(!cb.getFileNames(folders)){
        return false;
    }
}

