//
// Created by maierj on 5/11/20.
//

#ifndef GENERATEVIRTUALSEQUENCE_PREPAREMEGADEPTH_H
#define GENERATEVIRTUALSEQUENCE_PREPAREMEGADEPTH_H

#include <glob_includes.h>

struct megaDepthFolders{
    std::string mdImgF;//Folder holding images of a MegaDepth sub-set
    std::string mdDepth;//Folder holding depth files of a MegaDepth sub-set
    std::string sfmF;//Folder holding SfM files of a MegaDepth sub-set
    std::string sfmImgF;//Folder holding original images used within SfM of a MegaDepth sub-set
    std::string depthExt;//Extension of depth files

    megaDepthFolders(std::string &&mdImgF_, std::string &&mdDepth_, std::string &&sfmF_, std::string sfmImgF_, std::string depthExt_):
            mdImgF(move(mdImgF_)), mdDepth(move(mdDepth_)), sfmF(move(sfmF_)), sfmImgF(move(sfmImgF_)), depthExt(move(depthExt_)){}

    megaDepthFolders()= default;
};

#endif //GENERATEVIRTUALSEQUENCE_PREPAREMEGADEPTH_H
