//
// Created by maierj on 5/11/20.
//
#include <io_data.h>
#include "GTM/prepareMegaDepth.h"
#include "GTM/colmapBase.h"

using namespace colmap;
using namespace std;

bool loadMegaDepthFlow(const std::string &imgPath, const std::string &flowPath, std::vector<megaDepthData> &data);

bool convertMegaDepthData(const megaDepthFolders& folders, const std::string &flowSubFolder, std::vector<megaDepthData> &data){
    std::string flowPath = concatPath(getParentPath(folders.mdImgF), flowSubFolder);
    if(checkPathExists(flowPath)){
        return loadMegaDepthFlow(folders.mdImgF, flowPath, data);
    }
    if(!createDirectory(flowPath)){
        cerr << "Unable to create directory " << flowPath << endl;
        return false;
    }
    colmapBase cb;
    try {
        if (!cb.prepareColMapData(folders)) {
            return false;
        }
        if (!cb.getFileNames(folders)) {
            return false;
        }
    }catch (colmapException &e) {
        cerr << e.what() << endl;
        return false;
    }
    return cb.calculateFlow(flowPath, data);
}

bool loadMegaDepthFlow(const std::string &imgPath, const std::string &flowPath, std::vector<megaDepthData> &data){
    std::vector<std::string> flowFiles;
    if(loadImageSequenceNew(flowPath, "*.png", flowFiles) != 0){
        return false;
    }
    if(flowFiles.empty()){
        return false;
    }
    for(auto &fn: flowFiles){
        const string flowName = getFilenameFromPath(fn);
        size_t strpos = flowName.find('-');
        if(strpos == std::string::npos){
            continue;
        }
        string imgName1 = flowName.substr(0, strpos);
        imgName1 += ".jpg";
        imgName1 = concatPath(imgPath, imgName1);
        if(!checkFileExists(imgName1)){
            continue;
        }
        size_t strpos1 = flowName.find(".png");
        if(strpos1 == std::string::npos){
            continue;
        }
        string imgName2 = flowName.substr(strpos + 1, strpos1);
        imgName2 += ".jpg";
        imgName2 = concatPath(imgPath, imgName2);
        if(!checkFileExists(imgName2)){
            continue;
        }
        cv::Mat flow;
        if(!convertImageFlowFile(fn, nullptr, nullptr, flow)){
            continue;
        }
        data.emplace_back(move(imgName1), move(imgName2), move(flow));
    }
    return !data.empty();
}

