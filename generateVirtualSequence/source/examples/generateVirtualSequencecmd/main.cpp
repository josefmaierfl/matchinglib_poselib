//Released under the MIT License - https://opensource.org/licenses/MIT
//
//Copyright (c) 2020 Josef Maier
//
//Permission is hereby granted, free of charge, to any person obtaining
//a copy of this software and associated documentation files (the "Software"),
//to deal in the Software without restriction, including without limitation
//the rights to use, copy, modify, merge, publish, distribute, sublicense,
//and/or sell copies of the Software, and to permit persons to whom the
//Software is furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included
//in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//Author: Josef Maier (josefjohann-dot-maier-at-gmail-dot-at)

 #include "loadMatches.h"
 #include "argvparser.h"

#include "io_data.h"
#include "readPointClouds.h"
#include "readGTM.h"

 using namespace CommandLineProcessing;

 void SetupCommandlineParser(ArgvParser& cmd, int argc, char* argv[])
 {
     cmd.setIntroductoryDescription("Example program to show how the generated matches should be loaded.\n"
                                    "It prints descriptor distance statistics of loaded data.");
     //define error codes
     cmd.addErrorCode(0, "Success");
     cmd.addErrorCode(1, "Error");

     cmd.setHelpOption("help", "h","Example interface for reading data of generated semi-real-world stereo sequences. "
                                   "It supports reading Ground Truth Matches (GTM) in addition to data of single stereo frames, full "
                                   "sequences (3D data and correspondence information), and multiple sequences "
                                   "(3D data and correspondence information based on multiple parameter sets)");
     cmd.defineOption("file", "<Use if you only want to load matches for a single stereo frame. "
                              "Option fileRt must also be provided.\n"
                              "Path and filename (including file ending) of single stereo frame matches.>",
             ArgvParser::OptionRequiresValue);
     cmd.defineOption("fileRt", "<Use if you only want to load matches for a single stereo frame. "
                                "Option file must also be provided.\n"
                                "Path and filename (including file ending) of the sequence data of a single frame.>",
                      ArgvParser::OptionRequiresValue);
     cmd.defineOption("sequPath", "<Use if you want to load a full sequence including 3D data and matches.\n"
                                "By additionally providing option matchPath, only matches from the given folder are loaded.\n"
                                "Otherwise, all folders (multiple sequences with different matches but identical 3D data) are loaded.\n"
                                "Full path of the sequence (last folder corresponds to a hash value and must contain\n"
                                "yaml/xml files, *.pcd files, and one or more folders (hash values) including yaml/xml "
                                "files with matching information)>",
                      ArgvParser::OptionRequiresValue);
     cmd.defineOption("matchPath", "<Use if you want to load one full sequence including 3D data and matches.\n"
                                  "Option sequPath must also be provided.\n"
                                  "Full path to the matches (last folder corresponds to a hash value and must contain "
                                  "yaml/xml files with matching information)>",
                      ArgvParser::OptionRequiresValue);
     cmd.defineOption("gtm_file", "<Use if you only want to load Ground Truth Matches (GTM) of a specific image pair "
                              "from datasets Oxford, KITTI, or MegaDepth.\n"
                              "Path and filename (including file ending) must be provided.>",
                      ArgvParser::OptionRequiresValue);

     /// finally parse and handle return codes (display help etc...)
     int result = -1;
     result = cmd.parse(argc, argv);
     if (result != ArgvParser::NoParserError)
     {
         cout << cmd.parseErrorDescription(result);
         exit(1);
     }
 }

 bool getEnding(const string &path, const string &baseFile, string &ending){
     std::vector<std::string> filenames;
     if(!loadImageSequenceNew(path, baseFile, filenames, true)){
         cerr << "Wrong file structure" << endl;
         return false;
     }
     if(filenames.size() != 1){
         cerr << "Wrong file structure" << endl;
         return false;
     }
     size_t pos1 = filenames[0].rfind(".gz");
     if(pos1 != string::npos){
         pos1 -= 1;
     }
     size_t pos = filenames[0].rfind('.', pos1);
     if(pos == string::npos){
         cerr << "Cannot extract file ending." << endl;
         return false;
     }
     ending = filenames[0].substr(pos);
     return true;
 }

 bool read3Ddata(const string &path, const string &ending, const size_t &nrFrames, vector<data3D> &sequData){
    for(size_t i = 0; i < nrFrames; ++i){
        data3D d3;
        string file = "sequSingleFrameData_" + to_string(i) + ending;
        file = concatPath(path, file);
        if(!checkFileExists(file)){
            cerr << "Missing file " << file << endl;
            return false;
        }
        if(!readCamParsFromDisk(file, d3)){
            return false;
        }
        sequData.emplace_back(move(d3));
    }
    return true;
 }

 bool readMatchesSequence(const string &path, const string &ending, const size_t &nrFrames, vector<sequMatches> &matchData){
     for(size_t i = 0; i < nrFrames; ++i){
         sequMatches sm;
         string file = "matchSingleFrameData_" + to_string(i) + ending;
         file = concatPath(path, file);
         if(!checkFileExists(file)){
             cerr << "Missing file " << file << endl;
             return false;
         }
         if(!readMatchesFromDisk(file, sm)){
             return false;
         }
         matchData.emplace_back(move(sm));
     }
     return true;
 }

bool readMultipleSequenceMatches(const string &path,
                                 const vector<string> &matchFolders,
                                 const string &ending,
                                 const size_t &nrFrames,
                                 vector<vector<sequMatches>> &matchData){
    for(auto &folder: matchFolders){
        vector<sequMatches> matchData1;
        string pathm = concatPath(path, folder);
        if(!checkPathExists(pathm)){
            cerr << "Cannot find path " << pathm << endl;
            return false;
        }
        if(!readMatchesSequence(pathm, ending, nrFrames, matchData1)){
            return false;
        }
        matchData.emplace_back(move(matchData1));
    }
    return true;
}

bool getMult2FrameToFrameMatches(const sequParameters &sequPars,
                                 const std::vector<data3D> &sequData,
                                 const std::vector<std::vector<sequMatches>> &matchData1,
                                 std::vector<std::vector<FrameToFrameMatches>> &f2f_matches){
     for(auto &mm: matchData1){
         std::vector<FrameToFrameMatches> f2f;
         if(getMultFrameToFrameMatches(sequPars, sequData, mm, f2f)){
             f2f_matches.emplace_back(move(f2f));
         }
     }
     return !f2f_matches.empty();
 }

void getMultFull3Ddata(const std::vector<data3D> &sequData, const sequParameters &sequPars, std::vector<std::vector<sequMatches>> &matchData1){
     for(auto& matches: matchData1){
         getFull3Ddata(sequData, sequPars, matches);
     }
 }

 void getMultAllDynamic3Dpoints(std::vector<std::vector<sequMatches>> &matchData1){
     for(auto& matches: matchData1){
         getAllDynamic3Dpoints(matches);
     }
 }

 void getDescriptorDistStat(const cv::Mat &descr1, const cv::Mat &descr2, double &minDist, double &maxDist, double &meanDist, double &medianDist){
     bool useHamming = descr1.type() == CV_8UC1;
     vector<double> dists;
     double sum = 0;
     for (int i = 0; i < descr1.rows; ++i) {
         if(useHamming){
             dists.emplace_back(cv::norm(descr1.row(i), descr2.row(i), cv::NORM_HAMMING));
         }else{
             dists.emplace_back(cv::norm(descr1.row(i), descr2.row(i), cv::NORM_L2));
         }
         sum += dists.back();
     }
     sort(dists.begin(), dists.end());
     const size_t nr_elems = dists.size();
     if(nr_elems % 2){
         medianDist = dists[(nr_elems - 1) / 2];
     }else{
         const size_t nr_elems1 = nr_elems / 2;
         medianDist = (dists[nr_elems1 - 1] + dists[nr_elems1]) / 2.;
     }
     minDist = dists[0];
     maxDist = dists.back();
     meanDist = sum / static_cast<double>(dists.size());
 }

void printDistStat(const double &minDist,
                   const double &maxDist,
                   const double &meanDist,
                   const double &medianDist,
                   const bool &isTP,
                   const size_t &frameNr1,
                   const size_t &frameNr2,
                   const bool &cam1IsFirstStereo = true,
                   const bool &isGTM = false,
                   const string &gtm_id = ""){
    stringstream text;
    int precimi = 1, precima = 1, precimea = 1, precimed = 1;
    auto a = static_cast<int>(round(1e3 * (minDist - floor(minDist))));
    if(a > 0){
        precimi = 4;
    }
    a = static_cast<int>(round(1e3 * (maxDist - floor(maxDist))));
    if(a > 0){
        precima = 4;
    }
    a = static_cast<int>(round(1e3 * (meanDist - floor(meanDist))));
    if(a > 0){
        precimea = 4;
    }
    a = static_cast<int>(round(1e3 * (medianDist - floor(medianDist))));
    if(a > 0){
        precimed = 4;
    }

    if(!isGTM) {
        text << "Descriptor distances of ";
        if (isTP) {
            text << "TP ";
        } else {
            text << "TN ";
        }
    }
    if(isGTM){
        text << "Squared distance of GTM correspondences with ID " << gtm_id << " to original GT position: ";
    }else if(frameNr1 == frameNr2){
        text << "stereo correspondences of frame " << frameNr1 << ": ";
    }else{
        text << "correspondences ";
        if(cam1IsFirstStereo){
            text << "in first stereo camera ";
        }else{
            text << "in second stereo camera ";
        }
        text << "of frames " << frameNr1 << " and " << frameNr2 << ": ";
    }
    text << "Median: " << setprecision(precimed) << medianDist << ", ";
    text << "Mean: " << setprecision(precimea) << meanDist << ", ";
    text << "Min: " << setprecision(precimi) << minDist << ", ";
    text << "Max: " << setprecision(precima) << maxDist;
    cout << text.str() << endl;
 }

void getTNTPDescriptorDistances(const cv::Mat &descr1,
                                const cv::Mat &descr2,
                                const vector<bool> &inlier,
                                const vector<cv::DMatch> &matches,
                                const size_t &frameNr1,
                                const size_t &frameNr2,
                                const bool &cam1IsFirstStereo = true,
                                const bool &isGTM = false,
                                const string &gtm_id = ""){
     Mat d1p, d2p, d1n, d2n;
     d1p.reserve(descr1.rows);
     d2p.reserve(descr1.rows);
     d1n.reserve(descr1.rows);
     d2n.reserve(descr1.rows);
     for(auto &m: matches){
         if(inlier[m.queryIdx]){
             d1p.push_back(descr1.row(m.queryIdx));
             d2p.push_back(descr2.row(m.trainIdx));
         }else{
             d1n.push_back(descr1.row(m.queryIdx));
             d2n.push_back(descr2.row(m.trainIdx));
         }
     }
     double minDist, maxDist, meanDist, medianDist;
     if(!d1p.empty()){
         getDescriptorDistStat(d1p, d2p, minDist, maxDist, meanDist, medianDist);
         printDistStat(minDist, maxDist, meanDist, medianDist, true, frameNr1, frameNr2, cam1IsFirstStereo);
     }
     if(!d1n.empty()){
         getDescriptorDistStat(d1n, d2n, minDist, maxDist, meanDist, medianDist);
         printDistStat(minDist, maxDist, meanDist, medianDist, false, frameNr1, frameNr2, cam1IsFirstStereo);
     }
 }

 void getSquaredDistanceGTM_stat(const gtmData &data){
     vector<double> dists;
     size_t nr_elems = data.matchesGT.size();
     dists.reserve(nr_elems);
     double sum = 0;
     for(auto &m: data.matchesGT){
         dists.emplace_back(static_cast<double>(m.distance));
         sum += dists.back();
     }
     sort(dists.begin(), dists.end());
     double medianDist, minDist, maxDist, meanDist;
     if(nr_elems % 2){
         medianDist = dists[(nr_elems - 1) / 2];
     }else{
         const size_t nr_elems1 = nr_elems / 2;
         medianDist = (dists[nr_elems1 - 1] + dists[nr_elems1]) / 2.;
     }
     minDist = dists[0];
     maxDist = dists.back();
     meanDist = sum / static_cast<double>(dists.size());
     printDistStat(minDist, maxDist, meanDist, medianDist, false, 0, 0, false, true, data.quality.id);
 }

 int main(int argc, char* argv[])
 {
     ArgvParser cmd;
     SetupCommandlineParser(cmd, argc, argv);

     if(cmd.foundOption("file") || cmd.foundOption("fileRt")){
         if(!(cmd.foundOption("file") && cmd.foundOption("fileRt"))){
             cerr << "For loading single stereo frame matches, both options file and fileRt must be provided." << endl;
             return EXIT_FAILURE;
         }
         std::string filename = cmd.optionValue("file");
         std::string filenameRt = cmd.optionValue("fileRt");
         if(!checkFileExists(filename)){
             cerr << "File " << filename << " not found." << endl;
             return EXIT_FAILURE;
         }
         if(!checkFileExists(filenameRt)){
             cerr << "File " << filenameRt << " not found." << endl;
             return EXIT_FAILURE;
         }
         sequMatches sm;
         data3D d3;
         //Get correspondence data of a single stereo frame
         if(!readMatchesFromDisk(filename, sm)){
             cerr << "Unable to load matches." << endl;
             return EXIT_FAILURE;
         }else{
             cout << "Loading matches successful" << endl;
         }
         //Get 3D data of a single stereo frame
         if(!readCamParsFromDisk(filenameRt, d3)){
             cerr << "Unable to load 3D data." << endl;
             return EXIT_FAILURE;
         }else{
             cout << "Loading 3D data successful" << endl;
         }
         getTNTPDescriptorDistances(sm.frameDescriptors1,
                                    sm.frameDescriptors2,
                                    sm.frameInliers,
                                    sm.frameMatches,
                                    d3.actFrameCnt,
                                    d3.actFrameCnt);
     }else if(cmd.foundOption("sequPath")){
         string sequPath = cmd.optionValue("sequPath");
         if(!checkPathExists(sequPath)){
             cerr << "Path " << sequPath << " not found." << endl;
             return EXIT_FAILURE;
         }
         //Get file extension (yaml/xml) for data files
         string data_ending;
         string sequParsF = "sequPars";
         if(!getEnding(sequPath, sequParsF, data_ending)){
             return EXIT_FAILURE;
         }
         //Get parameters used for generating 3D data
         sequParameters sequPars;
         sequParsF += data_ending;
         sequParsF = concatPath(sequPath, sequParsF);
         if(!readSequenceParameters(sequParsF, sequPars)){
             cerr << "Failed to load sequence parameters." << endl;
             return EXIT_FAILURE;
         }
         //Get 3D data
         vector<data3D> sequData;
         if(!read3Ddata(sequPath, data_ending, sequPars.totalNrFrames, sequData)){
             return EXIT_FAILURE;
         }
         //Get point clouds of static and dynamic objects
         PointClouds ptCl;
         if(!readPointClouds(sequPath, sequPars.nrMovObjAllFrames, ptCl)){
             cerr << "Necessary 3D PCL point cloud files could not be loaded!" << endl;
             return EXIT_FAILURE;
         }
         if(cmd.foundOption("matchPath")){
             string matchPath = cmd.optionValue("matchPath");
             if(!checkPathExists(matchPath)){
                 cerr << "Path " << matchPath << " not found." << endl;
                 return EXIT_FAILURE;
             }
             //Get correspondence data
             vector<sequMatches> matchData1;
             if(!readMatchesSequence(matchPath, data_ending, sequPars.totalNrFrames, matchData1)){
                 return EXIT_FAILURE;
             }
             for(size_t i = 0; i < matchData1.size(); ++i){
                 getTNTPDescriptorDistances(matchData1[i].frameDescriptors1,
                                            matchData1[i].frameDescriptors2,
                                            matchData1[i].frameInliers,
                                            matchData1[i].frameMatches,
                                            sequData[i].actFrameCnt,
                                            sequData[i].actFrameCnt);
             }
             //Get 3D data for every stereo pair
             getFull3Ddata(sequData, sequPars, matchData1);
             //Get dynamic elements
             getAllDynamic3Dpoints(matchData1);
             //Get frame to frame matches
             vector<FrameToFrameMatches> f2f_matches;
             getMultFrameToFrameMatches(sequPars, sequData, matchData1, f2f_matches);
             for(auto & f2f_matche : f2f_matches){
                 getTNTPDescriptorDistances(f2f_matche.descriptors_1_1,
                                            f2f_matche.descriptors_2_1,
                                            f2f_matche.inlier_mask_1,
                                            f2f_matche.matches_1,
                                            f2f_matche.frameNr,
                                            f2f_matche.frameNr + 1);
             }
             for(auto & f2f_matche : f2f_matches){
                 getTNTPDescriptorDistances(f2f_matche.descriptors_1_2,
                                            f2f_matche.descriptors_2_2,
                                            f2f_matche.inlier_mask_2,
                                            f2f_matche.matches_2,
                                            f2f_matche.frameNr,
                                            f2f_matche.frameNr + 1,
                                            false);
             }
         }else{
             //Get file extension (yaml/xml) for parameter files
             string base_ending;
             string matchInfoFName = "matchInfos";
             if(!getEnding(sequPath, matchInfoFName, base_ending)){
                 return EXIT_FAILURE;
             }
             //Get parameters used for generating correspondences
             matchInfoFName += base_ending;
             string matchParsFileName = concatPath(sequPath, matchInfoFName);
             std::vector<matchSequParameters> matchPars;
             if(!readMultipleMatchSequencePars(matchParsFileName, matchPars)){
                 return EXIT_FAILURE;
             }
             //Get folder names holding correspondence data
             vector<string> matchFolders;
             for(auto& pars: matchPars){
                 matchFolders.push_back(pars.hashMatchingPars);
             }
             //Get multiple correspondence data (different correspondence data, equal 3D data)
             vector<vector<sequMatches>> matchData;
             if(!readMultipleSequenceMatches(sequPath,
                                             matchFolders,
                                             data_ending,
                                             sequPars.totalNrFrames,
                                             matchData)){
                 return EXIT_FAILURE;
             }
             for(auto &ms: matchData) {
                 for (size_t i = 0; i < ms.size(); ++i) {
                     getTNTPDescriptorDistances(ms[i].frameDescriptors1,
                                                ms[i].frameDescriptors2,
                                                ms[i].frameInliers,
                                                ms[i].frameMatches,
                                                sequData[i].actFrameCnt,
                                                sequData[i].actFrameCnt);
                 }
             }
             //Get 3D data for every stereo pair
             getMultFull3Ddata(sequData, sequPars, matchData);
             //Get dynamic elements
             getMultAllDynamic3Dpoints(matchData);
             //Get frame to frame matches
             std::vector<std::vector<FrameToFrameMatches>> f2f_matches;
             getMult2FrameToFrameMatches(sequPars, sequData, matchData, f2f_matches);
             for(auto &ffm: f2f_matches){
                 for(auto & f2f_matche : ffm){
                     getTNTPDescriptorDistances(f2f_matche.descriptors_1_1,
                                                f2f_matche.descriptors_2_1,
                                                f2f_matche.inlier_mask_1,
                                                f2f_matche.matches_1,
                                                f2f_matche.frameNr,
                                                f2f_matche.frameNr + 1);
                 }
                 for(auto & f2f_matche : ffm){
                     getTNTPDescriptorDistances(f2f_matche.descriptors_1_2,
                                                f2f_matche.descriptors_2_2,
                                                f2f_matche.inlier_mask_2,
                                                f2f_matche.matches_2,
                                                f2f_matche.frameNr,
                                                f2f_matche.frameNr + 1,
                                                false);
                 }
             }
         }
     } else if(cmd.foundOption("gtm_file")){
         string gtm_file = cmd.optionValue("gtm_file");
         if(!checkFileExists(gtm_file)){
             cerr << "File " << gtm_file << " not found." << endl;
             return EXIT_FAILURE;
         }
         gtmData data;
         if(readGTMatchesDisk(gtm_file, data) != 0){
             cerr << "Failed to read GTM data: " << gtm_file << endl;
             return EXIT_FAILURE;
         }
         getSquaredDistanceGTM_stat(data);
     }
     else{
         cerr << "Options required. Use -h or --help for addition information." << endl;
         return EXIT_FAILURE;
     }
     return EXIT_SUCCESS;
 }
