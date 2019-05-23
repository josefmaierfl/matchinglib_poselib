#include "argvparser.h"

#include "io_data.h"

#include "getStereoCameraExtr.h"
#include "generateSequence.h"
#include "generateMatches.h"
#include "helper_funcs.h"
#include <time.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;
using namespace CommandLineProcessing;

struct specificStereoPars{
    cv::Mat R;
    cv::Mat t;
    cv::Mat K1;
    cv::Mat K2;

    specificStereoPars(){
        R = cv::Mat::eye(3,3,CV_64FC1);
        t = cv::Mat::zeros(3,1,CV_64FC1);
        K1 = cv::Mat::zeros(3,3,CV_64FC1);
        K2 = cv::Mat::zeros(3,3,CV_64FC1);
    }
};
struct stereoExtrPars{
    int nrStereoConfigs;
    int txChangeFRate;
    int tyChangeFRate;
    int tzChangeFRate;
    int rollChangeFRate;
    int pitchChangeFRate;
    int yawChangeFRate;
    double txLinChangeVal;
    double tyLinChangeVal;
    double tzLinChangeVal;
    double rollLinChangeVal;
    double pitchLinChangeVal;
    double yawLinChangeVal;
    double txStartVal;
    double tyStartVal;
    double tzStartVal;
    double rollStartVal;
    double pitchStartVal;
    double yawStartVal;
    std::pair<double,double> txRange;
    std::pair<double,double> tyRange;
    std::pair<double,double> tzRange;
    std::pair<double,double> rollRange;
    std::pair<double,double> pitchRange;
    std::pair<double,double> yawRange;
    bool txVariable;
    bool tyVariable;
    bool tzVariable;
    bool rollVariable;
    bool pitchVariable;
    bool yawVariable;
    bool useSpecificCamPars;
    std::vector<specificStereoPars> specialCamPars;
    double imageOverlap;
    cv::Size imgSize;

    stereoExtrPars(){
        nrStereoConfigs = 0;
        txChangeFRate = 0;
        tyChangeFRate = 0;
        tzChangeFRate = 0;
        rollChangeFRate = 0;
        pitchChangeFRate = 0;
        yawChangeFRate = 0;
        txLinChangeVal = 0;
        tyLinChangeVal = 0;
        tzLinChangeVal = 0;
        rollLinChangeVal = 0;
        pitchLinChangeVal = 0;
        yawLinChangeVal = 0;
        txStartVal = 0;
        tyStartVal = 0;
        tzStartVal = 0;
        rollStartVal = 0;
        pitchStartVal = 0;
        yawStartVal = 0;
        txRange = std::make_pair(0,0);
        tyRange = std::make_pair(0,0);
        tzRange = std::make_pair(0,0);
        rollRange = std::make_pair(0,0);
        pitchRange = std::make_pair(0,0);
        yawRange = std::make_pair(0,0);
        txVariable = false;
        tyVariable = false;
        tzVariable = false;
        rollVariable = false;
        pitchVariable = false;
        yawVariable = false;
        useSpecificCamPars = false;
        specialCamPars.clear();
        imageOverlap = 0;
        imgSize = cv::Size(0,0);
    }

    bool checkParameters() const{
        if((nrStereoConfigs <= 0) || (nrStereoConfigs > 1000)){
            cerr << "Invalid number of stereo configurations." << endl;
            return false;
        }
        if((txRange.first > txRange.second) && !nearZero(txRange.first - txRange.second)){
            cerr << "stereo tx value range is invalid." << endl;
            return false;
        }
        if((tyRange.first > tyRange.second) && !nearZero(tyRange.first - tyRange.second)){
            cerr << "stereo ty value range is invalid." << endl;
            return false;
        }
        if((tzRange.first > tzRange.second) && !nearZero(tzRange.first - tzRange.second)){
            cerr << "stereo tz value range is invalid." << endl;
            return false;
        }
        if((rollRange.first > rollRange.second) && !nearZero(rollRange.first - rollRange.second)){
            cerr << "stereo roll value range is invalid." << endl;
            return false;
        }
        if((pitchRange.first > pitchRange.second) && !nearZero(pitchRange.first - pitchRange.second)){
            cerr << "stereo pitch value range is invalid." << endl;
            return false;
        }
        if((yawRange.first > yawRange.second) && !nearZero(yawRange.first - yawRange.second)){
            cerr << "stereo yaw value range is invalid." << endl;
            return false;
        }
        double maxXY = max(max(abs(txRange.first), abs(txRange.second)),
                           max(abs(tyRange.first), abs(tyRange.second)));
        if(max(abs(tzRange.first), abs(tzRange.second)) > maxXY){
            cerr << "Max. absolute value of the range of tz (stereo) must be smaller than the max. absolute "
                    "value of tx and tz" << endl;
            return false;
        }
        maxXY = max(txStartVal, tyStartVal);
        if(tzStartVal > maxXY){
            cerr << "Start value for tz must be smaller than start values for tx and ty." << endl;
            return false;
        }
        if((imageOverlap < 0.1) || (imageOverlap > 1.0)){
            cerr << "Specified image overlap is out of range." << endl;
            return false;
        }
        return true;
    }
};
struct ellipsoidTrackPars{
    int xDirection;
    double xzExpansion;
    double xyExpansion;
    std::pair<double,double> thetaRange;

    ellipsoidTrackPars(){
        xDirection = 0;
        xzExpansion = 0;
        xyExpansion = 0;
        thetaRange = make_pair(0,0);
    }

    bool checkParameters() const{
        if((xDirection != -1) && (xDirection != 1)){
            cerr << "Parameter xDirection of ellipsoidal track can only be 1 or -1." << endl;
            return false;
        }
        if((xzExpansion < -100.0) || (xzExpansion > 100.0)){
            cerr << "Parameter xzExpansion of ellipsoidal track can only be between -100.0 and +100.0." << endl;
            return false;
        }
        if((xyExpansion < -100.0) || (xyExpansion > 100.0)){
            cerr << "Parameter xyExpansion of ellipsoidal track can only be between -100.0 and +100.0." << endl;
            return false;
        }
        if((thetaRange.first < -M_PI_2)
           || (thetaRange.second > M_PI_2)
           || ((thetaRange.first > thetaRange.second) && !nearZero(thetaRange.first - thetaRange.second))){
            cerr << "Parameter thetaRange of ellipsoidal track can only be between -PI/2 and +PI/2."
                    "The first parameter of the range must be smaller or equal the second." << endl;
            return false;
        }
        return true;
    }
};
struct randomTrackPars{
    int xDirection;
    std::pair<double,double> xzDirectionRange;
    std::pair<double,double> xyDirectionRange;
    double allowedChangeSD;

    randomTrackPars(){
        xDirection = 0;
        xzDirectionRange = make_pair(0,0);
        xyDirectionRange = make_pair(0,0);
        allowedChangeSD = 0;
    }

    bool checkParameters() const{
        if((xDirection != -1) && (xDirection != 1)){
            cerr << "Parameter xDirection of random track can only be 1 or -1." << endl;
            return false;
        }
        if((xzDirectionRange.first < -1000.0)
           || (xzDirectionRange.second > 1000.0)
           || ((xzDirectionRange.first > xzDirectionRange.second)
               && !nearZero(xzDirectionRange.first - xzDirectionRange.second))){
            cerr << "Parameter xzDirectionRange of random track can only be between -PI/2 and +PI/2."
                    "The first parameter of the range must be smaller or equal the second." << endl;
            return false;
        }
        if((xyDirectionRange.first < -1000.0)
           || (xyDirectionRange.second > 1000.0)
           || ((xyDirectionRange.first > xyDirectionRange.second)
               && !nearZero(xyDirectionRange.first - xyDirectionRange.second))){
            cerr << "Parameter xzDirectionRange of random track can only be between -PI/2 and +PI/2."
                    "The first parameter of the range must be smaller or equal the second." << endl;
            return false;
        }
        if((allowedChangeSD < 0.05) || (allowedChangeSD > 1.0)){
            cerr << "Parameter allowedChangeSD of random track can only be between 0.05 and 1.0." << endl;
            return false;
        }
        return true;
    }
};
struct additionalSequencePars{
    bool corrsPerRegionRandInit;
    int trackOption;
    ellipsoidTrackPars ellipsoidTrack;
    randomTrackPars randomTrack;
    int maxTrackElements;
    double rollCamTrack;
    double pitchCamTrack;
    double yawCamTrack;
    bool filterOccluded3D;
    uint32_t verbose;
    int LMverbose;
    bool acceptBadStereoPars;

    additionalSequencePars(){
        corrsPerRegionRandInit = false;
        trackOption = 0;
        ellipsoidTrack = ellipsoidTrackPars();
        randomTrack = randomTrackPars();
        maxTrackElements = 0;
        rollCamTrack = 0;
        pitchCamTrack = 0;
        yawCamTrack = 0;
        filterOccluded3D = false;
        verbose = 0;
        LMverbose = 0;
        acceptBadStereoPars = false;
    }

    bool checkParameters() const{
        if((trackOption < 1) || (trackOption > 3)){
            cerr << "Parameter trackOption is invalid." << endl;
            return false;
        }
        if(!ellipsoidTrack.checkParameters()){
            cerr << "Invalid parameters for ellipsoidal track." << endl;
            return false;
        }
        if(!randomTrack.checkParameters()){
            cerr << "Invalid parameters for random track." << endl;
            return false;
        }
        if((maxTrackElements <= 0) || (maxTrackElements > 10000)){
            cerr << "Parameter maxTrackElements is invalid." << endl;
            return false;
        }
        if(LMverbose > 100){
            cerr << "LMverbose out of range." << endl;
            return false;
        }
        return true;
    }
};

double getRandDoubleVal(std::default_random_engine rand_generator, double lowerBound, double upperBound);
int startEvaluation(ArgvParser& cmd);
bool genTemplateFile(const std::string &filename);
bool loadConfigFile(const std::string &filename,
                    StereoSequParameters &sequPars,
                    GenMatchSequParameters &matchPars,
                    stereoExtrPars &stereoPars,
                    additionalSequencePars &addPars);
bool checkConfigFileName(string &filename, const string &errMsgPart);
bool genStereoConfigurations(const int nrFrames,
                             const stereoExtrPars &stereoPars,
                             const additionalSequencePars &addPars,
                             std::vector<cv::Mat>& Rv,
                             std::vector<cv::Mat>& tv,
                             cv::Mat& K_1,
                             cv::Mat& K_2);
bool genSequenceConfig(const additionalSequencePars &addPars,
                       const stereoExtrPars &stereoPars,
                       StereoSequParameters &sequPars);

void SetupCommandlineParser(ArgvParser& cmd, int argc, char* argv[])
{
	cmd.setIntroductoryDescription("Test interface for generating randomized 3D scenes and matches");
	//define error codes
	cmd.addErrorCode(0, "Success");
	cmd.addErrorCode(1, "Error");

	cmd.setHelpOption("h", "help","Generation of 3D scenes and matches.");
	cmd.defineOption("img_path", "<Path to the images where the features should be extracted from (all required in one folder)>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("img_pref", "<Prefix and/or postfix for the used images.\n "
								 "It can include a folder structure that follows after the filepath, a file prefix, a '*' indicating the position of the number and a postfix. "
								 "If it is empty, all files from the folder img_path are used (also if img_pref only contains a folder ending with '/', every file within this folder is used). "
								 "It is possible to specify only a prefix with or without '*' at the end. "
								 "If a prefix is used, all characters until the first number (excluding) must be provided. "
								 "For a postfix, '*' must be placed before the postfix.\n "
								 "Valid examples : folder/pre_*post, *post, pre_*, pre_, folder/*post, folder/pre_*, folder/pre_, folder/, folder/folder/, folder/folder/pre_*post, ...>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("store_path", "<Path for storing the generated 3D scenes and matches. "
                                "If load_folder and load_type are given, store_path can be set to store_path=* "
                                "which indicates that the generated matches should be stored into the load_folder.>", ArgvParser::OptionRequiresValue);
	cmd.defineOption("load_folder", "<Path for loading an existing 3D scene for generating new matches. load_type must also be specified.>", ArgvParser::OptionRequiresValue);
	/*cmd.defineOption("load_type", "<File type of the stored 3D scene that must be provided if it is loaded. Options: "
					 "\n 0\t YAML without compression"
					 "\n 1\t YAML with compression (.yaml.gz)"
					 "\n 2\t XML without compression"
					 "\n 3\t XML with compression (.xml.gz)",
			ArgvParser::OptionRequiresValue);*/
    cmd.defineOption("conf_file", "<Path and filename (.yaml/.xml) for loading parameters for new 3D scenes and matches that should be generated.>", ArgvParser::OptionRequiresValue);
    cmd.defineOption("genConfTempl", "<Path and filename (.yaml/.xml) for generating a template file of parameters for new 3D scenes and matches that should be generated.>", ArgvParser::OptionRequiresValue);

	
	/// finally parse and handle return codes (display help etc...)
	int result = -1;
	result = cmd.parse(argc, argv);
	if (result != ArgvParser::NoParserError)
	{
		cout << cmd.parseErrorDescription(result);
		exit(1);
	}
}

int startEvaluation(ArgvParser& cmd)
{
	bool option_found = false;
	option_found = cmd.foundOption("img_path")
	        & cmd.foundOption("img_pref")
	        & cmd.foundOption("store_path")
	        & cmd.foundOption("conf_file");
	if(!option_found && !cmd.foundOption("genConfTempl")){
		cout << "Did not find necessary options for generating 3D scene and matches." << endl;
		return -1;
	}
	else if(cmd.foundOption("genConfTempl")){
	    string genConfTempl = cmd.optionValue("genConfTempl");
        if(!checkConfigFileName(genConfTempl, "generating a template")){
            return -1;
        }
        if(checkFileExists(genConfTempl)){
            cout << "Config file already exists. Do you want to replace it?(y/n)";
            string uip;
            cin >> uip;
            while ((uip != "y") && (uip != "n")) {
                cout << endl << "Try again:";
                cin >> uip;
            }
            cout << endl;
            if (uip == "n") {
                return -1;
            }else{
                if(!deleteFile(genConfTempl)){
                    cerr << "Unable to delete old config file." << endl;
                    return -1;
                }
            }
        }
        if(!genTemplateFile(genConfTempl)){
            cerr << "Unable to generate a template file. Check write permissions." << endl;
            return -1;
        }
	}
	else{
		string img_path, store_path, img_pref, load_folder, /*load_type,*/ conf_file;
        conf_file = cmd.optionValue("conf_file");
		img_path = cmd.optionValue("img_path");
		img_pref = cmd.optionValue("img_pref");
		store_path = cmd.optionValue("store_path");
        StereoSequParameters sequPars;
        GenMatchSequParameters matchPars;
        stereoExtrPars stereoPars;
        additionalSequencePars addPars;
        if(!checkConfigFileName(conf_file, "loading a configuration")){
            return -1;
        }
        if(!checkFileExists(conf_file)){
            cerr << "Configuration file for generating 3D scene and matches not found." << endl;
            return -1;
        }
        if(!loadConfigFile(conf_file, sequPars, matchPars, stereoPars, addPars)){
            cerr << "Unable to load configuration file." << endl;
            return -1;
        }
		if(cmd.foundOption("load_folder")){
			/*if(!cmd.foundOption("load_type")){
				cerr << "Unable to load 3D scene as the file type is missing (parameter load_type)." << endl;
				return -1;
			}*/
			load_folder = cmd.optionValue("load_folder");
			/*load_type = cmd.optionValue("load_type");
			int load_type_val = stoi(load_type);
			bool rwXMLinfo = false;
			bool compressedWrittenInfo = false;
			switch(load_type_val){
				case 0:
					rwXMLinfo = false;
					compressedWrittenInfo = false;
					break;
				case 1:
					rwXMLinfo = false;
					compressedWrittenInfo = true;
					break;
				case 2:
					rwXMLinfo = true;
					compressedWrittenInfo = false;
					break;
				case 3:
					rwXMLinfo = true;
					compressedWrittenInfo = true;
					break;
				default:
					cerr << "Wrong parameter value for the file type." << endl;
					return -1;
			}*/
			if(store_path == "*"){
                store_path.clear();//Indicates that the generated matches should be stored into the load path
			}
            matchPars.mainStorePath = store_path;
            matchPars.imgPath = img_path;
            matchPars.imgPrePostFix = img_pref;
            matchPars.parsValid = true;
            /*matchPars.rwXMLinfo = rwXMLinfo;
            matchPars.compressedWrittenInfo = compressedWrittenInfo;*/
            if(!matchPars.checkParameters()){
                return -1;
            }
            genMatchSequ sequ(load_folder,
                              matchPars,
                              addPars.verbose);
            if(!sequ.generateMatches()){
                cerr << "Unable to generate matches." << endl;
                return -1;
            }
		}else {
            std::vector<cv::Mat> Rv, tv;
            cv::Mat K_1, K_2;
		    if(!genStereoConfigurations((int)sequPars.nTotalNrFrames,
                stereoPars,
                addPars,
                Rv,
                tv,
                K_1,
                K_2)){
		        cerr << "Unable to calculate extrinsics." << endl;
		        return -1;
		    }
		    if(!genSequenceConfig(addPars, stereoPars, sequPars)){
                cerr << "Unable to generate a sequence." << endl;
                return -1;
		    }
            matchPars.mainStorePath = store_path;
            matchPars.imgPath = img_path;
            matchPars.imgPrePostFix = img_pref;
            matchPars.parsValid = true;
		    if(!matchPars.checkParameters()){
                cerr << "Paramters for calculating matches are invalid." << endl;
                return -1;
		    }
		    try {
                genMatchSequ sequ(stereoPars.imgSize,
                                  K_1,
                                  K_2,
                                  Rv,
                                  tv,
                                  sequPars,
                                  matchPars,
                                  addPars.filterOccluded3D,
                                  addPars.verbose);
                if(!sequ.generateMatches()){
                    cerr << "Failed to calculate matches!" << endl;
                    return -1;
                }
            }catch(exception &e){
		        cerr << "Exception: " << e.what() << endl;
		        return -1;
		    }catch(...){
		        cerr << "Unknown exception." << endl;
		        return -1;
		    }
		}
	}

	return 0;
}

bool genSequenceConfig(const additionalSequencePars &addPars,
                       const stereoExtrPars &stereoPars,
                       StereoSequParameters &sequPars){
    if(!addPars.checkParameters()){
        return false;
    }

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rand_generator(seed);
    std::mt19937 rand2(seed);

    if(addPars.corrsPerRegionRandInit){
        if(sequPars.corrsPerRegRepRate == 0){
            sequPars.corrsPerRegion.clear();
        }else{
            size_t corrsPerRegionSi = sequPars.nTotalNrFrames / sequPars.corrsPerRegRepRate;
            sequPars.corrsPerRegion.clear();
            for (size_t i = 0; i < corrsPerRegionSi; i++) {
                cv::Mat corrsPReg(3, 3, CV_64FC1);
                cv::randu(corrsPReg, Scalar(0), Scalar(1.0));
                corrsPReg /= sum(corrsPReg)[0];
                sequPars.corrsPerRegion.emplace_back(corrsPReg.clone());
            }
        }
    }

    if(addPars.trackOption == 1) {
        //Ellipsoid
        double b, a = (double) addPars.ellipsoidTrack.xDirection, c;
        b = a * addPars.ellipsoidTrack.xzExpansion;
        c = a * addPars.ellipsoidTrack.xyExpansion;
        double theta = addPars.ellipsoidTrack.thetaRange.first;
        double thetaDiff = addPars.ellipsoidTrack.thetaRange.second - theta;
        int midElems = addPars.maxTrackElements / 2;
        //Variation in height from one track element to the next based on the angle theta.
        double thetaPiece = (2.0 * thetaDiff)
                            / (double) max(addPars.maxTrackElements - 1, 1);
        //The track is nearly a loop closure (phi from 0 to nearly 2 pi)
        double phiPiece = (2.0 * M_PI - M_PI / (double) (10 * addPars.maxTrackElements))
                          / (double) max(addPars.maxTrackElements - 1, 1);
        sequPars.camTrack.clear();
        bool enableFlightMode = !nearZero(addPars.ellipsoidTrack.thetaRange.first
                                         - addPars.ellipsoidTrack.thetaRange.second);
        double phi = 0;
        for (int i = 0; i < addPars.maxTrackElements; i++) {
            Mat tr1 = Mat(3, 1, CV_64FC1);
            tr1.at<double>(0) = a * cos(theta) * cos(phi);
            tr1.at<double>(2) = b * cos(theta) * sin(phi);
            tr1.at<double>(1) = c * sin(theta);
            phi += phiPiece;
            if (enableFlightMode) {
                if(i < midElems) {
                    theta += thetaPiece;
                }else{
                    theta -= thetaPiece;
                }
            }
            sequPars.camTrack.emplace_back(tr1.clone());
        }
    }
    else if(addPars.trackOption == 3){
        //Random track
        double xy = 0, xz = 0;
        Mat tr1 = Mat(3, 1, CV_64FC1);
        //Generate a random camera movement vector based on the ranges given. The scaling is not important. The scaling and track segments for every image pair are caluclated within the class genStereoSequ
        tr1.at<double>(0) = (double)addPars.randomTrack.xDirection;
        xy = getRandDoubleVal(rand_generator,
                              addPars.randomTrack.xyDirectionRange.first,
                              addPars.randomTrack.xyDirectionRange.second);
        xz = getRandDoubleVal(rand_generator,
                              addPars.randomTrack.xzDirectionRange.first,
                              addPars.randomTrack.xzDirectionRange.second);
        tr1.at<double>(1) = tr1.at<double>(0) * xy;
        tr1.at<double>(2) = tr1.at<double>(0) * xz;
        sequPars.camTrack.clear();
        sequPars.camTrack.emplace_back(tr1.clone());
        std::normal_distribution<double> distributionNX2(1.0, abs(addPars.randomTrack.allowedChangeSD));
        for (int i = 0; i < addPars.maxTrackElements; i++) {
            xy = getRandDoubleVal(rand_generator,
                                  addPars.randomTrack.xyDirectionRange.first,
                                  addPars.randomTrack.xyDirectionRange.second);
            xz = getRandDoubleVal(rand_generator,
                                  addPars.randomTrack.xzDirectionRange.first,
                                  addPars.randomTrack.xzDirectionRange.second);
            //New x-direction depends on old x-direction
            tr1.at<double>(0) *= distributionNX2(rand_generator);
            //New y-direction mainly depends on old y-direction but also on a random part which depends on the x-direction
            tr1.at<double>(1) =
                    (2.0 * (tr1.at<double>(1) * distributionNX2(rand_generator)) + tr1.at<double>(0) * xy) /
                    3.0;
            //New z-direction mainly depends on old z-direction but also on a random part which depends on the x-direction
            tr1.at<double>(2) =
                    (2.0 * (tr1.at<double>(2) * distributionNX2(rand_generator)) + tr1.at<double>(0) * xz) /
                    3.0;
            sequPars.camTrack.push_back(sequPars.camTrack.back() + tr1);
        }
    }

    sequPars.R = eulerAnglesToRotationMatrix(addPars.rollCamTrack * M_PI / 180.0,
                                             addPars.pitchCamTrack * M_PI / 180.0,
                                             addPars.yawCamTrack * M_PI / 180.0);

    int minStUpDateFrequ_tmp[4] = {0,0,0,0};
    minStUpDateFrequ_tmp[0] = min(stereoPars.txChangeFRate, stereoPars.tyChangeFRate);
    if(minStUpDateFrequ_tmp[0] == 0)
        minStUpDateFrequ_tmp[0] = max(stereoPars.txChangeFRate, stereoPars.tyChangeFRate);
    minStUpDateFrequ_tmp[1] = min(stereoPars.tzChangeFRate, stereoPars.rollChangeFRate);
    if(minStUpDateFrequ_tmp[1] == 0)
        minStUpDateFrequ_tmp[1] = max(stereoPars.tzChangeFRate, stereoPars.rollChangeFRate);
    minStUpDateFrequ_tmp[2] = min(stereoPars.pitchChangeFRate, stereoPars.yawChangeFRate);
    if(minStUpDateFrequ_tmp[2] == 0)
        minStUpDateFrequ_tmp[2] = max(stereoPars.pitchChangeFRate, stereoPars.yawChangeFRate);
    minStUpDateFrequ_tmp[3] = min(minStUpDateFrequ_tmp[0], minStUpDateFrequ_tmp[1]);
    if(minStUpDateFrequ_tmp[3] == 0)
        minStUpDateFrequ_tmp[3] = max(minStUpDateFrequ_tmp[0], minStUpDateFrequ_tmp[1]);
    int minStUpDateFrequ = min(minStUpDateFrequ_tmp[2], minStUpDateFrequ_tmp[3]);
    if(minStUpDateFrequ == 0)
        minStUpDateFrequ = max(minStUpDateFrequ_tmp[2], minStUpDateFrequ_tmp[3]);
    sequPars.nFramesPerCamConf = (size_t)minStUpDateFrequ;
    if(sequPars.nFramesPerCamConf == 0)
        sequPars.nFramesPerCamConf = sequPars.nTotalNrFrames;

    if(!sequPars.checkParameters()){
        cerr << "Parameters for generating a 3D sequence are invalid." << endl;
        return false;
    }

    return true;
}

bool genStereoConfigurations(const int nrFrames,
                             const stereoExtrPars &stereoPars,
                             const additionalSequencePars &addPars,
                             std::vector<cv::Mat>& Rv,
                             std::vector<cv::Mat>& tv,
                             cv::Mat& K_1,
                             cv::Mat& K_2){
    if(!addPars.checkParameters()){
        return false;
    }
    Rv.clear();
    tv.clear();
    if(stereoPars.useSpecificCamPars){
        if(stereoPars.specialCamPars.empty()){
            cerr << "Flag useSpecificCamPars is set, but no specific stereo configurations are provided." << endl;
            return false;
        }
        stereoPars.specialCamPars[0].K1.copyTo(K_1);
        stereoPars.specialCamPars[0].K2.copyTo(K_2);
        Rv.reserve(stereoPars.specialCamPars.size());
        tv.reserve(stereoPars.specialCamPars.size());
        for(auto &i : stereoPars.specialCamPars){
            Rv.push_back(i.R);
            if((Rv.back().rows != Rv.back().cols) || (Rv.back().rows != 3) || (Rv.back().type() != CV_64FC1)){
                cerr << "Wrong format of given rotation matrices (stereo configurations)." << endl;
                return false;
            }
            if(!isMatRotationMat(Rv.back())){
                cerr << "One of the given rotations of the stereo configurations is not a rotation matrix." << endl;
                return false;
            }
            tv.push_back(i.t);
            if((tv.back().rows != 3) || (tv.back().cols != 1) || (tv.back().type() != CV_64FC1)){
                cerr << "Wrong format of given translation vectors (stereo configurations)." << endl;
                return false;
            }
            if(abs(tv.back().at<double>(2)) >= max(abs(tv.back().at<double>(0)), abs(tv.back().at<double>(1)))){
                cerr << "z-component of stereo configurations must be smaller than the largest "
                        "x- or y-vector-component of a single configuration." << endl;
                return false;
            }
            if((int)tv.size() >= stereoPars.nrStereoConfigs){
                break;
            }
        }
        if((int)tv.size() < stereoPars.nrStereoConfigs){
            cerr << "Number of desired stereo configurations does not match the number of provided R&t entries."
            << endl;
            return false;
        }
        if(stereoPars.nrStereoConfigs > nrFrames){
            cerr << "Too many stereo configurations or too less frames." << endl;
            return false;
        }
    }else{
        if(!stereoPars.checkParameters()){
            return false;
        }
        if(stereoPars.nrStereoConfigs > nrFrames){
            cerr << "Too many stereo configurations or too less frames." << endl;
            return false;
        }
        int minStUpDateFrequ_tmp[4] = {0,0,0,0};
        minStUpDateFrequ_tmp[0] = min(stereoPars.txChangeFRate, stereoPars.tyChangeFRate);
        if(minStUpDateFrequ_tmp[0] == 0)
            minStUpDateFrequ_tmp[0] = max(stereoPars.txChangeFRate, stereoPars.tyChangeFRate);
        minStUpDateFrequ_tmp[1] = min(stereoPars.tzChangeFRate, stereoPars.rollChangeFRate);
        if(minStUpDateFrequ_tmp[1] == 0)
            minStUpDateFrequ_tmp[1] = max(stereoPars.tzChangeFRate, stereoPars.rollChangeFRate);
        minStUpDateFrequ_tmp[2] = min(stereoPars.pitchChangeFRate, stereoPars.yawChangeFRate);
        if(minStUpDateFrequ_tmp[2] == 0)
            minStUpDateFrequ_tmp[2] = max(stereoPars.pitchChangeFRate, stereoPars.yawChangeFRate);
        minStUpDateFrequ_tmp[3] = min(minStUpDateFrequ_tmp[0], minStUpDateFrequ_tmp[1]);
        if(minStUpDateFrequ_tmp[3] == 0)
            minStUpDateFrequ_tmp[3] = max(minStUpDateFrequ_tmp[0], minStUpDateFrequ_tmp[1]);
        int minStUpDateFrequ = min(minStUpDateFrequ_tmp[2], minStUpDateFrequ_tmp[3]);
        if(minStUpDateFrequ == 0)
            minStUpDateFrequ = max(minStUpDateFrequ_tmp[2], minStUpDateFrequ_tmp[3]);
        if((minStUpDateFrequ == 0) && (stereoPars.nrStereoConfigs > 1)){
            cerr << "Linear changes of stereo extrinsics are not allowed which leads to only 1 configuration "
                    "but a number of " << stereoPars.nrStereoConfigs << " different configurations was provided."
                    << endl;
            return false;
        }else if(minStUpDateFrequ == 0){
            minStUpDateFrequ = nrFrames;
        }
        int maxStUpDateFrequ = max(max(max(stereoPars.txChangeFRate, stereoPars.tyChangeFRate),
                                   max(stereoPars.tzChangeFRate, stereoPars.rollChangeFRate)),
                                   max(stereoPars.pitchChangeFRate, stereoPars.yawChangeFRate));
        if(nrFrames / minStUpDateFrequ > stereoPars.nrStereoConfigs){
            cerr << "Too less stereo configurations or the update frequency of an extrinsic value is too high." << endl;
            return false;
        }
        if(maxStUpDateFrequ > nrFrames){
            cerr << "One or more extrinsic values would never be updated. Change the "
                    "update frequency of one or more extrinsic values or set a higher number of frames." << endl;
            return false;
        }
        bool variableExtr = stereoPars.txVariable || stereoPars.tyVariable
                || stereoPars.tzVariable || stereoPars.rollVariable || stereoPars.pitchVariable
                || stereoPars.yawVariable;
        std::vector<std::vector<double>> tx;
        std::vector<std::vector<double>> ty;
        std::vector<std::vector<double>> tz;
        std::vector<std::vector<double>> roll;
        std::vector<std::vector<double>> pitch;
        std::vector<std::vector<double>> yaw;
        if(variableExtr && (stereoPars.nrStereoConfigs > 1)){
            if(nearZero(stereoPars.txRange.first - stereoPars.txRange.second)){
                if(stereoPars.txVariable){
                    cerr << "You specified txVariable=1 but the given txRange is 0." << endl;
                    return false;
                }
                tx.emplace_back(std::vector<double>(1,stereoPars.txStartVal));
            }else {
                tx.emplace_back(std::vector<double>(1, stereoPars.txRange.first));
                tx.back().push_back(stereoPars.txRange.second);
            }
            if(nearZero(stereoPars.tyRange.first - stereoPars.tyRange.second)){
                if(stereoPars.tyVariable){
                    cerr << "You specified tyVariable=1 but the given tyRange is 0." << endl;
                    return false;
                }
                ty.emplace_back(std::vector<double>(1,stereoPars.tyStartVal));
            }else {
                ty.emplace_back(std::vector<double>(1, stereoPars.tyRange.first));
                ty.back().push_back(stereoPars.tyRange.second);
            }
            if(nearZero(stereoPars.tzRange.first - stereoPars.tzRange.second)){
                if(stereoPars.tzVariable){
                    cerr << "You specified tzVariable=1 but the given tzRange is 0." << endl;
                    return false;
                }
                tz.emplace_back(std::vector<double>(1,stereoPars.tzStartVal));
            }else {
                tz.emplace_back(std::vector<double>(1, stereoPars.tzRange.first));
                tz.back().push_back(stereoPars.tzRange.second);
            }
            if(nearZero(stereoPars.rollRange.first - stereoPars.rollRange.second)){
                if(stereoPars.rollVariable){
                    cerr << "You specified rollVariable=1 but the given rollRange is 0." << endl;
                    return false;
                }
                roll.emplace_back(std::vector<double>(1,stereoPars.rollStartVal));
            }else {
                roll.emplace_back(std::vector<double>(1, stereoPars.rollRange.first));
                roll.back().push_back(stereoPars.rollRange.second);
            }
            if(nearZero(stereoPars.pitchRange.first - stereoPars.pitchRange.second)){
                if(stereoPars.pitchVariable){
                    cerr << "You specified pitchVariable=1 but the given pitchRange is 0." << endl;
                    return false;
                }
                pitch.emplace_back(std::vector<double>(1,stereoPars.pitchStartVal));
            }else {
                pitch.emplace_back(std::vector<double>(1, stereoPars.pitchRange.first));
                pitch.back().push_back(stereoPars.pitchRange.second);
            }
            if(nearZero(stereoPars.yawRange.first - stereoPars.yawRange.second)){
                if(stereoPars.yawVariable){
                    cerr << "You specified yawVariable=1 but the given yawRange is 0." << endl;
                    return false;
                }
                yaw.emplace_back(std::vector<double>(1,stereoPars.yawStartVal));
            }else {
                yaw.emplace_back(std::vector<double>(1, stereoPars.yawRange.first));
                yaw.back().push_back(stereoPars.yawRange.second);
            }
            double maxXY = max((tx[0].size() == 1) ? abs(tx[0][0]) : max(abs(tx[0][0]), abs(tx[0][1])),
                               (ty[0].size() == 1) ? abs(ty[0][0]) : max(abs(ty[0][0]), abs(ty[0][1])));
            if(((tz[0].size() == 1) ? abs(tz[0][0]) : max(abs(tz[0][0]), abs(tz[0][1]))) > maxXY){
                cerr << "Max. absolute value of the used range of tz (stereo) must be smaller than the max. absolute "
                        "value of used tx and tz. Error is due to wrong ranges and starting values." << endl;
                return false;
            }
            //Check if the higher value has negative sign
            if(((tx[0].size() == 1) ? abs(tx[0][0]) : max(abs(tx[0][0]), abs(tx[0][1])))
            > (((ty[0].size() == 1) ? abs(ty[0][0]) : max(abs(ty[0][0]), abs(ty[0][1]))) + DBL_EPSILON)){
                if(tx[0].size() == 1){
                    if(tx[0][0] > 0){
                        cout << "tx (stereo) is bigger than ty but not negative. The bigger value must be negative. "
                                "Changing sign of the tx range." << endl;
                        tx[0][0] *= -1.0;
                    }
                }else {
                    if ((abs(tx[0][1]) > abs(tx[0][0])) && (tx[0][1] > 0)) {
                        if(tx[0][0] > 0) {
                            cout << "tx (stereo) is bigger than ty but not negative. "
                                       "The bigger value must be negative. "
                                       "Changing sign of the tx range." << endl;
                            tx[0][0] *= -1.0;
                            tx[0][1] *= -1.0;
                        }else{
                            cout << "The possible values in the range for tx are bigger than for ty but might be "
                                    "positive. The bigger value must be negative. "
                                    "Your camera alignment (horizontal/vertical) might change." << endl;
                        }
                    }else if(nearZero(tx[0][0] + tx[0][1])
                    || (abs(tx[0][0]) < ((ty[0].size() == 1) ? abs(ty[0][0]) : max(abs(ty[0][0]), abs(ty[0][1]))))){
                        cout << "The possible values in the range for tx are bigger than for ty. "
                                "Your camera alignment (horizontal/vertical) might change." << endl;
                    }
                }
            } else if(((ty[0].size() == 1) ? abs(ty[0][0]) : max(abs(ty[0][0]), abs(ty[0][1])))
            > (((tx[0].size() == 1) ? abs(tx[0][0]) : max(abs(tx[0][0]), abs(tx[0][1]))) + DBL_EPSILON)){
                if(ty[0].size() == 1){
                    if(ty[0][0] > 0) {
                        cout << "ty (stereo) is bigger than tx but not negative. The bigger value must be negative. "
                                "Changing sign of the ty range." << endl;
                        ty[0][0] *= -1.0;
                    }
                }else {
                    if ((abs(ty[0][1]) > abs(ty[0][0])) && (ty[0][1] > 0)) {
                        if(ty[0][0] > 0) {
                            cout << "ty (stereo) is bigger than tx but not negative. "
                                    "The bigger value must be negative. "
                                    "Changing sign of the ty range." << endl;
                            ty[0][0] *= -1.0;
                            ty[0][1] *= -1.0;
                        }else{
                            cout << "The possible values in the range for ty are bigger than for tx but might be "
                                    "positive. The bigger value must be negative. "
                                    "Your camera alignment (horizontal/vertical) might change." << endl;
                        }
                    }else if(nearZero(ty[0][0] + ty[0][1])
                             || (abs(ty[0][0]) < ((tx[0].size() == 1) ? abs(tx[0][0]) : max(abs(tx[0][0]), abs(tx[0][1]))))){
                        cout << "The possible values in the range for ty are bigger than for tx. "
                                "Your camera alignment (horizontal/vertical) might change." << endl;
                    }
                }
            }else if(nearZero(((ty[0].size() == 1) ? abs(ty[0][0]) : max(abs(ty[0][0]), abs(ty[0][1])))
            - ((tx[0].size() == 1) ? abs(tx[0][0]) : max(abs(tx[0][0]), abs(tx[0][1]))))){
                cerr << "Stereo configuration is neither horizontal nor vertical (you used the same maximal "
                        "absolute values for tx and ty." << endl;
                return false;
            }
            vector<cv::Mat> t_new1;
            vector<double> roll_new1, pitch_new1, yaw_new1;
            if(!(stereoPars.txVariable
            && stereoPars.tyVariable
            && stereoPars.tzVariable
            && stereoPars.rollVariable
            && stereoPars.pitchVariable
            && stereoPars.yawVariable)){
                //Calculate the extrinsic parameters for the first stereo configuration
                // within the given ranges to achieve an image overlap nearest to the given value
                // (in addition to a few other constraints)
                GenStereoPars newStereoPars;
                try {
                    newStereoPars = GenStereoPars(tx, ty, tz, roll, pitch, yaw,
                            stereoPars.imageOverlap, stereoPars.imgSize);
                }catch(exception &e){
                    cerr << "Exception: " << e.what() << endl;
                    return false;
                }catch(...){
                    cerr << "Unkown exception." << endl;
                    return false;
                }

                int err = 0, err_cnt = 0;
                do {
                    err = newStereoPars.optimizeRtf(addPars.LMverbose);
                    newStereoPars.getNewRandSeed();
                    err_cnt++;
                }while(err && (err != -1) && (err_cnt < 20));

                newStereoPars.getEulerAngles(roll_new1, pitch_new1, yaw_new1);
                t_new1 = newStereoPars.tis;
                K_1 = newStereoPars.K1;
                K_1.copyTo(K_2);

                if(err) {
                    cout << endl;
                    cout << "**************************************************************************" << endl;
                    cout << "User specified image overlap = " << std::setprecision(3) << stereoPars.imageOverlap
                         << endl;
                    cout << "f= " << std::setprecision(2) << K_1.at<double>(0, 0)
                         << " cx= " << K_1.at<double>(0, 2) << " cy= " << K_1.at<double>(1, 2) << endl;
                    for (size_t i = 0; i < roll_new1.size(); i++) {
                        cout << "tx= " << std::setprecision(6) << t_new1[i].at<double>(0)
                             << " ty= " << t_new1[i].at<double>(1)
                             << " tz= " << t_new1[i].at<double>(2) << endl;
                        cout << "roll= " << std::setprecision(3) << roll_new1[i]
                             << " pitch= " << pitch_new1[i]
                             << " yaw= " << yaw_new1[i] << endl;
                    }
                    cout << "**************************************************************************" << endl
                         << endl;
                }

                if(err == -1){
                    cerr << "Not able to generate a valid random stereo camera configuration. "
                            "Ranges of tx and ty should be adapted." << endl;
                    return false;
                }else if(err){
                    cerr << "Not able to reach desired extrinsic stereo parameters or "
                            "input parameters are not usable. Try different parameters." << endl;
                    return false;
                }

                //Fix values for linear changes
                if(!stereoPars.txVariable){
                    tx[0].resize(1);
                    tx[0][0] = t_new1[0].at<double>(0);
                }
                if(!stereoPars.tyVariable){
                    ty[0].resize(1);
                    ty[0][0] = t_new1[0].at<double>(1);
                }
                if(!stereoPars.tzVariable){
                    tz[0].resize(1);
                    tz[0][0] = t_new1[0].at<double>(2);
                }
                if(!stereoPars.rollVariable){
                    roll[0].resize(1);
                    roll[0][0] = roll_new1[0];
                }
                if(!stereoPars.pitchVariable){
                    pitch[0].resize(1);
                    pitch[0][0] = pitch_new1[0];
                }
                if(!stereoPars.yawVariable){
                    yaw[0].resize(1);
                    yaw[0][0] = yaw_new1[0];
                }
            }
            enum keepFixed{
                TX = 0x1,
                TY = 0x2,
                TZ = 0x4,
                ROLL = 0x8,
                PITCH = 0x10,
                YAW = 0x20
            };
            std::vector<uint16_t> fixLater(nrFrames, 0);
            int cnt = 1;
            for (int i = 1; i < nrFrames; ++i) {
                bool setNewRanges = ((stereoPars.txChangeFRate) && ((i % stereoPars.txChangeFRate) == 0))
                        || ((stereoPars.tyChangeFRate) && ((i % stereoPars.tyChangeFRate) == 0))
                        || ((stereoPars.tzChangeFRate) && ((i % stereoPars.tzChangeFRate) == 0))
                        || ((stereoPars.rollChangeFRate) && ((i % stereoPars.rollChangeFRate) == 0))
                        || ((stereoPars.pitchChangeFRate) && ((i % stereoPars.pitchChangeFRate) == 0))
                        || ((stereoPars.yawChangeFRate) && ((i % stereoPars.yawChangeFRate) == 0));
                if(stereoPars.txChangeFRate &&((i % stereoPars.txChangeFRate) == 0)) {
                    if (stereoPars.txVariable) {
                        tx.emplace_back(std::vector<double>(1, stereoPars.txRange.first));
                        tx.back().push_back(stereoPars.txRange.second);
                    }else{
                        tx.emplace_back(std::vector<double>(1, tx.back()[0] + stereoPars.txLinChangeVal));
                    }
                } else if(setNewRanges){
                    tx.push_back(tx.back());
                    if(stereoPars.txVariable) {
                        fixLater[cnt] |= keepFixed::TX;
                    }
                }

                if(stereoPars.tyChangeFRate && ((i % stereoPars.tyChangeFRate) == 0)) {
                    if (stereoPars.tyVariable) {
                        ty.emplace_back(std::vector<double>(1, stereoPars.tyRange.first));
                        ty.back().push_back(stereoPars.tyRange.second);
                    }else{
                        ty.emplace_back(std::vector<double>(1, ty.back()[0] + stereoPars.tyLinChangeVal));
                    }
                } else if(setNewRanges){
                    ty.push_back(ty.back());
                    if(stereoPars.tyVariable) {
                        fixLater[cnt] |= keepFixed::TY;
                    }
                }

                if(stereoPars.tzChangeFRate && ((i % stereoPars.tzChangeFRate) == 0)) {
                    if (stereoPars.tzVariable) {
                        tz.emplace_back(std::vector<double>(1, stereoPars.tzRange.first));
                        tz.back().push_back(stereoPars.tzRange.second);
                    }else{
                        tz.emplace_back(std::vector<double>(1, tz.back()[0] + stereoPars.tzLinChangeVal));
                    }
                } else if(setNewRanges){
                    tz.push_back(tz.back());
                    if(stereoPars.tzVariable) {
                        fixLater[cnt] |= keepFixed::TZ;
                    }
                }

                if(stereoPars.rollChangeFRate && ((i % stereoPars.rollChangeFRate) == 0)) {
                    if (stereoPars.rollVariable) {
                        roll.emplace_back(std::vector<double>(1, stereoPars.rollRange.first));
                        roll.back().push_back(stereoPars.rollRange.second);
                    }else{
                        roll.emplace_back(std::vector<double>(1, roll.back()[0] + stereoPars.rollLinChangeVal));
                    }
                } else if(setNewRanges){
                    roll.push_back(roll.back());
                    if(stereoPars.rollVariable) {
                        fixLater[cnt] |= keepFixed::ROLL;
                    }
                }

                if(stereoPars.pitchChangeFRate && ((i % stereoPars.pitchChangeFRate) == 0)) {
                    if (stereoPars.pitchVariable) {
                        pitch.emplace_back(std::vector<double>(1, stereoPars.pitchRange.first));
                        pitch.back().push_back(stereoPars.pitchRange.second);
                    }else{
                        pitch.emplace_back(std::vector<double>(1, pitch.back()[0] + stereoPars.pitchLinChangeVal));
                    }
                } else if(setNewRanges){
                    pitch.push_back(pitch.back());
                    if(stereoPars.pitchVariable) {
                        fixLater[cnt] |= keepFixed::PITCH;
                    }
                }

                if(stereoPars.yawChangeFRate && ((i % stereoPars.yawChangeFRate) == 0)) {
                    if (stereoPars.yawVariable) {
                        yaw.emplace_back(std::vector<double>(1, stereoPars.yawRange.first));
                        yaw.back().push_back(stereoPars.yawRange.second);
                    }else{
                        yaw.emplace_back(std::vector<double>(1, yaw.back()[0] + stereoPars.yawLinChangeVal));
                    }
                } else if(setNewRanges){
                    yaw.push_back(yaw.back());
                    if(stereoPars.yawVariable) {
                        fixLater[cnt] |= keepFixed::YAW;
                    }
                }

                if(setNewRanges) {
                    maxXY = max((tx[cnt].size() == 1) ? abs(tx[cnt][0]) : max(abs(tx[cnt][0]), abs(tx[cnt][1])),
                                (ty[cnt].size() == 1) ? abs(ty[cnt][0]) : max(abs(ty[cnt][0]), abs(ty[cnt][1])));

                    if(((tz[cnt].size() == 1) ? abs(tz[cnt][0]) : max(abs(tz[cnt][0]), abs(tz[cnt][1]))) > maxXY) {
                        cerr << "Max. absolute value of the used range of tz (stereo) must be smaller than the "
                                "max. absolute value of used tx and tz. "
                                "Error is due to wrong linear change rates." << endl;
                        return false;
                    }

                    //Check if the higher value has negative sign
                    if(((tx[cnt].size() == 1) ? abs(tx[cnt][0]) : max(abs(tx[cnt][0]), abs(tx[cnt][1])))
                       > (((ty[cnt].size() == 1) ? abs(ty[cnt][0]) : max(abs(ty[cnt][0]), abs(ty[cnt][1])))
                       + DBL_EPSILON)) {
                        if(tx[cnt].size() == 1){
                            if(tx[cnt][0] > 0){
                                cerr << "The stereo camera configuration changed from vertical to horizontal "
                                        "alignment or the 2 stereo cameras switched their position "
                                        "(LEFT <-> RIGHT). Only 1 stereo configuration is allowed and a switch "
                                        "in position is not allowed. Change the parameters for creating vector "
                                        "elements tx and/or ty." << endl;
                                return false;
                            }
                        }else {
                            if ((abs(tx[cnt][1]) > abs(tx[cnt][0])) && (tx[cnt][1] > 0)) {
                                cerr << "The stereo camera configuration changed from vertical to horizontal "
                                           "alignment or the 2 stereo cameras switched their position "
                                           "(LEFT <-> RIGHT). Only 1 stereo configuration is allowed and a switch "
                                           "in position is not allowed. Change the parameters for creating vector "
                                           "elements tx and/or ty." << endl;
                                return false;
                            }
                        }
                    } else if(((ty[cnt].size() == 1) ? abs(ty[cnt][0]) : max(abs(ty[cnt][0]), abs(ty[cnt][1])))
                              > (((tx[cnt].size() == 1) ? abs(tx[cnt][0]) : max(abs(tx[cnt][0]), abs(tx[cnt][1])))
                              + DBL_EPSILON)) {
                        if(ty[cnt].size() == 1){
                            if(ty[cnt][0] > 0) {
                                cerr << "The stereo camera configuration changed from horizontal to vertical "
                                        "alignment or the 2 stereo cameras switched their position "
                                        "(TOP <-> BOTTOM). Only 1 stereo configuration is allowed and a switch "
                                        "in position is not allowed. Change the parameters for creating vector "
                                        "elements tx and/or ty." << endl;
                                return false;
                            }
                        }else {
                            if ((abs(ty[cnt][1]) > abs(ty[cnt][0])) && (ty[cnt][1] > 0)) {
                                cerr << "The stereo camera configuration changed from horizontal to vertical "
                                           "alignment or the 2 stereo cameras switched their position "
                                           "(TOP <-> BOTTOM). Only 1 stereo configuration is allowed and a switch "
                                           "in position is not allowed. Change the parameters for creating vector "
                                           "elements tx and/or ty." << endl;
                                return false;
                            }
                        }
                    } /*else if(nearZero(((ty[cnt].size() == 1) ? abs(ty[cnt][0]) : max(abs(ty[cnt][0]), abs(ty[cnt][1])))
                                      - ((tx[cnt].size() == 1) ? abs(tx[cnt][0]) : max(abs(tx[cnt][0]), abs(tx[cnt][1]))))) {
                        cerr << "Stereo configuration is neither horizontal nor vertical during creation "
                                "of different stereo parameters. "
                                "Change the parameters for creating vector elements tx and/or ty." << endl;
                        return false;
                    }*/
                    cnt++;
                }
                if((int)tx.size() >= stereoPars.nrStereoConfigs){
                    break;
                }
            }

            CV_Assert(cnt == (int)tx.size());
            if(cnt < (int)fixLater.size()) {
                fixLater.erase(fixLater.begin() + cnt, fixLater.end());
            }

            uint16_t isParFixed = 0;
            for(auto &i : fixLater){
                isParFixed += i;
            }

            //Get the parameters
            GenStereoPars newStereoPars;
            try {
                newStereoPars = GenStereoPars(tx, ty, tz, roll, pitch, yaw,
                                              stereoPars.imageOverlap, stereoPars.imgSize);
            }catch(exception &e){
                cerr << "Exception: " << e.what() << endl;
                return false;
            }catch(...){
                cerr << "Unkown exception." << endl;
                return false;
            }
            int err = 0, err_cnt = 0;
            do {
                err = newStereoPars.optimizeRtf(addPars.LMverbose);
                newStereoPars.getNewRandSeed();
                err_cnt++;
            }while(err && (err != -1) && (err_cnt < 20));

            roll_new1.clear();
            pitch_new1.clear();
            yaw_new1.clear();
            newStereoPars.getEulerAngles(roll_new1, pitch_new1, yaw_new1);
            t_new1 = newStereoPars.tis;
            K_1 = newStereoPars.K1;
            K_1.copyTo(K_2);

            if(err || (isParFixed == 0)) {
                cout << endl;
                cout << "**************************************************************************" << endl;
                cout << "User specified image overlap = " << std::setprecision(3) << stereoPars.imageOverlap
                     << endl;
                cout << "f= " << std::setprecision(2) << K_1.at<double>(0, 0)
                     << " cx= " << K_1.at<double>(0, 2) << " cy= " << K_1.at<double>(1, 2) << endl;
                for (size_t i = 0; i < roll_new1.size(); i++) {
                    cout << "tx= " << std::setprecision(6) << t_new1[i].at<double>(0)
                         << " ty= " << t_new1[i].at<double>(1)
                         << " tz= " << t_new1[i].at<double>(2) << endl;
                    cout << "roll= " << std::setprecision(3) << roll_new1[i]
                         << " pitch= " << pitch_new1[i]
                         << " yaw= " << yaw_new1[i] << endl;
                }
                cout << "**************************************************************************" << endl
                     << endl;
            }

            if(err == -1){
                cerr << "Not able to generate a valid random stereo camera configuration. "
                        "Ranges of tx and ty should be adapted." << endl;
                return false;
            }else if(err){
                cerr << "Not able to reach desired extrinsic stereo parameters or "
                        "input parameters are not usable. Try different parameters." << endl;
                return false;
            }

            if(isParFixed){
                //Adapt the camera matrix to get the best result
                for (int i = 0; i < cnt; ++i) {
                    if(fixLater[i] & TX){
                        tx[i][0] =  tx[i - 1][0];
                    }else{
                        tx[i].resize(1);
                        tx[i][0] =  t_new1[i].at<double>(0);
                    }
                    if(fixLater[i] & TY){
                        ty[i][0] =  ty[i - 1][0];
                    }else{
                        ty[i].resize(1);
                        ty[i][0] =  t_new1[i].at<double>(1);
                    }
                    if(fixLater[i] & TZ){
                        tz[i][0] =  tz[i - 1][0];
                    }else{
                        tz[i].resize(1);
                        tz[i][0] =  t_new1[i].at<double>(2);
                    }
                    if(fixLater[i] & ROLL){
                        roll[i][0] =  roll[i - 1][0];
                    }else{
                        roll[i].resize(1);
                        roll[i][0] =  roll_new1[i];
                    }
                    if(fixLater[i] & PITCH){
                        pitch[i][0] =  pitch[i - 1][0];
                    }else{
                        pitch[i].resize(1);
                        pitch[i][0] =  pitch_new1[i];
                    }
                    if(fixLater[i] & YAW){
                        yaw[i][0] =  yaw[i - 1][0];
                    }else{
                        yaw[i].resize(1);
                        yaw[i][0] =  yaw_new1[i];
                    }
                }
                //Get the parameters
                GenStereoPars newStereoPars1;
                try {
                    newStereoPars1 = GenStereoPars(tx, ty, tz, roll, pitch, yaw,
                                                  stereoPars.imageOverlap, stereoPars.imgSize);
                }catch(exception &e){
                    cerr << "Exception: " << e.what() << endl;
                    return false;
                }catch(...){
                    cerr << "Unkown exception." << endl;
                    return false;
                }
                err_cnt = 0;
                do {
                    err = newStereoPars1.optimizeRtf(addPars.LMverbose);
                    newStereoPars1.getNewRandSeed();
                    err_cnt++;
                }while(err && (err != -1) && (err_cnt < 20));

                roll_new1.clear();
                pitch_new1.clear();
                yaw_new1.clear();
                newStereoPars1.getEulerAngles(roll_new1, pitch_new1, yaw_new1);
                t_new1 = newStereoPars1.tis;
                K_1 = newStereoPars1.K1;
                K_1.copyTo(K_2);

                cout << endl;
                cout << "**************************************************************************" << endl;
                cout << "User specified image overlap = " << std::setprecision(3) << stereoPars.imageOverlap
                     << endl;
                cout << "f= " << std::setprecision(2) << K_1.at<double>(0, 0)
                     << " cx= " << K_1.at<double>(0, 2) << " cy= " << K_1.at<double>(1, 2) << endl;
                for (size_t i = 0; i < roll_new1.size(); i++) {
                    cout << "tx= " << std::setprecision(6) << t_new1[i].at<double>(0)
                         << " ty= " << t_new1[i].at<double>(1)
                         << " tz= " << t_new1[i].at<double>(2) << endl;
                    cout << "roll= " << std::setprecision(3) << roll_new1[i]
                         << " pitch= " << pitch_new1[i]
                         << " yaw= " << yaw_new1[i] << endl;
                }
                cout << "**************************************************************************" << endl
                     << endl;

                if(err == -1){
                    cerr << "Not able to generate a valid random stereo camera configuration. "
                            "Ranges of tx and ty should be adapted." << endl;
                    return false;
                }else if(err){
                    cerr << "Not able to reach desired extrinsic stereo parameters or "
                            "desired image overlap." << endl;
                    if(addPars.acceptBadStereoPars){
                        double minOvLap = stereoPars.imageOverlap + newStereoPars1.getNegMaxOvLapError();
                        if((minOvLap > 0.25) || ((stereoPars.imageOverlap <= 0.35) && (minOvLap > 0.05))) {
                            cout << "Accepting parameters as specified in config file." << endl;
                        }else{
                            return false;
                        }
                    }else {
                        cout << "Do you want to proceed with the parameters shown above? (y/n)";
                        string uip;
                        cin >> uip;
                        while ((uip != "y") && (uip != "n")) {
                            cout << endl << "Try again:";
                            cin >> uip;
                        }
                        cout << endl;
                        if (uip == "n") {
                            return false;
                        }
                    }
                }
                if(!newStereoPars1.getCamPars(Rv, tv, K_1, K_2)){
                    cerr << "Unable to get valid stereo camera parameters." << endl;
                    return false;
                }
            }else if(!newStereoPars.getCamPars(Rv, tv, K_1, K_2)){
                cerr << "Unable to get valid stereo camera parameters." << endl;
                return false;
            }
        }else{
            //Linear change of all stereo parameters
            if(nearZero(stereoPars.txRange.first - stereoPars.txRange.second)){
                tx.emplace_back(std::vector<double>(1,stereoPars.txStartVal));
            }else {
                tx.emplace_back(std::vector<double>(1, stereoPars.txRange.first));
                tx.back().push_back(stereoPars.txRange.second);
            }
            if(nearZero(stereoPars.tyRange.first - stereoPars.tyRange.second)){
                ty.emplace_back(std::vector<double>(1,stereoPars.tyStartVal));
            }else {
                ty.emplace_back(std::vector<double>(1, stereoPars.tyRange.first));
                ty.back().push_back(stereoPars.tyRange.second);
            }
            if(nearZero(stereoPars.tzRange.first - stereoPars.tzRange.second)){
                tz.emplace_back(std::vector<double>(1,stereoPars.tzStartVal));
            }else {
                tz.emplace_back(std::vector<double>(1, stereoPars.tzRange.first));
                tz.back().push_back(stereoPars.tzRange.second);
            }
            if(nearZero(stereoPars.rollRange.first - stereoPars.rollRange.second)){
                roll.emplace_back(std::vector<double>(1,stereoPars.rollStartVal));
            }else {
                roll.emplace_back(std::vector<double>(1, stereoPars.rollRange.first));
                roll.back().push_back(stereoPars.rollRange.second);
            }
            if(nearZero(stereoPars.pitchRange.first - stereoPars.pitchRange.second)){
                pitch.emplace_back(std::vector<double>(1,stereoPars.pitchStartVal));
            }else {
                pitch.emplace_back(std::vector<double>(1, stereoPars.pitchRange.first));
                pitch.back().push_back(stereoPars.pitchRange.second);
            }
            if(nearZero(stereoPars.yawRange.first - stereoPars.yawRange.second)){
                yaw.emplace_back(std::vector<double>(1,stereoPars.yawStartVal));
            }else {
                yaw.emplace_back(std::vector<double>(1, stereoPars.yawRange.first));
                yaw.back().push_back(stereoPars.yawRange.second);
            }


            double maxXY = max((tx[0].size() == 1) ? abs(tx[0][0]) : max(abs(tx[0][0]), abs(tx[0][1])),
                               (ty[0].size() == 1) ? abs(ty[0][0]) : max(abs(ty[0][0]), abs(ty[0][1])));
            if(((tz[0].size() == 1) ? abs(tz[0][0]) : max(abs(tz[0][0]), abs(tz[0][1]))) > maxXY){
                cerr << "Max. absolute value of the used range of tz (stereo) must be smaller than the max. absolute "
                        "value of used tx and tz. Error is due to wrong ranges and starting values." << endl;
                return false;
            }
            //Check if the higher value has negative sign
            if(((tx[0].size() == 1) ? abs(tx[0][0]) : max(abs(tx[0][0]), abs(tx[0][1])))
               > (((ty[0].size() == 1) ? abs(ty[0][0]) : max(abs(ty[0][0]), abs(ty[0][1]))) + DBL_EPSILON)){
                if(tx[0].size() == 1){
                    if(tx[0][0] > 0){
                        cout << "tx (stereo) is bigger than ty but not negative. The bigger value must be negative. "
                                "Changing sign of the tx range." << endl;
                        tx[0][0] *= -1.0;
                    }
                }else {
                    if ((abs(tx[0][1]) > abs(tx[0][0])) && (tx[0][1] > 0)) {
                        if(tx[0][0] > 0) {
                            cout << "tx (stereo) is bigger than ty but not negative. "
                                    "The bigger value must be negative. "
                                    "Changing sign of the tx range." << endl;
                            tx[0][0] *= -1.0;
                            tx[0][1] *= -1.0;
                        }else{
                            cout << "The possible values in the range for tx are bigger than for ty but might be "
                                    "positive. The bigger value must be negative. "
                                    "Your camera alignment (horizontal/vertical) might change." << endl;
                        }
                    }else if(nearZero(tx[0][0] + tx[0][1])
                             || (abs(tx[0][0]) < ((ty[0].size() == 1) ? abs(ty[0][0]) : max(abs(ty[0][0]), abs(ty[0][1]))))){
                        cout << "The possible values in the range for tx are bigger than for ty. "
                                "Your camera alignment (horizontal/vertical) might change." << endl;
                    }
                }
            } else if(((ty[0].size() == 1) ? abs(ty[0][0]) : max(abs(ty[0][0]), abs(ty[0][1])))
                      > (((tx[0].size() == 1) ? abs(tx[0][0]) : max(abs(tx[0][0]), abs(tx[0][1]))) + DBL_EPSILON)){
                if(ty[0].size() == 1){
                    if(ty[0][0] > 0) {
                        cout << "ty (stereo) is bigger than tx but not negative. The bigger value must be negative. "
                                "Changing sign of the ty range." << endl;
                        ty[0][0] *= -1.0;
                    }
                }else {
                    if ((abs(ty[0][1]) > abs(ty[0][0])) && (ty[0][1] > 0)) {
                        if(ty[0][0] > 0) {
                            cout << "ty (stereo) is bigger than tx but not negative. "
                                    "The bigger value must be negative. "
                                    "Changing sign of the ty range." << endl;
                            ty[0][0] *= -1.0;
                            ty[0][1] *= -1.0;
                        }else{
                            cout << "The possible values in the range for ty are bigger than for tx but might be "
                                    "positive. The bigger value must be negative. "
                                    "Your camera alignment (horizontal/vertical) might change." << endl;
                        }
                    }else if(nearZero(ty[0][0] + ty[0][1])
                             || (abs(ty[0][0]) < ((tx[0].size() == 1) ? abs(tx[0][0]) : max(abs(tx[0][0]), abs(tx[0][1]))))){
                        cout << "The possible values in the range for ty are bigger than for tx. "
                                "Your camera alignment (horizontal/vertical) might change." << endl;
                    }
                }
            }else if(nearZero(((ty[0].size() == 1) ? abs(ty[0][0]) : max(abs(ty[0][0]), abs(ty[0][1])))
                              - ((tx[0].size() == 1) ? abs(tx[0][0]) : max(abs(tx[0][0]), abs(tx[0][1]))))){
                cerr << "Stereo configuration is neither horizontal nor vertical (you used the same maximal "
                        "absolute values for tx and ty." << endl;
                return false;
            }

            //Calculate the extrinsic parameters for the first stereo configuration
            // within the given ranges to achieve an image overlap nearest to the given value
            // (in addition to a few other constraints)
            vector<cv::Mat> t_new1;
            vector<double> roll_new1, pitch_new1, yaw_new1;
            GenStereoPars newStereoPars;
            try {
                newStereoPars = GenStereoPars(tx, ty, tz, roll, pitch, yaw,
                                              stereoPars.imageOverlap, stereoPars.imgSize);
            }catch(exception &e){
                cerr << "Exception: " << e.what() << endl;
                return false;
            }catch(...){
                cerr << "Unkown exception." << endl;
                return false;
            }
            int err = 0, err_cnt = 0;
            do {
                err = newStereoPars.optimizeRtf(addPars.LMverbose);
                newStereoPars.getNewRandSeed();
                err_cnt++;
            }while(err && (err != -1) && (err_cnt < 20));

            newStereoPars.getEulerAngles(roll_new1, pitch_new1, yaw_new1);
            t_new1 = newStereoPars.tis;
            K_1 = newStereoPars.K1;
            K_1.copyTo(K_2);

            GenStereoPars newStereoPars1;
            if(stereoPars.nrStereoConfigs == 1){
                newStereoPars1 = newStereoPars;
            }else {
                if (err) {
                    cout << endl;
                    cout << "**************************************************************************" << endl;
                    cout << "User specified image overlap = " << std::setprecision(3) << stereoPars.imageOverlap
                         << endl;
                    cout << "f= " << std::setprecision(2) << K_1.at<double>(0, 0)
                         << " cx= " << K_1.at<double>(0, 2) << " cy= " << K_1.at<double>(1, 2) << endl;
                    for (size_t i = 0; i < roll_new1.size(); i++) {
                        cout << "tx= " << std::setprecision(6) << t_new1[i].at<double>(0)
                             << " ty= " << t_new1[i].at<double>(1)
                             << " tz= " << t_new1[i].at<double>(2) << endl;
                        cout << "roll= " << std::setprecision(3) << roll_new1[i]
                             << " pitch= " << pitch_new1[i]
                             << " yaw= " << yaw_new1[i] << endl;
                    }
                    cout << "**************************************************************************" << endl
                         << endl;
                }

                if (err == -1) {
                    cerr << "Not able to generate a valid random stereo camera configuration. "
                            "Ranges of tx and ty should be adapted." << endl;
                    return false;
                } else if (err) {
                    cerr << "Not able to reach desired extrinsic stereo parameters or "
                            "input parameters are not usable. Try different parameters." << endl;
                    return false;
                }

                //Fix values for linear changes
                tx[0].resize(1);
                tx[0][0] = t_new1[0].at<double>(0);
                ty[0].resize(1);
                ty[0][0] = t_new1[0].at<double>(1);
                tz[0].resize(1);
                tz[0][0] = t_new1[0].at<double>(2);
                roll[0].resize(1);
                roll[0][0] = roll_new1[0];
                pitch[0].resize(1);
                pitch[0][0] = pitch_new1[0];
                yaw[0].resize(1);
                yaw[0][0] = yaw_new1[0];

                int cnt = 1;
                for (int i = 1; i < nrFrames; ++i) {
                    bool setNewRanges = ((stereoPars.txChangeFRate) && ((i % stereoPars.txChangeFRate) == 0))
                                        || ((stereoPars.tyChangeFRate) && ((i % stereoPars.tyChangeFRate) == 0))
                                        || ((stereoPars.tzChangeFRate) && ((i % stereoPars.tzChangeFRate) == 0))
                                        || ((stereoPars.rollChangeFRate) && ((i % stereoPars.rollChangeFRate) == 0))
                                        || ((stereoPars.pitchChangeFRate) && ((i % stereoPars.pitchChangeFRate) == 0))
                                        || ((stereoPars.yawChangeFRate) && ((i % stereoPars.yawChangeFRate) == 0));
                    if (stereoPars.txChangeFRate && ((i % stereoPars.txChangeFRate) == 0)) {
                        tx.emplace_back(std::vector<double>(1, tx.back()[0] + stereoPars.txLinChangeVal));
                    } else if (setNewRanges) {
                        tx.push_back(tx.back());
                    }

                    if (stereoPars.tyChangeFRate && ((i % stereoPars.tyChangeFRate) == 0)) {
                        ty.emplace_back(std::vector<double>(1, ty.back()[0] + stereoPars.tyLinChangeVal));
                    } else if (setNewRanges) {
                        ty.push_back(ty.back());
                    }

                    if (stereoPars.tzChangeFRate && ((i % stereoPars.tzChangeFRate) == 0)) {
                        tz.emplace_back(std::vector<double>(1, tz.back()[0] + stereoPars.tzLinChangeVal));
                    } else if (setNewRanges) {
                        tz.push_back(tz.back());
                    }

                    if (stereoPars.rollChangeFRate && ((i % stereoPars.rollChangeFRate) == 0)) {
                        roll.emplace_back(std::vector<double>(1, roll.back()[0] + stereoPars.rollLinChangeVal));
                    } else if (setNewRanges) {
                        roll.push_back(roll.back());
                    }

                    if (stereoPars.pitchChangeFRate && ((i % stereoPars.pitchChangeFRate) == 0)) {
                        pitch.emplace_back(std::vector<double>(1, pitch.back()[0] + stereoPars.pitchLinChangeVal));
                    } else if (setNewRanges) {
                        pitch.push_back(pitch.back());
                    }

                    if (stereoPars.yawChangeFRate && ((i % stereoPars.yawChangeFRate) == 0)) {
                        yaw.emplace_back(std::vector<double>(1, yaw.back()[0] + stereoPars.yawLinChangeVal));
                    } else if (setNewRanges) {
                        yaw.push_back(yaw.back());
                    }

                    if (setNewRanges) {
                        maxXY = max(abs(tx[cnt][0]), abs(ty[cnt][0]));
                        if (abs(tz[cnt][0]) > maxXY) {
                            cerr << "Max. absolute value of the used range of tz (stereo) must be smaller than the "
                                    "max. absolute value of used tx and tz. "
                                    "Error is due to wrong linear change rates." << endl;
                            return false;
                        }

                        //Check if the higher value has negative sign
                        if (abs(tx[cnt][0]) > (abs(ty[cnt][0]) + DBL_EPSILON)) {
                            if (tx[cnt][0] > 0) {
                                cerr
                                        << "The stereo camera configuration changed from vertical to horizontal alignment or "
                                           "the 2 stereo cameras switched their position (LEFT <-> RIGHT). "
                                           "Only 1 stereo configuration is allowed and a switch in position is not allowed. "
                                           "Change the parameters for creating vector elements tx and/or ty." << endl;
                                return false;
                            }
                        } else if (abs(ty[cnt][0]) > (abs(tx[cnt][0]) + DBL_EPSILON)) {
                            if (ty[cnt][0] > 0) {
                                cerr
                                        << "The stereo camera configuration changed from horizontal to vertical alignment or "
                                           "the 2 stereo cameras switched their position (TOP <-> BOTTOM). "
                                           "Only 1 stereo configuration is allowed and a switch in position is not allowed. "
                                           "Change the parameters for creating vector elements tx and/or ty." << endl;
                                return false;
                            }
                        } else if (nearZero(abs(ty[cnt][0]) - abs(tx[cnt][0]))) {
                            cerr << "Stereo configuration is neither horizontal nor vertical during creation "
                                    "of different stereo parameters. "
                                    "Change the parameters for creating vector elements tx and/or ty." << endl;
                            return false;
                        }
                        cnt++;
                    }
                    if ((int) tx.size() >= stereoPars.nrStereoConfigs) {
                        break;
                    }
                }
                CV_Assert(cnt == (int) tx.size());
                //Get the parameters (adapt camera matrix and check extrinsics)
                try {
                    newStereoPars1 = GenStereoPars(tx, ty, tz, roll, pitch, yaw,
                                                   stereoPars.imageOverlap, stereoPars.imgSize);
                } catch (exception &e) {
                    cerr << "Exception: " << e.what() << endl;
                    return false;
                } catch (...) {
                    cerr << "Unkown exception." << endl;
                    return false;
                }
                err_cnt = 0;
                do {
                    err = newStereoPars1.optimizeRtf(addPars.LMverbose);
                    newStereoPars1.getNewRandSeed();
                    err_cnt++;
                } while (err && (err != -1) && (err_cnt < 20));

                roll_new1.clear();
                pitch_new1.clear();
                yaw_new1.clear();
                newStereoPars1.getEulerAngles(roll_new1, pitch_new1, yaw_new1);
                t_new1 = newStereoPars1.tis;
                K_1 = newStereoPars1.K1;
                K_1.copyTo(K_2);
            }

            cout << endl;
            cout << "**************************************************************************" << endl;
            cout << "User specified image overlap = " << std::setprecision(3) << stereoPars.imageOverlap
                 << endl;
            cout << "f= " << std::setprecision(2) << K_1.at<double>(0, 0)
                 << " cx= " << K_1.at<double>(0, 2) << " cy= " << K_1.at<double>(1, 2) << endl;
            for (size_t i = 0; i < roll_new1.size(); i++) {
                cout << "tx= " << std::setprecision(6) << t_new1[i].at<double>(0)
                     << " ty= " << t_new1[i].at<double>(1)
                     << " tz= " << t_new1[i].at<double>(2) << endl;
                cout << "roll= " << std::setprecision(3) << roll_new1[i]
                     << " pitch= " << pitch_new1[i]
                     << " yaw= " << yaw_new1[i] << endl;
            }
            cout << "**************************************************************************" << endl
                 << endl;

            if(err == -1){
                cerr << "Not able to generate a valid random stereo camera configuration. "
                        "Ranges/linear change rates of tx and ty should be adapted." << endl;
                return false;
            }else if(err){
                cerr << "Not able to reach desired extrinsic stereo parameters or "
                        "desired image overlap." << endl;
                if(addPars.acceptBadStereoPars){
                    double minOvLap = stereoPars.imageOverlap + newStereoPars1.getNegMaxOvLapError();
                    if((minOvLap > 0.25) || ((stereoPars.imageOverlap < 0.35) && (minOvLap > 0.05))) {
                        cout << "Accepting parameters as specified in config file." << endl;
                    }else{
                        return false;
                    }
                }else {
                    cout << "Do you want to proceed with the parameters shown above? (y/n)";
                    string uip;
                    cin >> uip;
                    while ((uip != "y") && (uip != "n")) {
                        cout << endl << "Try again:";
                        cin >> uip;
                    }
                    cout << endl;
                    if (uip == "n") {
                        return false;
                    }
                }
            }
            if(!newStereoPars1.getCamPars(Rv, tv, K_1, K_2)){
                cerr << "Unable to get valid stereo camera parameters." << endl;
                return false;
            }
        }
    }

    return true;
}

bool checkConfigFileName(string &filename, const string &errMsgPart){
    if (filename.find('\\') != std::string::npos)
        std::replace(filename.begin(), filename.end(), '\\', '/');
    size_t lastslash = filename.find_last_of('/');
    if(lastslash == string::npos){
        cerr << "The given option for " << errMsgPart << " file doesnt contain a path." << endl;
        return false;
    }
    string templPath = filename.substr(0, lastslash);
    if(!checkPathExists(templPath)){
        cerr << "The path for " << errMsgPart << " file does not exist!" << endl;
        return false;
    }
    string tmpFileName = filename.substr(lastslash + 1);
    size_t extNamePos = tmpFileName.find_last_of('.');
    if(extNamePos == string::npos){
        cerr << "The given option for " << errMsgPart << " file doesnt contain a file type." << endl;
        return false;
    }
    if(extNamePos == 0){
        cerr << "The given option for " << errMsgPart << " file doesnt contain a file name." << endl;
        return false;
    }
    string extName = tmpFileName.substr(extNamePos + 1);
    std::transform(extName.begin(), extName.end(), extName.begin(), ::tolower);
    if((extName != "yaml") && (extName != "yml") && (extName != "xml")){
        cerr << "The given option for " << errMsgPart << " file doesnt contain a "
                "valid file type (yaml/yml/xml)." << endl;
        return false;
    }
    return true;
}

bool genTemplateFile(const std::string &filename){
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rand_generator(seed);
    std::mt19937 rand2(seed);

    cvWriteComment(*fs, "This file contains user specific parameters used to generate "
                        "multiple consecutive stereo frames with correspondences.\n\n", 0);

    cvWriteComment(*fs, "Number of total frames. Max. 10000", 0);
    fs << "nrTotalFrames" << 100;

    cvWriteComment(*fs, "---- Options for stereo extrinsics ----\n\n", 0);

    cvWriteComment(*fs, "Number of different stereo configurations (max. 1000). \n"
                        "The number of frames per camera configuration is controlled by the smallest number of "
                        "the change rates (e.g. txChangeFRate) for tx, ty, tz, roll, pitch, and yaw.", 0);
    fs << "nrStereoConfigs" << 10;

    cvWriteComment(*fs, "Specifies after how many frames the x-component for the translation vector between "
                        "the stereo cameras shall be changed linearly by a given value. \n"
                        "Set this value to 0, if you do not want to change the x-component linearly.", 0);
    fs << "txChangeFRate" << 20;

    cvWriteComment(*fs, "Specifies after how many frames the y-component for the translation vector between "
                        "the stereo cameras shall be changed linearly by a given value. \n"
                        "Set this value to 0, if you do not want to change the y-component linearly.", 0);
    fs << "tyChangeFRate" << 10;

    cvWriteComment(*fs, "Specifies after how many frames the z-component for the translation vector between "
                        "the stereo cameras shall be changed linearly by a given value. \n"
                        "Set this value to 0, if you do not want to change the z-component linearly.", 0);
    fs << "tzChangeFRate" << 60;

    cvWriteComment(*fs, "Specifies after how many frames the roll angle (angle about x-axis) between "
                        "the stereo cameras shall be changed linearly by a given value. \n"
                        "Set this value to 0, if you do not want to change the roll linearly.", 0);
    fs << "rollChangeFRate" << 20;

    cvWriteComment(*fs, "Specifies after how many frames the pitch angle (angle about y-axis) between "
                        "the stereo cameras shall be changed linearly by a given value. \n"
                        "Set this value to 0, if you do not want to change the pitch linearly.", 0);
    fs << "pitchChangeFRate" << 30;

    cvWriteComment(*fs, "Specifies after how many frames the yaw angle (angle about z-axis) between "
                        "the stereo cameras shall be changed linearly by a given value. \n"
                        "Set this value to 0, if you do not want to change the yaw linearly.", 0);
    fs << "yawChangeFRate" << 0;

    cvWriteComment(*fs, "Linear change rate for tx (x-component of the translation vector "
                        "between the stereo cameras).", 0);
    fs << "txLinChangeVal" << -0.01;

    cvWriteComment(*fs, "Linear change rate for ty (y-component of the translation vector "
                        "between the stereo cameras).", 0);
    fs << "tyLinChangeVal" << 0.001;

    cvWriteComment(*fs, "Linear change rate for tz (z-component of the translation vector "
                        "between the stereo cameras).", 0);
    fs << "tzLinChangeVal" << 0.0005;

    cvWriteComment(*fs, "Linear change rate for the roll angle (angle in degrees about x-axis) "
                        "between the stereo cameras.", 0);
    fs << "rollLinChangeVal" << 0.5;

    cvWriteComment(*fs, "Linear change rate for the pitch angle (angle in degrees about y-axis) "
                        "between the stereo cameras.", 0);
    fs << "pitchLinChangeVal" << 0.1;

    cvWriteComment(*fs, "Linear change rate for the yaw angle (angle in degrees about z-axis) "
                        "between the stereo cameras.", 0);
    fs << "yawLinChangeVal" << 0;

    cvWriteComment(*fs, "Start value for tx (x-component (right) of the translation vector "
                        "between the stereo cameras). \nOnly used if the specified range is 0.", 0);
    fs << "txStartVal" << -1.0;

    cvWriteComment(*fs, "Start value for ty (y-component (down) of the translation vector "
                        "between the stereo cameras). \nOnly used if the specified range is 0.", 0);
    fs << "tyStartVal" << 0.1;

    cvWriteComment(*fs, "Start value for tz (z-component (forward) of the translation vector "
                        "between the stereo cameras). \nOnly used if the specified range is 0.", 0);
    fs << "tzStartVal" << 0.01;

    cvWriteComment(*fs, "Start value for the roll angle (angle in degrees about x-axis, "
                        "right handed coordinate system, R = R_y * R_z * R_x) "
                        "between the stereo cameras. \nOnly used if the specified range is 0.", 0);
    fs << "rollStartVal" << 2.0;

    cvWriteComment(*fs, "Start value for the pitch angle (angle in degrees about y-axis, "
                        "right handed coordinate system, R = R_y * R_z * R_x) "
                        "between the stereo cameras. \nOnly used if the specified range is 0.", 0);
    fs << "pitchStartVal" << -8.2;

    cvWriteComment(*fs, "Start value for the yaw angle (angle in degrees about z-axis, "
                        "right handed coordinate system, R = R_y * R_z * R_x) "
                        "between the stereo cameras. \nOnly used if the specified range is 0.", 0);
    fs << "yawStartVal" << 0.3;

    cvWriteComment(*fs, "Possible range of the initial tx (x-component (right) of the translation vector "
                        "between the stereo cameras) to be able to meet a user specific image overlap. \n"
                        "The optimal tx starting-value is by default only estimated for the "
                        "first stereo configuration. If specified by txVariable, the range is used for every new "
                        "configuration. \nIf the range (difference between both values) is 0, "
                        "tx will be kept fixed at the given start value for tx.", 0);
    fs << "txRange";
    fs << "{" << "first" << 0;
    fs << "second" << 0 << "}";

    cvWriteComment(*fs, "Possible range of the initial ty (y-component (down) of the translation vector "
                        "between the stereo cameras) to be able to meet a user specific image overlap. \n"
                        "The optimal ty starting-value is by default only estimated for the "
                        "first stereo configuration. If specified by tyVariable, the range is used for every new "
                        "configuration. \nIf the range (difference between both values) is 0, "
                        "ty will be kept fixed at the given start value for ty.", 0);
    fs << "tyRange";
    fs << "{" << "first" << -0.1;
    fs << "second" << 0.1 << "}";

    cvWriteComment(*fs, "Possible range of the initial tz (z-component (forward) of the translation vector "
                        "between the stereo cameras) to be able to meet a user specific image overlap. \n"
                        "The optimal tz starting-value is by default only estimated for the "
                        "first stereo configuration. If specified by tzVariable, the range is used for every new "
                        "configuration. \nIf the range (difference between both values) is 0, "
                        "tz will be kept fixed at the given start value for tz.", 0);
    fs << "tzRange";
    fs << "{" << "first" << 0;
    fs << "second" << 0 << "}";

    cvWriteComment(*fs, "Possible range for the initial roll angle (angle in degrees about x-axis "
                        "between the stereo cameras, right handed coordinate system, R = R_y * R_z * R_x) \n"
                        "to be able to meet a user specific image overlap. "
                        "The optimal roll starting-value is by default only estimated for the \n"
                        "first stereo configuration. If specified by rollVariable, the range is used for every new "
                        "configuration. \nIf the range (difference between both values) is 0, "
                        "the roll angle will be kept fixed at the given start value for roll.", 0);
    fs << "rollRange";
    fs << "{" << "first" << -5.0;
    fs << "second" << 7.0 << "}";

    cvWriteComment(*fs, "Possible range for the initial pitch angle (angle in degrees about y-axis "
                        "between the stereo cameras, right handed coordinate system, R = R_y * R_z * R_x) \n"
                        "to be able to meet a user specific image overlap. "
                        "The optimal pitch starting-value is by default only estimated for the \n"
                        "first stereo configuration. If specified by pitchVariable, the range is used for every new "
                        "configuration. \nIf the range (difference between both values) is 0, "
                        "the pitch angle will be kept fixed at the given start value for pitch.", 0);
    fs << "pitchRange";
    fs << "{" << "first" << -15.0;
    fs << "second" << 10.0 << "}";

    cvWriteComment(*fs, "Possible range for the initial yaw angle (angle in degrees about z-axis "
                        "between the stereo cameras, right handed coordinate system, R = R_y * R_z * R_x) \n"
                        "to be able to meet a user specific image overlap. "
                        "The optimal yaw starting-value is by default only estimated for the \n"
                        "first stereo configuration. If specified by yawVariable, the range is used for every new "
                        "configuration. \nIf the range (difference between both values) is 0, "
                        "the yaw angle will be kept fixed at the given start value for yaw.", 0);
    fs << "yawRange";
    fs << "{" << "first" << 0;
    fs << "second" << 0 << "}";

    cvWriteComment(*fs, "Use the specified tx-range for every new stereo configuration (not only for the first) "
                        "where tx should change (depends on txChangeFRate). \n"
                        "If this value is 1, an optimal value is selected within the given range for every new changed "
                        "stereo configuration to meet a \nspecific image overlap between the 2 stereo cameras. "
                        "Otherwise set the value to 0.", 0);
    fs << "txVariable" << 0;

    cvWriteComment(*fs, "Use the specified ty-range for every new stereo configuration (not only for the first) "
                        "where ty should change (depends on tyChangeFRate). \n"
                        "If this value is 1, an optimal value is selected within the given range for every new changed "
                        "stereo configuration to meet a \nspecific image overlap between the 2 stereo cameras. "
                        "Otherwise set the value to 0.", 0);
    fs << "tyVariable" << 0;

    cvWriteComment(*fs, "Use the specified tz-range for every new stereo configuration (not only for the first) "
                        "where tz should change (depends on tzChangeFRate). \n"
                        "If this value is 1, an optimal value is selected within the given range for every new changed "
                        "stereo configuration to meet a \nspecific image overlap between the 2 stereo cameras. "
                        "Otherwise set the value to 0.", 0);
    fs << "tzVariable" << 0;

    cvWriteComment(*fs, "Use the specified roll-range for every new stereo configuration (not only for the first) "
                        "where the roll angle should change (depends on rollChangeFRate). \n"
                        "If this value is 1, an optimal value is selected within the given range for every new changed "
                        "stereo configuration to meet a \nspecific image overlap between the 2 stereo cameras. "
                        "Otherwise set the value to 0.", 0);
    fs << "rollVariable" << 0;

    cvWriteComment(*fs, "Use the specified pitch-range for every new stereo configuration (not only for the first) "
                        "where the pitch angle should change (depends on pitchChangeFRate). \n"
                        "If this value is 1, an optimal value is selected within the given range for every new changed "
                        "stereo configuration to meet a \nspecific image overlap between the 2 stereo cameras. "
                        "Otherwise set the value to 0.", 0);
    fs << "pitchVariable" << 0;

    cvWriteComment(*fs, "Use the specified yaw-range for every new stereo configuration (not only for the first) "
                        "where the yaw angle should change (depends on yawChangeFRate). \n"
                        "If this value is 1, an optimal value is selected within the given range for every new changed "
                        "stereo configuration to meet a \nspecific image overlap between the 2 stereo cameras. "
                        "Otherwise set the value to 0.", 0);
    fs << "yawVariable" << 0;

    cvWriteComment(*fs, "Set this variable to 1 if you want to use your own extrinsic and intrinsic camera "
                        "parameters (set them in specificCamPars). \nOtherwise, set this value to 0.", 0);
    fs << "useSpecificCamPars" << 0;
    cvWriteComment(*fs, "Input for specific extrinsic and intrinsic stereo camera parameters. The "
                        "translation vectors must include the scale factor. \nRotation matrices must be "
                        "generated by R = R_y * R_z * R_x (right handed coordinate system, "
                        "x-axis: right, y-axis: down, z-axis: forward); x2 = R * x1 + t. \n"
                        "Every list element specifies a new stereo pair. To use these parameters,"
                        "useSpecificCamPars must be set to 1. \n"
                        "Currently there is only support for different extrinsics. The camera matrices "
                        "stay the same as for the first stereo configuration \nregardless of what is entered "
                        "for the camera matrices in the subsequent configurations. \n"
                        "The update frequence (in frames) is calculated by nrTotalFrames devided by "
                        "the number of entered stereo configurations (here) or nrStereoConfigs "
                        "(whatever value is smaller; nrStereoConfigs is not allowed to be larger than the "
                        "number of entries entered here).", 0);
    vector<Mat> Rsu(2), tsu(2), K1su(2), K2su(2);
    Rsu[0] = eulerAnglesToRotationMatrix(2.0 * M_PI / 180.0,
                                         -4.0 * M_PI / 180.0,
                                         0.3 * M_PI / 180.0);
    Rsu[1] = eulerAnglesToRotationMatrix(2.2 * M_PI / 180.0,
                                         -3.9 * M_PI / 180.0,
                                         0.35 * M_PI / 180.0);
    tsu[0] = (Mat_<double>(3,1) << -0.95, 0.05, 0.02);
    tsu[1] = (Mat_<double>(3,1) << -0.94, 0.03, 0.02);
    double f = 1280.0 / (2.0 * tan(PI / 4.0));
    K1su[0] = Mat::eye(3, 3, CV_64FC1);
    K1su[0].at<double>(0, 0) = f * 0.995;
    K1su[0].at<double>(1, 1) = f * 1.03;
    K1su[0].at<double>(0, 2) = 1.01 * 1280.0 / 2.0;
    K1su[0].at<double>(1, 2) = 1.025 * 720.0 / 2.0;
    K1su[1] = K1su[0].clone();
    K2su[0] = Mat::eye(3, 3, CV_64FC1);
    K2su[0].at<double>(0, 0) = f * 1.01;
    K2su[0].at<double>(1, 1) = f * 1.02;
    K2su[0].at<double>(0, 2) = 0.98 * 1280.0 / 2.0;
    K2su[0].at<double>(1, 2) = 1.03 * 720.0 / 2.0;
    K2su[1] = K2su[0].clone();
    fs << "specificCamPars" << "[";
    for (int k = 0; k < 2; ++k) {
        fs << "{" << "R" << Rsu[k];
        fs << "t" << tsu[k];
        fs << "K1" << K1su[k];
        fs << "K2" << K2su[k] << "}";
    }
    fs << "]";

    cvWriteComment(*fs, "Desired image overlap of both stereo images at mid depth (see corrsPerRegionRandInit). \n"
                        "Value range: 0.1 to 1.0", 0);
    fs << "imageOverlap" << 0.8;

    cvWriteComment(*fs, "---- Options for generating 3D scenes ----\n\n", 0);

    cvWriteComment(*fs, "Inlier ratio range for all stereo frames.", 0);
    fs << "inlRatRange";
    fs << "{" << "first" << 0.4;
    fs << "second" << 0.95 << "}";
    cvWriteComment(*fs, "Inlier ratio change rate from pair to pair. \n"
                        "If 0, the inlier ratio within the given range is always the same for every image pair "
                        "(it is selected within the given inlier range in the beginning). \n"
                        "If 100.0, the inlier ratio is chosen completely random within the given range for every "
                        "stereo frame separately. \nFor values between 0 and 100.0, the inlier ratio selected is "
                        "not allowed to change more than this factor from the inlier ratio "
                        "of the last stereo frame.", 0);
    fs << "inlRatChanges" << 20.0;
    cvWriteComment(*fs, "Number of true positives (TP) range for all stereo frames.", 0);
    fs << "truePosRange";
    fs << "{" << "first" << 30;
    fs << "second" << 1000 << "}";
    cvWriteComment(*fs, "True positives change rate from pair to pair. \nIf 0, the true positives within the "
                        "given range are always the same for every image pair "
                        "(it is selected within the given TP range in the beginning). \n"
                        "If 100.0, the true positives are chosen completely random within the given range. "
                        "For values between 0 and 100.0, \nthe true positives selected are not allowed to "
                        "change more than this factor from the true positives of the last stereo frame.", 0);
    fs << "truePosChanges" << 40.0;
    cvWriteComment(*fs, "Minimum distance between keypoints in the first (left or top) "
                        "stereo image for every frame", 0);
    fs << "minKeypDist" << 4.0;
    cvWriteComment(*fs, "Portion of correspondences at depth regions (near, mid, and far). "
                        "The values are double precision and the values must not sum to 1.0. \n"
                        "This is performed internally.", 0);
    fs << "corrsPerDepth";
    fs << "{" << "near" << 0.3;
    fs << "mid" << 0.1;
    fs << "far" << 0.6 << "}";
    cvWriteComment(*fs, "List of portions of image correspondences at regions "
                        "(Matrices must be 3x3 as the image is divided into 9 regions with similar size). \n"
                        "Maybe this values do not hold: Also depends on back-projected 3D-points "
                        "from prior frames. \nThe values are double precision and the values must not sum to 1.0. "
                        "This is performed internally. "
                        "\nIf more than one matrix is provided, corrsPerRegRepRate specifies the number of "
                        "subsequent frames for which a matrix is valid. \n"
                        "After all matrices are used, the first one is used again. "
                        "If only one matrix is provided, it is used for every frame.", 0);
    std::vector<cv::Mat> corrsPerRegion;
    for (int i = 0; i < 3; i++) {
        cv::Mat corrsPReg(3, 3, CV_64FC1);
        cv::randu(corrsPReg, Scalar(0), Scalar(1.0));
        corrsPReg /= sum(corrsPReg)[0];
        corrsPerRegion.push_back(corrsPReg.clone());
    }
    fs << "corrsPerRegion" << "[";
    for (auto &i : corrsPerRegion) {
        fs << i;
    }
    fs << "]";
    cvWriteComment(*fs, "Repeat rate of portion of correspondences at regions (corrsPerRegion). If more than one "
                        "matrix of portions of correspondences at regions is provided, \n"
                        "this number specifies the number of subsequent frames for which such a matrix is valid. "
                        "After all matrices are used, the first one is used again. \n"
                        "If 0 and no matrix of portions of correspondences at regions is provided, "
                        "as many random matrices as frames are randomly generated.", 0);
    fs << "corrsPerRegRepRate" << 5;
    cvWriteComment(*fs, "If 1 and corrsPerRegRepRate=0, corrsPerRegion is initialized randomly "
                        "for every frame seperately. \n"
                        "If 1 and 0 < corrsPerRegRepRate < nrTotalFrames, "
                        "nrTotalFrames / corrsPerRegRepRate different random corrsPerRegion are calculated. \n"
                        "If 0, the values from corrsPerRegion are used.", 0);
    fs << "corrsPerRegionRandInit" << 0;
    cvWriteComment(*fs, "Portion of depths per region (must be 3x3). For each of the 3x3=9 image regions, "
                        "the portion of near, mid, and far depths can be specified. \n"
                        "Far depth beginning corresponds to mid depth end calculated by f_b = f * b^2 / 0.15 with "
                        "the focal length f, baseline length b and an approximate correspondence accuracy of 0.15; \n"
                        "Far depth end f_e = 20 * f_b; Near depth beginning n_b corresponds to the depth where "
                        "the views of the 2 stereo cameras start to overlap; \n"
                        "Mid depth beginning and near depth end are caluclated by m_b = (f_b + n_b) / 2; \n"
                        "The values entered here are double precision and they must not sum to 1.0. This is performed "
                        "internally. \nIf the overall depth definition (corrsPerDepth) is not met, this tensor "
                        "is adapted. If this list is left empty ([]), it is initialized randomly. \n"
                        "Maybe this values do not hold: Also depends on back-projected 3D-points "
                        "from prior frames.", 0);
    std::vector<std::vector<depthPortion>> depthsPerRegion;
    depthsPerRegion.resize(3, std::vector<depthPortion>(3));
    for (size_t y = 0; y < 3; y++) {
        for (size_t x = 0; x < 3; x++) {
            depthsPerRegion[y][x] = depthPortion(getRandDoubleVal(rand_generator, 0, 1.0),
                                                 getRandDoubleVal(rand_generator, 0, 1.0),
                                                 getRandDoubleVal(rand_generator, 0, 1.0));
        }
    }
    fs << "depthsPerRegion" << "[";
    for (auto &i : depthsPerRegion) {
        for (auto &j : i) {
            fs << "{" << "near" << j.near;
            fs << "mid" << j.mid;
            fs << "far" << j.far << "}";
        }
    }
    fs << "]";
    cvWriteComment(*fs, "Min and Max number of connected depth areas "
                        "(in the image domain (same size as the used image size) where they are generated \n"
                        "(each pixel in the image domain holds a depth value)) per region (must be 3x3). \n"
                        "The minimum number (first) must be larger 0. The maximum number is bounded by "
                        "a minimum area with similar depths, which is 16 pixels. \n"
                        "The final number of connected depth areas per region is chosen randomly between "
                        "min and max for every frame. \n"
                        "If this list is left empty ([]), it is initialized randomly. "
                        "If min and max are equal, exactly this number of connected depth areas is used. \n"
                        "Maybe this values do not hold: Also depends on back-projected 3D-points "
                        "from prior frames.", 0);
    std::vector<std::vector<std::pair<size_t, size_t>>> nrDepthAreasPReg;
    nrDepthAreasPReg.resize(3, std::vector<std::pair<size_t, size_t>>(3));
    for (size_t y = 0; y < 3; y++) {
        for (size_t x = 0; x < 3; x++) {
            nrDepthAreasPReg[y][x] = std::pair<size_t, size_t>(1, (size_t) rand2() % 20 + 1);
            int nrDepthAreasPRegEqu = (int) (rand2() % 2);
            if (nrDepthAreasPRegEqu == 1) {
                nrDepthAreasPReg[y][x].first = nrDepthAreasPReg[y][x].second;
            }
        }
    }
    fs << "nrDepthAreasPReg" << "[";
    for (auto &i : nrDepthAreasPReg) {
        for (auto &j : i) {
            fs << "{" << "first" << (int) j.first;
            fs << "second" << (int) j.second << "}";
        }
    }
    fs << "]";

    cvWriteComment(*fs, "If 1, an ellipsoid is used as camera track (consecutive positions of the "
                        "top/left stereo camera center. \n"
                        "If 2, a custom track can be entered into camTrack. \n"
                        "If 3, a random track will be generated.", 0);
    fs << "trackOption" << 1;
    cvWriteComment(*fs, "Ellipsoid parameters: \nxDirection (either -1 (left) or +1 (right)); \n"
                        "xzExpansion (value range -100 to +100.0, no 0) describes how much "
                        "smaller/larger the expansion in x (right) direction is compared to z (foward) - \n"
                        "an absolute value below 1.0 stands for a larger x-expansion, a value equal +/-1.0 for an "
                        "equal expansion (circle) and an absolute value larger 1.0 for a larger y-expansion; \n"
                        "xyExpansion (value range -100.0 to +100.0) describes a factor of the mean used height "
                        "(down; if no flight mode is used the height stays always the same; \n"
                        "height = xDirection * xyExpansion * sin(theta)) compared to the expansion in x direction; \n"
                        "thetaRange holds the minimum and maximum elevation angle (value range -PI/2 to +PI/2) in "
                        "y-direction (down). \nIf the values of the range are equal, flight mode is disabled "
                        "and the height (y) stays the same over the whole track. \n"
                        "For the camera positions, loop closing is performed whereas the last camera position "
                        "is not the same as the first, but near to it. \nThe scale of the generated track is not "
                        "important as it is changed internally that it fits all stereo frames. \n"
                        "maxTrackElements (see next option) specifies how many track segments within the "
                        "ellipsoid are generated.", 0);
    fs << "ellipsoidTrack";
    fs << "{" << "xDirection" << 1;
    fs << "xzExpansion" << 0.3;
    fs << "xyExpansion" << -0.2;
    fs << "thetaRange";
    fs << "{" << "min" << -M_PI_4;
    fs << "max" << M_PI_4 << "}" << "}";

    cvWriteComment(*fs, "maxTrackElements specifies the number of track segments generated within an ellipsoid or "
                        "during a randomized track generation (max 10000 segments)", 0);
    fs << "maxTrackElements" << 100;

    cvWriteComment(*fs, "Parameters for random generation of a track with maxTrackElements segments: \n"
                        "xDirection (either -1 (left) or +1 (right)); \n"
                        "xzDirectionRange (value range -1000.0 to +1000.0) specifies the vector direction in "
                        "x (right) compared to z (forward) direction - \nan absolute value below 1.0 stands for a "
                        "direction mainly in x and an absolute value above 1.0 stands for a direction mainly in z; \n"
                        "xyDirectionRange (value range -1000.0 to +1000.0) specifies the vector direction in "
                        "x (right) compared to y (down) direction - \nan absolute value below 1.0 stands for a "
                        "direction mainly in x compared to y and an absolute value higher 1.0 for a direction mainly "
                        "in y compared to x; \nThe direction of a subsequent track element depends on the "
                        "direction of the track element before - \nThe amount it can change depends on the "
                        "factor allowedChangeSD (value range 0.05 to 1.0) - it corresponds to the standard deviation "
                        "centered aground 1.0 (New track sement: \ntx_new = allowedChange_1 * tx_old, \n"
                        "ty_new = (2.0 * allowedChange_2 * ty_old + tx_new * xyDirectionRandInRange_new) / 3.0, \n"
                        "tz_new = (2.0 * allowedChange_3 * tz_old + tx_new * xzDirectionRandInRange_new) / 3.0, \n"
                        "new_track_position = old_track_position + [tx_new; ty_new; tz_new]). \n"
                        "The scale of the generated track is not important as it is changed internally that it "
                        "fits all stereo frames.", 0);
    fs << "randomTrack";
    fs << "{" << "xDirection" << 1;
    fs << "xzDirectionRange";
    fs << "{" << "first" << 0.2;
    fs << "second" << 0.9 << "}";
    fs << "xyDirectionRange";
    fs << "{" << "first" << -0.1;
    fs << "second" << 0.1 << "}";
    fs << "allowedChangeSD" << 0.3 << "}";

    cvWriteComment(*fs, "Movement direction or track of the cameras (manual input). Input a custom track as a list of"
                        "cv::Mat with the format [x_pos; y_pos; z_pos]. \nThe scale of the generated track is not "
                        "important as it is changed internally that it fits all stereo frames. \nIf you enter only "
                        "1 vector [x_pos; y_pos; z_pos], it is interpreted as a directional vector and the "
                        "camera center of the left/top stereo camera moves into this direction.", 0);
    vector<Mat> camTrack;
    Mat singlePos = (Mat_<double>(3,1) << 2.0, 0, 1.0);
    camTrack.emplace_back(singlePos.clone());
    singlePos = (Mat_<double>(3,1) << 3.0, -0.1, 1.3);
    camTrack.emplace_back(singlePos.clone());
    singlePos = (Mat_<double>(3,1) << 4.5, -0.05, 1.4);
    camTrack.emplace_back(singlePos.clone());
    fs << "camTrack" << "[";
    for (auto &i : camTrack) {
        fs << i;
    }
    fs << "]";

    cvWriteComment(*fs, "Relative velocity of the camera movement (value between 0 and 10.0; must be larger 0). "
                        "The velocity is relative to the baseline length between the stereo cameras. \n"
                        "Thus, the absolute camera velocity (equals distance between camera centers) "
                        "along the camera track is relCamVelocity * norm(t_stereo), \nwhere t_stereo is "
                        "the translation vector between the 2 stereo cameras. \n"
                        "The total length of a camera track is the absolute camera velocity times the "
                        "number of frames", 0);
    fs << "relCamVelocity" << 2.0;
    cvWriteComment(*fs, "Rotation angle about the x-axis (roll in degrees, right handed) of the stereo pair (centered "
                        "at camera centre of left/top stereo camera) on the track. \nThis rotation can change the camera "
                        "orientation for which without rotation the z - component of the relative movement "
                        "vector coincides with the principal axis of the camera. \n"
                        "The rotation matrix is generated using the notation R_y * R_z * R_x.", 0);
    fs << "rollCamTrack" << 0;
    cvWriteComment(*fs, "Rotation angle about the y-axis (pitch in degrees, right handed) of the stereo pair (centered "
                        "at camera centre of left/top stereo camera) on the track. \nThis rotation can change the camera "
                        "orientation for which without rotation the z - component of the relative movement "
                        "vector coincides with the principal axis of the camera. \n"
                        "The rotation matrix is generated using the notation R_y * R_z * R_x.", 0);
    fs << "pitchCamTrack" << -90.0;
    cvWriteComment(*fs, "Rotation angle about the z-axis (yaw in degrees, right handed) of the stereo pair (centered "
                        "at camera centre of left/top stereo camera) on the track. \nThis rotation can change the camera "
                        "orientation for which without rotation the z - component of the relative movement "
                        "vector coincides with the principal axis of the camera. \n"
                        "The rotation matrix is generated using the notation R_y * R_z * R_x.", 0);
    fs << "yawCamTrack" << 1.0;
    cvWriteComment(*fs, "Number of moving objects in the scene at the beginning. If a moving object is visible again "
                        "in a subsequent frame, it is backprojected to the image plane. \nIf the portion of "
                        "backprojected correspondences on a moving object compared to the frame where it was "
                        "generated drops below a user specified threshold (see minMovObjCorrPortion), \n"
                        "it is removed from the scene. If too many moving objects were removed and the number of "
                        "remaining moving objects drops below minNrMovObjs, new moving objects are inserted.", 0);
    fs << "nrMovObjs" << 4;
    cvWriteComment(*fs, "Possible starting positions (regions in the image) of moving objects in the image "
                        "(must be 3x3 boolean (CV_8UC1))", 0);
    Mat startPosMovObjs = (Mat_<bool>(3,3) <<
            true, true, false,
            true, false, false,
            false, false, false);
    fs << "startPosMovObjs" << startPosMovObjs;
    cvWriteComment(*fs, "Relative area range of moving objects. \nArea range relative to the image area in "
                        "the beginning. The final occupied area of every moving object in the image is selected "
                        "randomly within the given range. \nAs moving objects are always put into the foreground "
                        "it is advisable too use small relative area ranges depending on the number of \n"
                        "demanded moving objects to allow static elements to be visible in the image.", 0);
    fs << "relAreaRangeMovObjs";
    fs << "{" << "first" << 0.05;
    fs << "second" << 0.15 << "}";
    cvWriteComment(*fs, "Depths of moving objects. Moving objects are always visible and not covered by "
                        "other static objects. \nIf the number of given depths is 1, this depth is used for "
                        "every object. \nIf the number of given depths is equal \"nrMovObjs\", the corresponding "
                        "depth is used for every moving object. \nIf the number of given depths is smaller "
                        "and between 2 and 3, the depths for the moving objects are selected uniformly "
                        "distributed from the given depths. \nFor a number of given depths larger 3 and unequal "
                        "to \"nrMovObjs\", a portion for every depth that should be used can be defined \n"
                        "(e.g. 3 x far, 2 x near, 1 x mid -> 3 / 6 x far, 2 / 6 x near, 1 / 6 x mid). \n"
                        "The following values for different depths can be used: "
                        "NEAR: 1, MID: 2, FAR: 4", 0);
    std::vector<depthClass> movObjDepth(3);
    movObjDepth[0] = depthClass::NEAR;
    movObjDepth[1] = depthClass::MID;
    movObjDepth[2] = depthClass::FAR;
    fs << "movObjDepth" << "[";
    for (auto &i : movObjDepth) {
        fs << (int) i;
    }
    fs << "]";
    cvWriteComment(*fs, "Movement direction of moving objects relative to the camera movement (must be 3x1 double "
                        "cv::Mat). \nThis vector will be normalized and added to the normalized direction vector "
                        "of the actual camera track segment. \nThus, the visibility of moving objects is mainly "
                        "influenced by their depth of occurrence, \nthis relative direction vector and the "
                        "rotation of the stereo pair relative to the camera movement direction "
                        "(see rollCamTrack, pitchCamTrack, yawCamTrack). \n"
                        "The movement direction is linear and does not change if the movement direction "
                        "of the camera changes during the lifetime of a moving object. \nThe moving object is removed, "
                        "if it is no longer visible in both stereo cameras.", 0);
    Mat movObjDir = (Mat_<double>(3,1) << 0.6, 0, 0.4);
    fs << "movObjDir" << movObjDir;
    cvWriteComment(*fs, "Relative velocity range of moving objects based on relative camera "
                        "velocity (relCamVelocity). \nThe actual relative velocity of every moving object is chosen "
                        "randomly between the given range. \n"
                        "Absolute_moving_object_velocity = chosen_relMovObjVel * absolute_camera_velocity; \nThe "
                        "absolute moving object velocity is multiplied with the movement direction vector "
                        "of the moving object to get the positional change from frame to frame. \n"
                        "Entered values must be between 0 and 100.0; Must be larger 0;", 0);
    fs << "relMovObjVelRange";
    fs << "{" << "first" << 0.5;
    fs << "second" << 2.0 << "}";
    cvWriteComment(*fs, "Minimal portion of correspondences on moving objects for removing them. \n"
                        "If the portion of visible correspondences drops below this value, the whole moving "
                        "object is removed. \nZero means, that the moving object is only removed if there is no "
                        "visible correspondence in the stereo pair. \nOne means, that a single missing correspondence "
                        "leads to deletion. Values between 0 and 1.0;", 0);
    fs << "minMovObjCorrPortion" << 0.2;
    cvWriteComment(*fs, "Relative portion of correspondences on moving object (relative to the full number of "
                        "correspondences of a stereo frame). \nThe number of correspondences is limited by "
                        "the size of objects visible in the images and the minimal distance between "
                        "correspondences. \nValue range: >0, <1.0", 0);
    fs << "CorrMovObjPort" << 0.16;
    cvWriteComment(*fs, "Minimum number of moving objects over the whole track. \nIf the number of moving "
                        "obects drops below this number during camera movement, as many new moving objects are "
                        "inserted until \"nrMovObjs\" is reached. \nIf 0, no new moving objects are inserted "
                        "if every preceding moving object is out of sight.", 0);
    fs << "minNrMovObjs" << 1;
    cvWriteComment(*fs, "Minimal and maximal percentage (0 to 1.0) of random distortion of the camera matrices "
                        "K1 & K2 based on their initial values \n(only the focal lengths and image centers are "
                        "randomly distorted). The distorted camera matrices are only generated for storing \n"
                        "them to output. For generating the ground truth (matches, ...), the correct "
                        "camera matrices are used.", 0);
    fs << "distortCamMat";
    fs << "{" << "first" << 0.02;
    fs << "second" << 0.1 << "}";
    cvWriteComment(*fs, "Image size of both stereo cameras", 0);
    fs << "imgSize";
    fs << "{" << "width" << 1280;
    fs << "height" << 720 << "}";
    cvWriteComment(*fs, "If 1, filtering occluded static 3D points during backprojection is enabled. \nOtherwise, "
                        "set this option to 0. Enabling this option significantly reduces the speed of calculating "
                        "3D scenes.", 0);
    fs << "filterOccluded3D" << 0;

    cvWriteComment(*fs, "---- Options for generating matches ----\n\n", 0);

    cvWriteComment(*fs, "Name of keypoint detector. The following types are supported: \n"
                        "FAST, MSER, ORB, BRISK, KAZE, AKAZE, STAR, MSD. \nIf non-free code is enabled "
                        "in the CMAKE project while building the code, SIFT and SURF are also available.", 0);
    fs << "keyPointType" << "BRISK";
    cvWriteComment(*fs, "Name of descriptor extractor. The following types are supported: \n"
                        "BRISK, ORB, KAZE, AKAZE, FREAK, DAISY, LATCH, BGM, BGM_HARD, BGM_BILINEAR, LBGM, "
                        "BINBOOST_64, BINBOOST_128, BINBOOST_256, VGG_120, VGG_80, VGG_64, VGG_48, RIFF, BOLD. \n"
                        "If non-free code is enabled in the CMAKE project while building the code, SIFT and SURF "
                        "are also available. AKAZE and KAZE descriptors might violate the used keypErrDistr "
                        "as they store specific information in the class_id "
                        "field of the keypoint which is not valid for a shifted keypoint position.", 0);
    fs << "descriptorType" << "FREAK";
    cvWriteComment(*fs, "Keypoint detector error (1) or error normal distribution (0). \nIf 1, the position "
                        "detected by the keypoint detector is used (which typically does not coincide with the "
                        "GT position. \nIf 0, an normal distributed (parameters from option keypErrDistr) "
                        "error is added to the GT position.", 0);
    fs << "keypPosErrType" << 0;
    cvWriteComment(*fs, "Keypoint error distribution (first=mean, second=standard deviation)", 0);
    fs << "keypErrDistr";
    fs << "{" << "first" << 0.1;
    fs << "second" << 1.2 << "}";
    cvWriteComment(*fs, "Noise (first=mean, second=standard deviation) on the image intensity (0-255) applied "
                        "on the image patches for descriptor calculation.", 0);
    fs << "imgIntNoise";
    fs << "{" << "first" << 10.0;
    fs << "second" << 15.0 << "}";
    cvWriteComment(*fs, "If 1, all PCL point clouds and necessary information to load a cam sequence "
                        "with correspondences are stored to disk. \nThis is useful if you want to load an "
                        "already generated 3D sequence later on and calculate a different type of descriptor \n"
                        "for the correspondences or if you want to use a different keypoint position "
                        "accuracy, ...", 0);
    fs << "storePtClouds" << 1;
    cvWriteComment(*fs, "If 1, the parameters and information are stored and read in XML format. "
                        "If 0, YAML format is used.", 0);
    fs << "rwXMLinfo" << 0;
    cvWriteComment(*fs, "If 1, the stored information and parameters are compressed (appends .gz to the "
                        "generated files. Otherwise, set this option to 0.", 0);
    fs << "compressedWrittenInfo" << 1;
    cvWriteComment(*fs, "If 1 and too less images to extract features are provided (resulting in too less keypoints), "
                        "only as many frames with GT matches are generated as keypoints are available. \n"
                        "Otherwise, set this option to 0.", 0);
    fs << "takeLessFramesIfLessKeyP" << 0;

    cvWriteComment(*fs, "Verbosity options (set them to 1 or 0).", 0);
    fs << "verbosity";
    fs << "{" << "SHOW_INIT_CAM_PATH" << 0;
    fs << "SHOW_BUILD_PROC_MOV_OBJ" << 0;
    fs << "SHOW_MOV_OBJ_DISTANCES" << 0;
    fs << "SHOW_MOV_OBJ_3D_PTS" << 0;
    fs << "SHOW_MOV_OBJ_CORRS_GEN" << 0;
    fs << "SHOW_BUILD_PROC_STATIC_OBJ" << 0;
    fs << "SHOW_STATIC_OBJ_DISTANCES" << 0;
    fs << "SHOW_STATIC_OBJ_CORRS_GEN" << 0;
    fs << "SHOW_STATIC_OBJ_3D_PTS" << 0;
    fs << "SHOW_MOV_OBJ_MOVEMENT" << 0;
    fs << "SHOW_BACKPROJECT_OCCLUSIONS_MOV_OBJ" << 0;
    fs << "SHOW_BACKPROJECT_OCCLUSIONS_STAT_OBJ" << 0;
    fs << "SHOW_BACKPROJECT_MOV_OBJ_CORRS" << 0;
    fs << "SHOW_STEREO_INTERSECTION" << 0;
    fs << "SHOW_COMBINED_CORRESPONDENCES" << 0;
    fs << "PRINT_WARNING_MESSAGES" << 1;
    fs << "SHOW_IMGS_AT_ERROR" << 0;
    fs << "SHOW_PLANES_FOR_HOMOGRAPHY" << 0;
    fs << "SHOW_WARPED_PATCHES" << 0;
    fs << "SHOW_PATCHES_WITH_NOISE" << 0 << "}";

    cvWriteComment(*fs, "Verbosity option for calculating the stereo camera configurations. "
                        "Prints the intermediate error values/results of the Levenberg Marquardt iterations. \n"
                        "Results of every LMverbose'th iteration are printed. Value range: 0-100. "
                        "Use 0 to disable.", 0);
    fs << "LMverbose" << 0;

    cvWriteComment(*fs, "If 1, extrinsic stereo parameters are also accepted for further processing if they do "
                        "not completely fulfill the user specified values (like image overlap area). "
                        "Otherwise, set this value to 0. In this case, you will be asked if you want to accept the "
                        "shown stereo parameters in case the LM algorithm was not able to find a good "
                        "solution.", 0);
    fs << "acceptBadStereoPars" << 0;

    fs.release();

    return true;
}

bool loadConfigFile(const std::string &filename,
                    StereoSequParameters &sequPars,
                    GenMatchSequParameters &matchPars,
                    stereoExtrPars &stereoPars,
                    additionalSequencePars &addPars){
    sequPars = StereoSequParameters();
    matchPars = GenMatchSequParameters();
    stereoPars = stereoExtrPars();
    addPars = additionalSequencePars();
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Failed to open " << filename << endl;
        return false;
    }

    int tmp = 0;
    fs["nrTotalFrames"] >> tmp;
    if((tmp <= 0) || (tmp > 10000)){
        cerr << "Incorrect nrTotalFrames." << endl;
        return false;
    }
    sequPars.nTotalNrFrames = (size_t)tmp;
    fs["nrStereoConfigs"] >> stereoPars.nrStereoConfigs;
    fs["txChangeFRate"] >> stereoPars.txChangeFRate;
    fs["tyChangeFRate"] >> stereoPars.tyChangeFRate;
    fs["tzChangeFRate"] >> stereoPars.tzChangeFRate;
    fs["rollChangeFRate"] >> stereoPars.rollChangeFRate;
    fs["pitchChangeFRate"] >> stereoPars.pitchChangeFRate;
    fs["yawChangeFRate"] >> stereoPars.yawChangeFRate;
    fs["txLinChangeVal"] >> stereoPars.txLinChangeVal;
    fs["tyLinChangeVal"] >> stereoPars.tyLinChangeVal;
    fs["tzLinChangeVal"] >> stereoPars.tzLinChangeVal;
    fs["rollLinChangeVal"] >> stereoPars.rollLinChangeVal;
    fs["pitchLinChangeVal"] >> stereoPars.pitchLinChangeVal;
    fs["yawLinChangeVal"] >> stereoPars.yawLinChangeVal;
    fs["txStartVal"] >> stereoPars.txStartVal;
    fs["tyStartVal"] >> stereoPars.tyStartVal;
    fs["tzStartVal"] >> stereoPars.tzStartVal;
    fs["rollStartVal"] >> stereoPars.rollStartVal;
    fs["pitchStartVal"] >> stereoPars.pitchStartVal;
    fs["yawStartVal"] >> stereoPars.yawStartVal;
    FileNode n = fs["txRange"];
    double first_dbl = 0, second_dbl = 0;
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    stereoPars.txRange = make_pair(first_dbl, second_dbl);
    n = fs["tyRange"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    stereoPars.tyRange = make_pair(first_dbl, second_dbl);
    n = fs["tzRange"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    stereoPars.tzRange = make_pair(first_dbl, second_dbl);
    n = fs["rollRange"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    stereoPars.rollRange = make_pair(first_dbl, second_dbl);
    n = fs["pitchRange"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    stereoPars.pitchRange = make_pair(first_dbl, second_dbl);
    n = fs["yawRange"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    stereoPars.yawRange = make_pair(first_dbl, second_dbl);
    fs["txVariable"] >> stereoPars.txVariable;
    fs["tyVariable"] >> stereoPars.tyVariable;
    fs["tzVariable"] >> stereoPars.tzVariable;
    fs["rollVariable"] >> stereoPars.rollVariable;
    fs["pitchVariable"] >> stereoPars.pitchVariable;
    fs["yawVariable"] >> stereoPars.yawVariable;
    fs["useSpecificCamPars"] >> stereoPars.useSpecificCamPars;
    n = fs["specificCamPars"];
    if (n.type() != FileNode::SEQ) {
        cerr << "specificCamPars is not a sequence! FAIL" << endl;
        return false;
    }
    stereoPars.specialCamPars.clear();
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it) {
        FileNode n1 = *it;
        specificStereoPars stP;
        n1["R"] >> stP.R;
        n1["t"] >> stP.t;
        n1["K1"] >> stP.K1;
        n1["K2"] >> stP.K2;
        stereoPars.specialCamPars.emplace_back(stP);
    }
    /*for(auto& i : stereoPars.specialCamPars){
        for (int j = 0; j < 9; ++j) {
            cout << "R_" << j << " = " << i.R.at<double>(j) << endl;
            cout.flush();
        }
    }*/
    fs["imageOverlap"] >> stereoPars.imageOverlap;

    n = fs["inlRatRange"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    sequPars.inlRatRange = make_pair(first_dbl, second_dbl);
    fs["inlRatChanges"] >> sequPars.inlRatChanges;
    n = fs["truePosRange"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    sequPars.truePosRange = make_pair(first_dbl, second_dbl);
    fs["truePosChanges"] >> sequPars.truePosChanges;
    fs["minKeypDist"] >> sequPars.minKeypDist;
    n = fs["corrsPerDepth"];
    n["near"] >> sequPars.corrsPerDepth.near;
    if(sequPars.corrsPerDepth.near < 0){
        cerr << "Invalid parameter corrsPerDepth (near)." << endl;
        return false;
    }
    n["mid"] >> sequPars.corrsPerDepth.mid;
    if(sequPars.corrsPerDepth.mid < 0){
        cerr << "Invalid parameter corrsPerDepth (mid)." << endl;
        return false;
    }
    n["far"] >> sequPars.corrsPerDepth.far;
    if(sequPars.corrsPerDepth.far < 0){
        cerr << "Invalid parameter corrsPerDepth (far)." << endl;
        return false;
    }

    n = fs["corrsPerRegion"];
    if (n.type() != FileNode::SEQ) {
        cerr << "corrsPerRegion is not a sequence! FAIL" << endl;
        return false;
    }
    sequPars.corrsPerRegion.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        Mat m;
        it >> m;
        if((m.rows != 3) || (m.cols != 3)){
            cerr << "Invalid parameter corrsPerRegion." << endl;
            return false;
        }
        double sumregs = 0;
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.cols; ++j) {
                if(m.at<double>(i,j) < 0){
                    cerr << "Invalid parameter corrsPerRegion." << endl;
                    return false;
                }
                sumregs += m.at<double>(i,j);
            }
        }
        if(nearZero(sumregs)){
            cerr << "Invalid parameter corrsPerRegion." << endl;
            return false;
        }
        sequPars.corrsPerRegion.push_back(m.clone());
    }

    fs["corrsPerRegRepRate"] >> tmp;
    if(tmp < 0){
        cerr << "Invalid parameter corrsPerRegRepRate." << endl;
        return false;
    }
    sequPars.corrsPerRegRepRate = (size_t) tmp;

    n = fs["depthsPerRegion"];
    if (n.type() != FileNode::SEQ) {
        cerr << "depthsPerRegion is not a sequence! FAIL" << endl;
        return false;
    }
    sequPars.depthsPerRegion = vector<vector<depthPortion>>(3, vector<depthPortion>(3));
    it = n.begin(), it_end = n.end();
    size_t idx = 0, x = 0, y = 0;
    for (; it != it_end; ++it) {
        y = idx / 3;
        x = idx % 3;

        FileNode n1 = *it;
        n1["near"] >> sequPars.depthsPerRegion[y][x].near;
        if(sequPars.depthsPerRegion[y][x].near < 0){
            cerr << "Invalid parameter depthsPerRegion (near)." << endl;
            return false;
        }
        n1["mid"] >> sequPars.depthsPerRegion[y][x].mid;
        if(sequPars.depthsPerRegion[y][x].mid < 0){
            cerr << "Invalid parameter depthsPerRegion (mid)." << endl;
            return false;
        }
        n1["far"] >> sequPars.depthsPerRegion[y][x].far;
        if(sequPars.depthsPerRegion[y][x].far < 0){
            cerr << "Invalid parameter depthsPerRegion (far)." << endl;
            return false;
        }
        if(idx > 8){
            cerr << "Incorrect # of entries in depthsPerRegion." << endl;
            return false;
        }
        idx++;
    }

    n = fs["nrDepthAreasPReg"];
    if (n.type() != FileNode::SEQ) {
        cerr << "nrDepthAreasPReg is not a sequence! FAIL" << endl;
        return false;
    }
    sequPars.nrDepthAreasPReg = vector<vector<pair<size_t, size_t>>>(3, vector<pair<size_t, size_t>>(3));
    int first_int = 0, second_int = 0;
    it = n.begin(), it_end = n.end();
    idx = 0;
    for (; it != it_end; ++it) {
        y = idx / 3;
        x = idx % 3;

        FileNode n1 = *it;
        n1["first"] >> first_int;
        n1["second"] >> second_int;
        if((first_int < 1) || ((second_int < 1))){
            cerr << "Invalid parameter nrDepthAreasPReg." << endl;
            return false;
        }
        sequPars.nrDepthAreasPReg[y][x] = make_pair((size_t) first_int, (size_t) second_int);
        if(idx > 8){
            cerr << "Incorrect # of entries in nrDepthAreasPReg." << endl;
            return false;
        }
        idx++;
    }

    n = fs["camTrack"];
    if (n.type() != FileNode::SEQ) {
        cerr << "camTrack is not a sequence! FAIL" << endl;
        return false;
    }
    sequPars.camTrack.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        Mat m;
        it >> m;
        sequPars.camTrack.emplace_back(m.clone());
    }

    fs["relCamVelocity"] >> sequPars.relCamVelocity;

    fs["nrMovObjs"] >> tmp;
    if(tmp < 0){
        cerr << "Invalid parameter nrMovObjs." << endl;
        return false;
    }
    sequPars.nrMovObjs = (size_t) tmp;

    fs["startPosMovObjs"] >> sequPars.startPosMovObjs;

    n = fs["relAreaRangeMovObjs"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    sequPars.relAreaRangeMovObjs = make_pair(first_dbl, second_dbl);

    n = fs["movObjDepth"];
    if (n.type() != FileNode::SEQ) {
        cerr << "camTrack is not a sequence! FAIL" << endl;
        return false;
    }
    sequPars.movObjDepth.clear();
    it = n.begin(), it_end = n.end();
    while (it != it_end) {
        it >> tmp;
        if((tmp != (int)depthClass::NEAR) && (tmp != (int)depthClass::MID) && (tmp != (int)depthClass::FAR)){
            cerr << "Values of depth classes are invalid." << endl;
            return false;
        }
        sequPars.movObjDepth.push_back((depthClass) tmp);
    }

    fs["movObjDir"] >> sequPars.movObjDir;

    n = fs["relMovObjVelRange"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    sequPars.relMovObjVelRange = make_pair(first_dbl, second_dbl);

    fs["minMovObjCorrPortion"] >> sequPars.minMovObjCorrPortion;

    fs["CorrMovObjPort"] >> sequPars.CorrMovObjPort;

    fs["minNrMovObjs"] >> tmp;
    if(tmp < 0){
        cerr << "Invalid parameter minNrMovObjs." << endl;
        return false;
    }
    sequPars.minNrMovObjs = (size_t) tmp;

    n = fs["distortCamMat"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    sequPars.distortCamMat = make_pair(first_dbl, second_dbl);

    randomTrackPars randTrackP = randomTrackPars();
    ellipsoidTrackPars ellipsTrackP = ellipsoidTrackPars();

    fs["corrsPerRegionRandInit"] >> addPars.corrsPerRegionRandInit;
    fs["trackOption"] >> addPars.trackOption;
    n = fs["ellipsoidTrack"];
    n["xDirection"] >> ellipsTrackP.xDirection;
    n["xzExpansion"] >> ellipsTrackP.xzExpansion;
    n["xyExpansion"] >> ellipsTrackP.xyExpansion;
    FileNode n1 = n["thetaRange"];
    n1["min"] >> first_dbl;
    n1["max"] >> second_dbl;
    ellipsTrackP.thetaRange = make_pair(first_dbl, second_dbl);

    addPars.ellipsoidTrack = ellipsTrackP;
    fs["maxTrackElements"] >> addPars.maxTrackElements;

    n = fs["randomTrack"];
    n["xDirection"] >> randTrackP.xDirection;
    n1 = n["xzDirectionRange"];
    n1["first"] >> first_dbl;
    n1["second"] >> second_dbl;
    randTrackP.xzDirectionRange = make_pair(first_dbl, second_dbl);
    n1 = n["xyDirectionRange"];
    n1["first"] >> first_dbl;
    n1["second"] >> second_dbl;
    randTrackP.xyDirectionRange = make_pair(first_dbl, second_dbl);
    n["allowedChangeSD"] >> randTrackP.allowedChangeSD;

    addPars.randomTrack = randTrackP;
    fs["rollCamTrack"] >> addPars.rollCamTrack;
    fs["pitchCamTrack"] >> addPars.pitchCamTrack;
    fs["yawCamTrack"] >> addPars.yawCamTrack;
    fs["filterOccluded3D"] >> addPars.filterOccluded3D;

    n = fs["imgSize"];
    n["width"] >> first_int;
    n["height"] >> second_int;
    stereoPars.imgSize = Size(first_int, second_int);

    fs["keyPointType"] >> matchPars.keyPointType;
    fs["descriptorType"] >> matchPars.descriptorType;
    fs["keypPosErrType"] >> matchPars.keypPosErrType;
    n = fs["keypErrDistr"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    matchPars.keypErrDistr = make_pair(first_dbl, second_dbl);
    n = fs["imgIntNoise"];
    n["first"] >> first_dbl;
    n["second"] >> second_dbl;
    matchPars.imgIntNoise = make_pair(first_dbl, second_dbl);
    fs["storePtClouds"] >> matchPars.storePtClouds;
    fs["rwXMLinfo"] >> matchPars.rwXMLinfo;
    fs["compressedWrittenInfo"] >> matchPars.compressedWrittenInfo;
    fs["takeLessFramesIfLessKeyP"] >> matchPars.takeLessFramesIfLessKeyP;

    n = fs["verbosity"];
    bool tmp_bool = false;
    addPars.verbose = 0;
    n["SHOW_INIT_CAM_PATH"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_INIT_CAM_PATH;
    n["SHOW_BUILD_PROC_MOV_OBJ"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_BUILD_PROC_MOV_OBJ;
    n["SHOW_MOV_OBJ_DISTANCES"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_MOV_OBJ_DISTANCES;
    n["SHOW_MOV_OBJ_3D_PTS"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_MOV_OBJ_3D_PTS;
    n["SHOW_MOV_OBJ_CORRS_GEN"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_MOV_OBJ_CORRS_GEN;
    n["SHOW_BUILD_PROC_STATIC_OBJ"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_BUILD_PROC_STATIC_OBJ;
    n["SHOW_STATIC_OBJ_DISTANCES"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_STATIC_OBJ_DISTANCES;
    n["SHOW_STATIC_OBJ_CORRS_GEN"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_STATIC_OBJ_CORRS_GEN;
    n["SHOW_STATIC_OBJ_3D_PTS"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_STATIC_OBJ_3D_PTS;
    n["SHOW_MOV_OBJ_MOVEMENT"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_MOV_OBJ_MOVEMENT;
    n["SHOW_BACKPROJECT_OCCLUSIONS_MOV_OBJ"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_BACKPROJECT_OCCLUSIONS_MOV_OBJ;
    n["SHOW_BACKPROJECT_OCCLUSIONS_STAT_OBJ"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_BACKPROJECT_OCCLUSIONS_STAT_OBJ;
    n["SHOW_BACKPROJECT_MOV_OBJ_CORRS"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_BACKPROJECT_MOV_OBJ_CORRS;
    n["SHOW_STEREO_INTERSECTION"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_STEREO_INTERSECTION;
    n["SHOW_COMBINED_CORRESPONDENCES"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_COMBINED_CORRESPONDENCES;
    n["PRINT_WARNING_MESSAGES"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= PRINT_WARNING_MESSAGES;
    n["SHOW_IMGS_AT_ERROR"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_IMGS_AT_ERROR;
    n["SHOW_PLANES_FOR_HOMOGRAPHY"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_PLANES_FOR_HOMOGRAPHY;
    n["SHOW_WARPED_PATCHES"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_WARPED_PATCHES;
    n["SHOW_PATCHES_WITH_NOISE"] >> tmp_bool;
    if(tmp_bool) addPars.verbose |= SHOW_PATCHES_WITH_NOISE;

    fs["LMverbose"] >> addPars.LMverbose;
    fs["acceptBadStereoPars"] >> addPars.acceptBadStereoPars;

    fs.release();

    return true;
}

/** @function main */
int main( int argc, char* argv[])
{
	ArgvParser cmd;
	SetupCommandlineParser(cmd, argc, argv);
	int err = startEvaluation(cmd);
	if(err){
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

double getRandDoubleVal(std::default_random_engine rand_generator, double lowerBound, double upperBound)
{
	rand_generator = std::default_random_engine((unsigned int)std::rand());
	std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
	return distribution(rand_generator);
}