#ifndef CONFIGPARAMS_H
#define CONFIGPARAMS_H

#include <string>
#include <iostream>
//#include "ConfigFileReader.h"

namespace USACConfig
{
	enum RandomSamplingMethod	  {SAMP_UNIFORM, SAMP_PROSAC, SAMP_UNIFORM_MM, SAMP_MODEL_COMPLETE};
	enum VerifMethod			  {VERIF_STANDARD, VERIF_SPRT};
	enum LocalOptimizationMethod  {LO_NONE, LO_LOSAC};
	//enum PoseTransformationType     { TRANS_ESSENTIAL, TRANS_ROTATION, TRANS_TRANSLATION, TRANS_NO_MOTION };

	// common USAC parameters
	struct Common
	{
		// change these parameters according to the problem of choice
		Common(): confThreshold		     (0.95), 
			      minSampleSize		     (7),
				  minSampleSizeDegenerate(0),
				  inlierThreshold		 (0.001),
				  maxHypotheses		     (100000),
  				  maxSolutionsPerSample  (3),
				  numDataPoints          (0),
				  numDataPointsDegenerate(0),
				  prevalidateSample 	 (false),
				  prevalidateModel	     (false),
				  testDegeneracy	 	 (false),
				  randomSamplingMethod   (SAMP_UNIFORM),
				  verifMethod			 (VERIF_STANDARD),
				  localOptMethod         (LO_NONE)//,
				  //posetype				 (TRANS_ESSENTIAL)
		{}
		double				    confThreshold;
		unsigned int		    minSampleSize;  
		unsigned int			minSampleSizeDegenerate;
		double				    inlierThreshold;   
		unsigned int		    maxHypotheses;	
		unsigned int		    maxSolutionsPerSample; 
		unsigned int		    numDataPoints;
		unsigned int			numDataPointsDegenerate;
		bool					prevalidateSample;
		bool					prevalidateModel;
		bool					testDegeneracy;
		RandomSamplingMethod    randomSamplingMethod;
		VerifMethod			    verifMethod;
		LocalOptimizationMethod localOptMethod;
		//PoseTransformationType  posetype;
	};

	// PROSAC parameters
	struct Prosac
	{
		Prosac(): maxSamples 		    (200000),
				  beta                  (0.05),
				  nonRandConf           (0.95),
				  minStopLen	        (20),
				  //sortedPointsFile		(""),		// leave blank if not reading from file
				  sortedPointIndices    (NULL)		// this should point to an array of point indices
													// sorted in decreasing order of quality scores
		{}
		unsigned int  maxSamples;
		double		  beta;
		double        nonRandConf;
		unsigned int  minStopLen;
		//std::string   sortedPointsFile;
		unsigned int* sortedPointIndices;
	};

	// SPRT parameters
	struct Sprt
	{
		Sprt(): tM      (200.0),
				mS	    (2.38),
				delta   (0.05),
				epsilon (0.2)
		{}
		double tM;
		double mS;
		double delta;
		double epsilon;
	};

	// LOSAC parameters
	struct Losac
	{
		Losac(): innerSampleSize		  (14),
		 		 innerRansacRepetitions   (10),
				 thresholdMultiplier	  (2.0),
				 numStepsIterative	      (4)
		{}
		unsigned int innerSampleSize;
		unsigned int innerRansacRepetitions;
		double		 thresholdMultiplier;
		unsigned int numStepsIterative;
	};

}

// main configuration struct that is passed to USAC
class ConfigParams
{
public:
	// to be overridden to read in model specific data
	//virtual bool initParamsFromConfigFile(std::string& configFilePath);

	// function to read in common usac parameters from config file
	//bool initUSACParamsFromConfigFile(const ConfigFileReader& cfr);

	ConfigParams(USACConfig::Common common_,
		USACConfig::Prosac prosac_,
		USACConfig::Sprt sprt_,
		USACConfig::Losac losac_) :
		common(common_),
		prosac(prosac_),
		sprt(sprt_),
		losac(losac_) {
		// verify parameter values 
		if (common.confThreshold < 0 || common.confThreshold > 1)
		{
			std::cout << "RANSAC confidence value must be between 0 and 1. Setting it to 0.95!" << std::endl;
			common.confThreshold = 0.95;
		}
		
		if ((common.randomSamplingMethod != USACConfig::SAMP_UNIFORM) &&
			(common.randomSamplingMethod != USACConfig::SAMP_PROSAC) &&
			(common.randomSamplingMethod != USACConfig::SAMP_MODEL_COMPLETE))
		{
			std::cerr << "Random sampling method " << common.randomSamplingMethod << " not recognized. Setting it to USACConfig::SAMP_UNIFORM" << std::endl;
			common.randomSamplingMethod = USACConfig::SAMP_UNIFORM;
		}
				
		if ((common.verifMethod != USACConfig::VERIF_STANDARD) &&
			(common.verifMethod != USACConfig::VERIF_SPRT))
		{
			std::cerr << "Verification method " << common.verifMethod << " not recognized. Setting it to USACConfig::VERIF_STANDARD" << std::endl;
			common.verifMethod = USACConfig::VERIF_STANDARD;
		}
				
		if ((common.localOptMethod != USACConfig::LO_NONE) &&
			(common.localOptMethod != USACConfig::LO_LOSAC))
		{
			std::cerr << "Local optimization method " << common.localOptMethod << " not recognized. Setting it to USACConfig::LO_NONE" << std::endl;
			common.localOptMethod = USACConfig::LO_NONE;
		}
	}

	USACConfig::Common     common;
	USACConfig::Prosac     prosac;
	USACConfig::Sprt       sprt;
	USACConfig::Losac      losac;
};


#endif