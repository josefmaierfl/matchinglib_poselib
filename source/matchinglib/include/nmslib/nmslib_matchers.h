/**********************************************************************************************************
FILE: nmslib_matchers.h

PLATFORM: Windows 7, MS Visual Studio 2014, OpenCV 3.2

CODE: C++

AUTOR: Josef Maier, AIT Austrian Institute of Technology

DATE: February 2017

LOCATION: TechGate Vienna, Donau-City-Strasse 1, 1220 Vienna

VERSION: 1.0

DISCRIPTION: This file provides functionalities for matching features using different algorithms of the NMSLIB
**********************************************************************************************************/

#pragma once

#include "opencv2/core/core.hpp"
#include "glob_includes.h"

#include <thread>
#include <mutex>

#include "space.h"
#include "params.h"
#include "spacefactory.h"
#include "index.h"
#include "methodfactory.h"
#include "query_creator.h"
#include "utils.h"
#include "init.h"
#include "knnqueue.h"
#include "knnquery.h"
#include "query_creator.h"
#include "query.h"
#include "object.h"
#include "method/small_world_rand.h"

	using namespace cv;
	using namespace std;

	namespace matchinglib
	{

		/* --------------------------- Defines --------------------------- */

		template <typename QueryType, typename QueryCreatorType, typename dist_t>
		struct  ThreadParams {
			ThreadParams(
				mutex&								UpdateStat,
				similarity::Space<dist_t>&			space,
				similarity::ObjectVector&			queryobjects,
				unsigned							ThreadQty,
				unsigned							QueryPart,
				const QueryCreatorType&				QueryCreator,
				const similarity::Index<dist_t>*    Method) :
				UpdateStat_(UpdateStat),
				space_(space),
				queryobjects_(queryobjects),
				ThreadQty_(ThreadQty),
				QueryPart_(QueryPart),
				QueryCreator_(QueryCreator),
				Method_(Method)
			{}

			mutex&								UpdateStat_;
			similarity::Space<dist_t>&			space_;
			similarity::ObjectVector&			queryobjects_;
			unsigned							ThreadQty_;
			unsigned							QueryPart_;
			const QueryCreatorType&				QueryCreator_;
			const similarity::Index<dist_t>*    Method_;

			vector<size_t>						queryIds;
			vector<unique_ptr<QueryType>>		queries; // queries with results
		};

		template <typename QueryType, typename QueryCreatorType, typename dist_t>
		struct SearchThread {
			void operator ()(ThreadParams<QueryType, QueryCreatorType, dist_t>& prm) {
				size_t numquery = prm.queryobjects_.size();

				unsigned QueryPart = prm.QueryPart_;
				unsigned ThreadQty = prm.ThreadQty_;

				for (size_t q = 0; q < numquery; ++q) {
					if ((q % ThreadQty) == QueryPart) {
						unique_ptr<QueryType> query(prm.QueryCreator_(prm.space_,
							prm.queryobjects_[q]));
						prm.Method_->Search(query.get());

						{
							lock_guard<mutex> g(prm.UpdateStat_);

							prm.queryIds.push_back(q);

							//query.get()->

							prm.queries.push_back(std::move(query));
						}
					}
				}
			}
		};

		/* --------------------- Function prototypes --------------------- */


		/* --------------------- Functions --------------------- */

		template <typename dist_t>
		similarity::Object* CreateObjFromVect1(similarity::IdType id, similarity::LabelType label, const vector<dist_t>& InpVect) {
			return new similarity::Object(id, label, InpVect.size() * sizeof(dist_t), &InpVect[0]);
		};

		/* This is a wrapper function for the NMSLIB
		*
		* Mat descrL              Input  -> Descriptors within the first or left image
		* Mat descrR              Input  -> Descriptors within the second or right image
		* vector<DMatch> matches  Output -> Matches
		* string methodStr        Input  -> Name of the NMSLIB matching method. See the NMSLIB documentation for options.
		* string spaceStr         Input  -> The name of the distance function like l2 or bit_hamming. See the NMSLIB documentation for options.
		* string indexParsStr     Input  -> The parameters for generating the search index. See the NMSLIB documentation for options.
		* string queryTimeParsStr Input  -> The parameters for searching. See the NMSLIB documentation for options.
		* bool ratioTest          Input  -> If true [Default=true], a ratio test is performed on the results.
		* unsigned ThreadQty      Input  -> The number of threads used for searching. A higher number must not lead to smaller search times.
		*
		* Return value:           0:     Everything ok
		*                  -1:     Too less features detected
		*                  -2:     Error creating feature detector
		*                  -3:     No such feature detector
		*/
		template <typename dist_t>
		int nmslibMatching(cv::Mat const& descrL,
			cv::Mat const& descrR,
			std::vector<cv::DMatch> & matches,
			std::string methodStr,
			std::string spaceStr,
			std::string indexParsStr,
			std::string queryTimeParsStr,
			bool ratioTest = true,
			unsigned ThreadQty = 0)
		{

			string SpaceType;
			similarity::ObjectVector queryobjects_;
			similarity::ObjectVector dataobjects_;
			vector<dist_t> vec;//holds one descriptor (each element holds one dimension)
			similarity::IdType id = 0;//index of deskriptors (int) from 0 to the total number of features -1
			similarity::LabelType label = 0;//can be left empty (int)
			shared_ptr<similarity::AnyParams> SpaceParams;
			shared_ptr<similarity::AnyParams> IndexParams;
			vector<shared_ptr<similarity::AnyParams>> vQueryTimeParams;
			const size_t K = 2;//knn = number of results per query -> BE CAREFUL: If the number is choosen larger than 2, the results for each query must be sorted
			unsigned ThreadQty_ = thread::hardware_concurrency();// 8;//Number of search threads
			float eps = 0;
		

			if (ThreadQty) ThreadQty_ = ThreadQty;
			ThreadQty_ = ThreadQty_ <= 0 ? 1 :ThreadQty_;

			similarity::ToLower(spaceStr);//the used space like "l2", "l1", "bit_hamming", ...
			similarity::ToLower(methodStr);//name of the matching-method
			
			similarity::initLibrary(LIB_LOGNONE,'\0');
			/*string logfilename = "C:\\work\\nmslib_log.txt";
			similarity::initLibrary(LIB_LOGFILE, logfilename.c_str());*/

			//Get the correct format for the space parameter, like "l2"
			{
				vector<string>     desc;
				similarity::ParseSpaceArg(spaceStr, SpaceType, desc);
				SpaceParams = shared_ptr<similarity::AnyParams>(new similarity::AnyParams(desc));
			}

			//Generate the space-object
			unique_ptr<similarity::Space<dist_t>> space(similarity::SpaceFactoryRegistry<dist_t>::
				Instance().CreateSpace(SpaceType, *SpaceParams));

			if (NULL == space.get()) {
				PREPARE_RUNTIME_ERR(err) << "Cannot create space: '" << SpaceType;
				THROW_RUNTIME_ERR(err);
			}

			//Bring the (train) data for indexing into the correct format
			if (descrR.type() == CV_8U)
			{
				int halfcols = descrR.cols / 2;
				halfcols *= 2;
				halfcols -= descrR.cols;
				for (int i = 0; i < descrR.rows; i++)
				{
					vec.clear();
					for (int j = 0; j < descrR.cols - 1; j = j + 2)
					{
						int twoVals = (int)(descrR.at<unsigned char>(i, j)) << 8;
						twoVals |= (int)(descrR.at<unsigned char>(i, j + 1));
						vec.push_back(twoVals);
					}
					if(halfcols)
						vec.push_back((int)(descrR.at<unsigned char>(i, descrR.cols - 1)));
					dataobjects_.push_back(CreateObjFromVect1(id, label, vec));
					id++;
				}
			}
			else
			{
				for (int i = 0; i < descrR.rows; i++)
				{
					vec.clear();
					for (int j = 0; j < descrR.cols; j++)
					{
						vec.push_back(descrR.at<dist_t>(i, j));
					}
					dataobjects_.push_back(CreateObjFromVect1(id, label, vec));
					id++;
				}
			}

			//Get the correct format for the index parameters, like indexParsStr="chunkBucket=1,bucketSize=10"
			{
				vector<string> desc;
				similarity::ParseArg(indexParsStr, desc);
				IndexParams = shared_ptr<similarity::AnyParams>(new similarity::AnyParams(desc));
			}

			//Generate the correct index-object
			shared_ptr<similarity::Index<dist_t>> IndexPtr(
				similarity::MethodFactoryRegistry<dist_t>::Instance().
				CreateMethod(false /* don't print progress */,
					methodStr,
					SpaceType, *space,
					dataobjects_)
			);

			//Create the index (structure of the tree)
			IndexPtr->CreateIndex(*IndexParams);

			//Get the correct format for the query-time parameters, like queryTimeParsStr="alphaLeft=2,alphaRight=2"
			{
				vector<string> desc;
				similarity::ParseArg(queryTimeParsStr, desc);
				vQueryTimeParams.push_back(shared_ptr<similarity::AnyParams>(new similarity::AnyParams(desc)));
			}

			const similarity::AnyParams& qtp = *vQueryTimeParams[0];
			IndexPtr->SetQueryTimeParams(qtp);

			////Bring the query data for indexing into the correct format
			id = 0;
			if (descrL.type() == CV_8U)
			{
				int halfcols = descrL.cols / 2;
				halfcols *= 2;
				halfcols -= descrL.cols;
				for (int i = 0; i < descrL.rows; i++)
				{
					vec.clear();
					for (int j = 0; j < descrL.cols - 1; j = j + 2)
					{
						int twoVals = (int)(descrL.at<unsigned char>(i, j)) << 8;
						twoVals |= (int)(descrL.at<unsigned char>(i, j + 1));
							vec.push_back(twoVals);
					}
					if (halfcols)
						vec.push_back((int)(descrL.at<unsigned char>(i, descrL.cols - 1)));
					queryobjects_.push_back(CreateObjFromVect1(id, label, vec));
					id++;
				}
			}
			else
			{
				for (int i = 0; i < descrL.rows; i++)
				{
					vec.clear();
					for (int j = 0; j < descrL.cols; j++)
					{
						vec.push_back(descrL.at<dist_t>(i, j));
					}
					queryobjects_.push_back(CreateObjFromVect1(id, label, vec));
					id++;
				}
			}

			//Set number of knn
			similarity::KNNCreator<dist_t>  cr(K, eps);

			space.get()->SetQueryPhase();

			vector<ThreadParams<similarity::KNNQuery<dist_t>, similarity::KNNCreator<dist_t>, dist_t>*> ThreadParamsVar(ThreadQty_);
			vector<thread> Threads(ThreadQty_);
			
			//Delets the thread-paramters that are generated with "new" in the next phase, when this object is released
			similarity::AutoVectDel<ThreadParams<similarity::KNNQuery<dist_t>, similarity::KNNCreator<dist_t>, dist_t>> DelThreadParams(ThreadParamsVar);
			mutex UpdateStat;

			//Prepare the data for (mutithreaded) searching
			for (unsigned QueryPart = 0; QueryPart < ThreadQty_; ++QueryPart) {
				ThreadParamsVar[QueryPart] = new ThreadParams<similarity::KNNQuery<dist_t>, similarity::KNNCreator<dist_t>, dist_t>(
					UpdateStat,
					*space.get(),
					queryobjects_,
					ThreadQty_,
					QueryPart,
					cr,
					IndexPtr.get());
			}

			//Start the (multithreaded) search
			if (ThreadQty_> 1) {
				for (unsigned QueryPart = 0; QueryPart < ThreadQty_; ++QueryPart) {
					Threads[QueryPart] = std::thread(SearchThread<similarity::KNNQuery<dist_t>, similarity::KNNCreator<dist_t>, dist_t>(),
						ref(*ThreadParamsVar[QueryPart]));
				}
				for (unsigned QueryPart = 0; QueryPart < ThreadQty_; ++QueryPart) {
					Threads[QueryPart].join();
				}
			}
			else {
				CHECK(ThreadQty_ == 1);
				SearchThread<similarity::KNNQuery<dist_t>, similarity::KNNCreator<dist_t>, dist_t>()(*ThreadParamsVar[0]);
			}

			//Extract the matches from NMSLIB's data structure
			vector<vector<cv::DMatch>> idVect;
			for (unsigned QueryPart = 0; QueryPart < ThreadQty_; ++QueryPart) {
				unsigned querySize = ThreadParamsVar[QueryPart]->queries.size();
				for (unsigned queryres = 0; queryres < querySize; queryres++)
				{
					cv::DMatch tmpm;
					vector<DMatch> distSorted;
					similarity::KNNQueue<dist_t>* help = ThreadParamsVar[QueryPart]->queries[queryres]->Result()->Clone();
					const similarity::Object* help1 = help->TopObject();
					tmpm.queryIdx = ThreadParamsVar[QueryPart]->queryIds[queryres];
					tmpm.trainIdx = help1->id();
					tmpm.distance = help->TopDistance();
					distSorted.push_back(tmpm);
					while ((help->Size() > 1))
					{
						help->Pop();
						help1 = help->TopObject();
						tmpm.trainIdx = help1->id();
						tmpm.distance = help->TopDistance();
						distSorted.push_back(tmpm);
					}
					delete help;
					sort(distSorted.begin(), distSorted.end(),
						[](DMatch first, DMatch second) {return first.distance < second.distance; });
					idVect.push_back(distSorted);
				}
			}

			/*for (unsigned QueryPart = 0; QueryPart < ThreadQty_; ++QueryPart) 
			{
				delete ThreadParamsVar[QueryPart];
			}*/

			for (int i = 0; i < dataobjects_.size(); i++)
			{
				delete (dataobjects_[i]);
			}
			for (int i = 0; i < queryobjects_.size(); i++)
			{
				delete (queryobjects_[i]);
			}

			//Ratio test
			if (ratioTest)
			{
				for (size_t q = 0; q < idVect.size(); q++)
				{
					if (idVect[q][0].distance < (0.75f * idVect[q][1].distance))
					{
						matches.push_back(idVect[q][0]);
					}
				}
			}
			else
			{
				for (size_t q = 0; q < idVect.size(); q++)
				{
					matches.push_back(idVect[q][0]);
				}
			}

			if (ThreadQty_ > 1)
			{
				sort(matches.begin(), matches.end(),
					[](DMatch first, DMatch second) {return first.queryIdx < second.queryIdx; });
			}

			return 0;
		}

	} // namepace matchinglib
