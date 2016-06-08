// Copyright (C) 2013 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)


/****************************************************************************
 *							IMPORTANT NOTICE								*
 ****************************************************************************
 *																			*
 * This code was completele rewritten by Josef Maier						*
 * (josef.maier.fl@ait.ac.at) and might not be usable anymore in			*
 * conjuction with the rest of this library. Most interfaces are the same	*
 * but the functionality is completely different to the original code. This	*
 * was necessary, because the original code was not correctly working in	*
 * terms of the intended algorithm.											*
 *																			*
 * For further information please contact Josef Maier, AIT Austrian			*
 * Institute of Technology.													*
 *																			*
 ****************************************************************************/

#pragma once

#ifndef THEIA_SOLVERS_ARRSAC_H_
#define THEIA_SOLVERS_ARRSAC_H_

//#include <chrono>
//#include <random>

#include <algorithm>
#include <vector>

#include "sequential_probability_ratio.h"
#include "estimator.h"
#include "sample_consensus_estimator.h"
#include "random_sampler.h"
#include "prosac_sampler.h"

// Implementation of ARRSAC, a "real-time" RANSAC algorithm, by Raguram
// et. al. (ECCV 2008). You only need to call the constructor and the Compute
// method to run ARRSAC on your data.
namespace theia {
// Helper struct for scoring the hypotheses in ARRSAC.
template <class Datum> struct ScoredData {
  Datum data;
  double score;
  ScoredData() {}
  ScoredData(const Datum& _data, double _score) : data(_data), score(_score) {}
};

// Comparator method so that we can call c++ algorithm sort.
template <class Datum>
bool CompareScoredData(ScoredData<Datum> i, ScoredData<Datum> j) {
  return i.score > j.score;
}

template <class Datum, class Model>
class Arrsac : public SampleConsensusEstimator<Datum, Model> {
 public:
  // Params:
  //   min_sample_size: The minimum number of samples needed to estimate a
  //     model.
  //   error_thresh: Error threshold for determining inliers vs. outliers. i.e.
  //     if the error is below this, the data point is an inlier.
  //   max_candidate_hyps: Maximum number of hypotheses in the initial
  //     hypothesis set
  //   block_size: Number of data points a hypothesis is evaluated against
  //     before preemptive ordering is used.
  //   nonmin_sample_size: Sample size for the inner RANSAC (nonminimal sample set) - 
  //     for epipolar geometry estimation 14 and for homography estimation 12 are optimal
  Arrsac(int min_sample_size, double error_thresh, int max_candidate_hyps = 500,
         int block_size = 100, int nonmin_sample_size = 14)
      : min_sample_size_(min_sample_size),
        error_thresh_(error_thresh),
        max_candidate_hyps_(max_candidate_hyps),
        block_size_(block_size),
		nonmin_sample_size_(nonmin_sample_size),
        sigma_(0.05),
        epsilon_(0.1),
        inlier_confidence_(0.95),
		num_models_verified_accum_(0), 
		time_compute_model_ratio_(250.0),
		num_rejected_hypotheses_(0),
		rejected_accum_inlier_ratio_(0.0),
		max_inner_ransac_its_(20) {}
  ~Arrsac() {}

  // Set sigma and epsilon SPRT params (see sequential_probability_ratio_test.h
  // for more information). This is an optional method, as it is only an initial
  // estimation of these parameters -- sigma and epsilon are adjusted and
  // re-estimated as the ARRSAC algorithm progresses.
  // inlier_confidence: confidence that there exists one set with no outliers.
  void SetOptionalParameters(double sigma, double epsilon,
                             double inlier_confidence) {
    sigma_ = sigma;
    epsilon_ = epsilon;
    inlier_confidence_ = inlier_confidence;
  }

  // Algorithm 2 in Raguram et. al.'s ARRSAC paper.
  // Params:
  //   data: Input data to generate a model from.
  //   estimator: Derived class used to esimate the model.
  //   best_model: Output parameter that will be filled with the best estimated
  //     model on success.
  // Return: true on successful estimation, false otherwise.
  bool Estimate(const std::vector<Datum>& data,
                const Estimator<Datum, Model>& estimator, Model* best_model);

  // This is sort of a hack. We make this method protected so that we can test
  // it easily. See arrsac_test.cc for more.
 protected:
  // Algorithm 3 in Raguram et. al.'s ARRSAC paper. Given data, generate an
  // initial set of hypotheses from a PROSAC-style sampling. This initial set of
  // hypotheses will be used to generate more hypotheses in the Compute
  // method. Returns the set of initial hypotheses.
  int GenerateInitialHypothesisSet(const std::vector<Datum>& data_input,
                                   const Estimator<Datum, Model>& estimator,
                                   std::vector<ScoredData<Model>>* accepted_hypotheses);

 private:
  // Minimum sample size to generate a model.
  int min_sample_size_;

  // Threshold for determining inliers.
  double error_thresh_;

  // Maximum candidate hypotheses to consider at any time.
  int max_candidate_hyps_;

  // The number of data points that the hypothesis is evaluated against before
  // preemption and re-ordering takes place.
  int block_size_;

  // SPRT Parameters. These parameters are tuned and updated as ARRSAC iterates.
  // Type 1 error estimation.
  double sigma_;

  // Estimated inlier ratio.
  double epsilon_;

  // Confidence that there exists at least one set with no outliers.
  double inlier_confidence_;

  // Accumulated number of generated hypotheses to calculate the average number
  // of hypotheses calculated per model
  int num_models_verified_accum_;

  // Set the sample size for non-minimal sampling according to
  // Chum et.al. in Locally Optimized RANSAC (14 for E)
  int nonmin_sample_size_;

  // Time to compute a hypothesis measured in time units necessary for 
  // evaluating one data point
  double time_compute_model_ratio_;// = 250.0;

  // Vars to keep track of the avg inlier ratio of rejected hypotheses.
  int num_rejected_hypotheses_;
  double rejected_accum_inlier_ratio_;

  // Max. number of hypotheses generation iterations using the inner RANSAC
  // principle
  int max_inner_ransac_its_;// = 20;

};

// -------------------------- Implementation -------------------------- //

template <class Datum, class Model>
int Arrsac<Datum, Model>::GenerateInitialHypothesisSet(
    const std::vector<Datum>& data_input,
    const Estimator<Datum, Model>& estimator,
    std::vector<ScoredData<Model>>* accepted_hypotheses) {
  //   set parameters for SPRT test, calculate initial value of A
  
  double decision_threshold;

  int k = 1; //Number of hypotheses generation iterations
  int k2 = 0; //Number of hypotheses generation iterations failed due to degeneracy
  int m_prime = max_candidate_hyps_;
  // Inner RANSAC variables.
  int inner_ransac_its = 0;
  bool inner_ransac = false;
  int max_num_inliers = 0;

  // We need a local copy of the data input so that we can modify/resize it for
  // inner ransac (which uses inliers from previous results as the sampling
  // universe).
  std::vector<Datum> data;

  // RandomSampler and PROSAC Sampler.
  RandomSampler<Datum> random_sampler(nonmin_sample_size_);
  ProsacSampler<Datum> prosac_sampler(min_sample_size_);
  while (k <= m_prime) {
    std::vector<Model> hypotheses;
    if (!inner_ransac) {
      // Generate hypothesis h(k) with k-th PROSAC sample.
      std::vector<Datum> prosac_subset;
      prosac_sampler.SetSampleNumber(k);
      prosac_sampler.Sample(data_input, &prosac_subset);
      if(!estimator.EstimateModel(prosac_subset, &hypotheses))
	  {
		  k2++;
		  k++;
		  continue;
	  }
    } else {
      // Generate hypothesis h(k) with subset generated from inliers of a
      // previous hypothesis.
      std::vector<Datum> random_subset;
	  bool valid_model;
      random_sampler.Sample(data, &random_subset);
	  if(random_sampler.getSampleSize() == min_sample_size_)
		  valid_model = estimator.EstimateModel(random_subset, &hypotheses);
	  else
		  valid_model = estimator.EstimateModelNonminimal(random_subset, &hypotheses);

      inner_ransac_its++;
      if (inner_ransac_its == max_inner_ransac_its_) {
        inner_ransac_its = 0;
        inner_ransac = false;
      }
	  if(!valid_model)
	  {
		  k2++;
		  k++;
		  continue;
	  }
    }

	//Recalculate the decision threshold depending on inlier ratio, sigma and the
	//average number of hypotheses per sample (e.g. 1 to 3 solutions for 7-pnt alg)
	num_models_verified_accum_ += (int)hypotheses.size();
	decision_threshold = CalculateSPRTDecisionThreshold(sigma_, epsilon_, 
														time_compute_model_ratio_, 
														num_models_verified_accum_/(k-k2));

    for (size_t j = 0; j < hypotheses.size(); j++) {
      int num_tested_points, observed_num_inliers;
      double observed_inlier_ratio;
	  std::vector<bool> inliers;
      // Evaluate hypothesis h(k) with SPRT.
      bool sprt_test = SequentialProbabilityRatioTest(
		  &estimator, data_input, hypotheses[j], 
		  error_thresh_, sigma_, epsilon_, decision_threshold,
          &num_tested_points, &observed_inlier_ratio, inliers,
		  &observed_num_inliers);

      // If the model was rejected by the SPRT test.
      if (!sprt_test) {
        // re-estimate params of SPRT (if required)
        // sigma = average of inlier ratios in bad models
        // TODO(cmsweeney): determine if this estimation (and epsilon) is: (answer from Josef: take the second)
        //    number of inliers observed / total number of points    or
        //    number of inliers observed / number of points observed in SPRT
        rejected_accum_inlier_ratio_ += observed_inlier_ratio;
        num_rejected_hypotheses_++;
		double sigma_temp = rejected_accum_inlier_ratio_ /
							static_cast<double>(num_rejected_hypotheses_);
		if(sigma_temp > 0)
			sigma_ = sigma_temp;
      } else if (observed_num_inliers > max_num_inliers) {
        // Else if hypothesis h(k) is accepted and has the largest support so
        // far.
		max_num_inliers = observed_num_inliers;
		ScoredData<Model> newhypothesis(hypotheses[j],(double)observed_num_inliers);
        accepted_hypotheses->push_back(newhypothesis);

		if(observed_num_inliers > min_sample_size_)
		{
			// Set parameters to force inner ransac to execute.
			inner_ransac = true;
			inner_ransac_its = 0;
			random_sampler.setSampleSize(max(min(nonmin_sample_size_,
										 (int)floor((float)max_num_inliers/2)),
										 min_sample_size_));

			// Set U_in = support of hypothesis h(k).
			data.clear();
			for (size_t i = 0; i < data_input.size(); i++) {
			  if (inliers[i])
				data.push_back(data_input[i]);
			}

			// Re-estimate params of SPRT.
			// Estimate epsilon as inlier ratio for largest size of support.
			if(observed_inlier_ratio == 1.0)
				epsilon_ = 0.9999;
			else
				epsilon_ = observed_inlier_ratio;
			// estimate inlier ratio e' and M_prime (eq 1) Cap M_prime at max of M
			// TODO(cmsweeney): verify that num_tested_points is the correct value
			// here and not data_input.size().
			m_prime =
				(int)ceil(log(1.0 - inlier_confidence_) /
					 log(1.0 - pow(epsilon_, (double)min_sample_size_)));
			m_prime = std::min(max_candidate_hyps_, m_prime);
		}
      }
	  else
	  {
		  ScoredData<Model> newhypothesis(hypotheses[j], observed_num_inliers);
		  accepted_hypotheses->push_back(newhypothesis);
	  }
    }
    k++;
  }
  if(accepted_hypotheses->empty())
	  return 0;
  return k-k2-1;
}

template <class Datum, class Model>
bool Arrsac<Datum, Model>::Estimate(const std::vector<Datum>& data,
                                    const Estimator<Datum, Model>& estimator,
                                    Model* best_model) {

  int sub_block_size = (int)floor((float)block_size_/5.0);
  int hyp_score_kill_thresh = (int)floor(7.0*(float)sub_block_size/12.0);

  // Generate Initial Hypothesis Test
  std::vector<ScoredData<Model>> hypotheses;
  const std::vector<Datum> initial_data(data.begin(),data.begin()+min(data.size(),(size_t)block_size_));
  int k = GenerateInitialHypothesisSet(initial_data, estimator, &hypotheses);

  if(k == 0)
	  return false;

  if(data.size() <= (size_t)block_size_)
  {
	  std::pair<double,int> highscore = std::make_pair(0.0,0);
	  for(unsigned int i = 0; i < hypotheses.size(); i++)
	  {
		  if(hypotheses[i].score > highscore.first)
		  {
			  highscore.first = hypotheses[i].score;
			  highscore.second = i;
		  }
	  }
	  *best_model = hypotheses[highscore.second].data;
	  return true;
  }

  // Score initial set.
  /*std::vector<ScoredData<Model> > hypotheses(initial_hypotheses.size());
  for (int i = 0; i < hypotheses.size(); i++) {
    hypotheses[i] = ScoredData<Model>(initial_hypotheses[i], 0.0);
    // Calculate inlier score for the hypothesis.
    for (int j = 0; j <= block_size_; j++) {
      if (estimator.Error(data[j], hypotheses[i].data) < error_thresh_)
        hypotheses[i].score += 1.0;
    }
  }*/

  RandomSampler<Datum> random_sampler(min_sample_size_);
  
  // Preemptive Evaluation
  int n = static_cast<int>(hypotheses.size());
  for (int i = block_size_ /*+ 1*/; i < (int)data.size(); i++) {

    if ((i+1) % block_size_ == 0) {

		// Reorder
		sort(hypotheses.begin(), hypotheses.end(), CompareScoredData<Model>);

      // Calculate best inlier ratio e' and num hypotheses M' (eq. 1).
      // Use a simple for loop. This should be really fast since the list was
      // recently sorted and the values can only have increased by 1.
      double max_inliers = hypotheses[0].score;
      
      // Estimate best inlier ratio.
      epsilon_ = max_inliers / static_cast<double>(i+1);
	  if(epsilon_ == 1.0)
		  epsilon_ = 0.9999;
      // Calculate number of hypotheses needed (eq. 1).
      int temp_max_candidate_hyps =
          static_cast<int>(ceil(log(1 - inlier_confidence_) /
                                log(1 - pow(epsilon_, i+1))));
      // M' = max(M,M').
      temp_max_candidate_hyps =
          std::min(max_candidate_hyps_, temp_max_candidate_hyps);

      // If we need more hypotheses, generate them now.
      if (temp_max_candidate_hyps > k) {
		  const std::vector<Datum> data_i(data.begin(),data.begin()+i+1);
		  int k2 = 0; //Number of hypotheses generation iterations failed due to degeneracy

        // Generate and evaluate M' - k new hypotheses on i data points.
        for (int j = 0; j < temp_max_candidate_hyps - k; j++) {
			std::vector<Datum> data_random_subset;
			random_sampler.Sample(data, &data_random_subset);

			// Estimate new hypothesis model.
			std::vector<Model> estimated_models;
			if(!estimator.EstimateModel(data_random_subset, &estimated_models))
			{
				k2++;
				continue;
			}

			//Recalculate the decision threshold depending on inlier ratio, sigma and the
			//average number of hypotheses per sample (e.g. 1 to 3 solutions for 7-pnt alg)
			num_models_verified_accum_ += (int)estimated_models.size();
			double decision_threshold = CalculateSPRTDecisionThreshold(sigma_, epsilon_, 
														time_compute_model_ratio_, 
														num_models_verified_accum_/
														(k+j+1-k2));


          for (size_t m = 0; m < estimated_models.size(); m++) {
			  int num_tested_points, observed_num_inliers;
			  double observed_inlier_ratio;
			  std::vector<bool> inliers;

			  // Evaluate hypothesis h(k) with SPRT.
			  bool sprt_test = SequentialProbabilityRatioTest(
				  &estimator, data_i, estimated_models[m], 
				  error_thresh_, sigma_, epsilon_, decision_threshold,
				  &num_tested_points, &observed_inlier_ratio, inliers,
				  &observed_num_inliers);

			  // If the model was rejected by the SPRT test.
			  if (!sprt_test) 
			  {
				// re-estimate params of SPRT (if required)
				// sigma = average of inlier ratios in bad models
				rejected_accum_inlier_ratio_ += observed_inlier_ratio;
				num_rejected_hypotheses_++;
				double sigma_temp = rejected_accum_inlier_ratio_ /
									static_cast<double>(num_rejected_hypotheses_);
				if(sigma_temp > 0)
					sigma_ = sigma_temp;
			  } 
			  else if (observed_num_inliers > (int)max_inliers) 
			  {
				// Else if hypothesis h(k) is accepted and has the largest support so
				// far.
				ScoredData<Model> new_hypothesis(estimated_models[m], observed_num_inliers);
				// Add newly generated model to the hypothesis set.
				hypotheses.push_back(new_hypothesis);

				max_inliers = static_cast<double>(observed_num_inliers);
				if(observed_inlier_ratio == 1.0)
					epsilon_ = 0.9999;
				else
					epsilon_ = observed_inlier_ratio;

				temp_max_candidate_hyps =
					static_cast<int>(ceil(log(1 - inlier_confidence_) /
                    log(1 - pow(epsilon_, i+1))));
				temp_max_candidate_hyps =
					std::min(max_candidate_hyps_, temp_max_candidate_hyps);
				if (temp_max_candidate_hyps <= (k+j+1))
					break;
			  }
			  else
			  {
				ScoredData<Model> new_hypothesis(estimated_models[m], observed_num_inliers);
				// Add newly generated model to the hypothesis set.
				hypotheses.push_back(new_hypothesis);
			  }
          }
		  k++;
        }

        // Update k to be our new maximum candidate hypothesis size.
        //k = temp_max_candidate_hyps;
		n = (int)hypotheses.size();
      }
	  else
	  {
			// Select n, the number of hypotheses to consider.
			int n1 = max(1,(int)floor((float)k * pow(2.0, -1.0 * floor((float)(i+1) / (float)block_size_))));
			//int n = std::min(f_i, static_cast<int>(hypotheses.size() / 2));
		    if( n1 < static_cast<int>(hypotheses.size()))
			{
				// Select hypothesis h(1)...h(n).
				hypotheses.resize(n1);
				n = n1;
			}
	  }
    }
	else if((i+1) % sub_block_size == 0)
	{
		sort(hypotheses.begin(), hypotheses.end(), CompareScoredData<Model>);
		int j = n-1;
		for(; j > 0; j--)
			if(hypotheses[j-1].score - hypotheses[j].score > hyp_score_kill_thresh)
				break;
		n = n-j;
		hypotheses.resize(n);
	}

	if (n == 1) {
      break;
    }

	// Score the hypotheses using data point i.
	  for (size_t j = 0; j < hypotheses.size(); j++)
	  {
		  if (estimator.Error(data[i], hypotheses[j].data) < error_thresh_)
			  hypotheses[j].score += 1.0;
	  }

  }

  // The best model should be at the beginning of the list since we only quit
  // when n==1.
  *best_model = hypotheses[0].data;
  return true;
}

}  // namespace theia

#endif  // THEIA_SOLVERS_ARRSAC_H_
