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

#ifndef THEIA_SOLVERS_SAMPLE_CONSENSUS_ESTIMATOR_H_
#define THEIA_SOLVERS_SAMPLE_CONSENSUS_ESTIMATOR_H_

//#include <glog/logging.h>
#include <memory>
#include <vector>

#include "estimator.h"
#include "quality_measurement.h"
#include "sampler.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace theia {
template <class Datum, class Model> class SampleConsensusEstimator {
 public:
  SampleConsensusEstimator() {}

  virtual ~SampleConsensusEstimator() {
	/*sampler_->~Sampler;
	sampler_ = NULL;
	quality_measurement_->~QualityMeasurement;
	quality_measurement_ = NULL;*/
  }
  // Computes the best-fitting model using RANSAC. Returns false if RANSAC
  // calculation fails and true (with the best_model output) if successful.
  // Params:
  //   data: the set from which to sample
  //   estimator: The estimator used to estimate the model based on the Datum
  //     and Model type
  //   best_model: The output parameter that will be filled with the best model
  //     estimated from RANSAC
  virtual bool Estimate(const std::vector<Datum>& data,
                        const Estimator<Datum, Model>& estimator,
                        Model* best_model);

  // Returns a bool vector with true for inliers, false for outliers.
  const std::vector<bool>& GetInliers() { return inliers_; }

  // Count the number of inliers.
  int GetNumInliers() {
    int num_inliers = 0;
	for(size_t i = 0; i < inliers_.size(); i++) {
      if (inliers_[i]) num_inliers++;
    }
    return num_inliers;
  }

  int GetNumIterations() { return num_iters_; }

 protected:
  // sampler: The class that instantiates the sampling strategy for this
  //   particular type of sampling consensus.
  // quality_measurement: class that instantiates the quality measurement of
  //   the data. This determines the stopping criterion.
  // max_iters: Maximum number of iterations to run RANSAC. To set the number
  //   of iterations based on the outlier probability, use SetMaxIters.
  SampleConsensusEstimator(Sampler<Datum>* sampler,
                           QualityMeasurement* quality_measurement,
                           int max_iters = 10000)
      : sampler_(sampler), quality_measurement_(quality_measurement),
	    max_iters_(max_iters), num_iters_(-1) {
    CHECK_NOTNULL(sampler);
    //sampler_.reset(sampler);
    CHECK_NOTNULL(quality_measurement);
    //quality_measurement_.reset(quality_measurement);
  }

  // Our sampling strategy.
  cv::Ptr<Sampler<Datum>> sampler_;

  // Our quality metric for the estimated model and data.
  cv::Ptr<QualityMeasurement> quality_measurement_;

  // Inliners from the recent data. Only valid if Estimate has been called!
  std::vector<bool> inliers_;

  // Max number of iterations to perform before terminating .
  int max_iters_;

  // Number of iterations performed before succeeding.
  int num_iters_;
};

template <class Datum, class Model>
bool SampleConsensusEstimator<Datum, Model>::Estimate(
    const std::vector<Datum>& data, const Estimator<Datum, Model>& estimator,
    Model* best_model) {
  assert(data.size() != 0);
  /*CHECK_GT(data.size(), 0)
      << "Cannot perform estimation with 0 data measurements!";*/

  double best_quality = static_cast<double>(QualityMeasurement::INVALID);
  for (int iters = 0; iters < max_iters_; iters++) {
    // Sample subset. Proceed if successfully sampled.
    std::vector<Datum> data_subset;
    if (!sampler_->Sample(data, &data_subset)) {
      continue;
    }

    // Estimate model from subset. Skip to next iteration if the model fails to
    // estimate.
    std::vector<Model> temp_models;
    if (!estimator.EstimateModel(data_subset, &temp_models)) {
      continue;
    }

    // Calculate residuals from estimated model.
	for(size_t j = 0; j < temp_models.size();j++) {
      std::vector<double> residuals = estimator.Residuals(data, temp_models[j]);

      // Determine quality of the generated model.
      std::vector<bool> temp_inlier_set(data.size());
      double sample_quality =
          quality_measurement_->Calculate(residuals, &temp_inlier_set);
      // Update best model if error is the best we have seen.
      if (quality_measurement_->Compare(sample_quality, best_quality) ||
          best_quality == static_cast<double>(QualityMeasurement::INVALID)) {
        *best_model = temp_models[j];
        best_quality = sample_quality;

        // If the inlier termination criterion is met, re-estimate the model
        // based on all inliers and return;
        if (quality_measurement_->SufficientlyHighQuality(best_quality)) {
          // Grab inliers to refine the model.
          std::vector<Datum> temp_consensus_set;
          for (size_t i = 0; i < temp_inlier_set.size(); i++) {
            if (temp_inlier_set[i]) {
              temp_consensus_set.push_back(data[i]);
            }
          }
          // Refine the model based on all current inliers.
          estimator.RefineModel(temp_consensus_set, best_model);
          num_iters_ = iters + 1;

          // Calculate the final inliers.
          inliers_.resize(data.size());
          std::vector<double> final_residuals =
              estimator.Residuals(data, *best_model);
          quality_measurement_->Calculate(final_residuals, &inliers_);
          return true;
        }
      }
    }
  }

  // Grab inliers to refine the model.
  std::vector<double> residuals = estimator.Residuals(data, *best_model);
  std::vector<bool> temp_inlier_set(data.size());
  quality_measurement_->Calculate(residuals, &temp_inlier_set);
  std::vector<Datum> temp_consensus_set;
  for (size_t i = 0; i < temp_inlier_set.size(); i++) {
    if (temp_inlier_set[i]) {
      temp_consensus_set.push_back(data[i]);
    }
  }
  // Refine the model based on all current inliers.
  estimator.RefineModel(temp_consensus_set, best_model);

  // Calculate the final inliers.
  inliers_.resize(data.size());
  std::vector<double> final_residuals = estimator.Residuals(data, *best_model);
  quality_measurement_->Calculate(final_residuals, &inliers_);

  return true;
}

}  // namespace theia

#endif  // THEIA_SOLVERS_SAMPLE_CONSENSUS_ESTIMATOR_H_
