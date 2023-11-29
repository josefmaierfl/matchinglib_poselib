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
#ifndef THEIA_SOLVERS_PROSAC_SAMPLER_H_
#define THEIA_SOLVERS_PROSAC_SAMPLER_H_

//#include <glog/logging.h>
#include <stdlib.h>
#include <algorithm>
//#include <chrono>
#include <random>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include "../../../../semirandom.h"
//#include "../../../../../../work_PfeLib/imgThreshComp/PfeCCLabeling_16u_C1IR/include/mer_twi.h"

#include "sampler.h"

namespace theia {
// Prosac sampler used for PROSAC implemented according to "Matching with PROSAC
// - Progressive Sampling Consensus" by Chum and Matas.
template <class Datum> class ProsacSampler : public Sampler<Datum> {
 public:
  // num_samples: the number of samples needed. Typically this corresponds to
  //   the minumum number of samples needed to estimate a model.
   explicit ProsacSampler(int num_samples,
                          std::mt19937 &mt,
                          int ransac_convergence_iterations = 200000)
       : num_samples_(num_samples),
         ransac_convergence_iterations_(ransac_convergence_iterations),
         kth_sample_number_(1),
         mt_(mt) /*,
          generator(std::chrono::system_clock::now().time_since_epoch().count())*/
   {
   }

  ~ProsacSampler() {}

  // Set the sample such that you are sampling the kth prosac sample (Eq. 6).
  void SetSampleNumber(int k) { kth_sample_number_ = k; }

  // Samples the input variable data and fills the vector subset with the prosac
  // samples.
  // NOTE: This assumes that data is in sorted order by quality where data[i] is
  // of higher quality than data[j] for all i < j.
  bool Sample(const std::vector<Datum>& data, std::vector<Datum>* subset) {
    // Set t_n according to the PROSAC paper's recommendation.
    double t_n = ransac_convergence_iterations_;
    int n = num_samples_;
    // From Equations leading up to Eq 3 in Chum et al.
    for (int i = 0; i < num_samples_; i++) {
      t_n *= static_cast<double>(n - i) / (data.size() - i);
    }

    double t_n_prime = 1.0;
    // Choose min n such that T_n_prime >= t (Eq. 5).
    for (int t = 1; t <= kth_sample_number_; t++) {
      if (t > t_n_prime && n < (int)data.size()) {
        double t_n_plus1 = (t_n * ((double)n + 1.0)) / ((double)n + 1.0 - (double)num_samples_);
        t_n_prime += ceil(t_n_plus1 - t_n);
        t_n = t_n_plus1;
        n++;
      }
    }
    subset->reserve(num_samples_);
	  // static cv::RNG rng;

	//To ensure a new set, the (n_old+1)'th or n'th point has to be chosen and the rest of the points
	//at random. Otherwise an identical set could be chosen.
    if (t_n_prime < kth_sample_number_) {
      // Randomly sample m data points from the top n data points.
      //std::uniform_int_distribution<int> distribution(0, n - 1);
      // std::vector<int> random_numbers;

      std::vector<int> numbers_all(n);
      std::iota(numbers_all.begin(), numbers_all.end(), 0);
      std::shuffle(numbers_all.begin(), numbers_all.end(), mt_);

      for (int i = 0; i < num_samples_; i++) {
        // Generate a random number that has not already been used.
        // int rand_number;
        // while (std::find(random_numbers.begin(), random_numbers.end(),
        //                  (rand_number = rng.uniform((int)0,n))) !=
        //        random_numbers.end()) {}

        // random_numbers.push_back(rand_number);

        // Push the *unique* random index back.
        // subset->push_back(data[rand_number]);
        subset->push_back(data.at(numbers_all.at(i)));
      }
    } else {
      //std::uniform_int_distribution<int> distribution(0, n - 2);
      // std::vector<int> random_numbers;

      std::vector<int> numbers_all(n - 1);
      std::iota(numbers_all.begin(), numbers_all.end(), 0);
      std::shuffle(numbers_all.begin(), numbers_all.end(), mt_);

      // Randomly sample m-1 data points from the top n-1 data points.
      for (int i = 0; i < num_samples_ - 1; i++) {
        // Generate a random number that has not already been used.
        // int rand_number;
        // while (std::find(random_numbers.begin(), random_numbers.end(),
        //                  (rand_number = rng.uniform((int)0,n-1))) !=
        //        random_numbers.end()) {}
        // random_numbers.push_back(rand_number);

        // Push the *unique* random index back.
        // subset->push_back(data[rand_number]);
        subset->push_back(data.at(numbers_all.at(i)));
      }
      // Make the last point from the nth position.
      subset->push_back(data[n-1]);
    }
    assert(subset->size() == num_samples_);/* << "Prosac subset is incorrect "
                                           << "size!";*/
    kth_sample_number_++;
    return true;
  }

 private:
  // Number of samples to return.
  int num_samples_;

  // Number of iterations of PROSAC before it just acts like ransac.
  int ransac_convergence_iterations_;

  // The kth sample of prosac sampling.
  int kth_sample_number_;

  // Random number generator seed
  //std::default_random_engine generator;
  std::mt19937 &mt_;
};

}  // namespace theia

#endif  // THEIA_SOLVERS_PROSAC_SAMPLER_H_
