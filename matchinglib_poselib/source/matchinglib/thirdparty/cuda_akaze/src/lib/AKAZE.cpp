//=============================================================================
//
// AKAZE.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Toshiba Research Europe Ltd (1)
//               TrueVision Solutions (2)
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file AKAZE.cpp
 * @brief Main class for detecting and describing binary features in an
 * accelerated nonlinear scale space
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "AKAZE.h"
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>  //%%%%

// #define CUDA_API_PER_THREAD_DEFAULT_STREAM
// #include <cuda_runtime_api.h>

using namespace std;
using namespace libAKAZECU;

/* ************************************************************************* */

void Matcher::bfmatch(cv::Mat &desc_query, cv::Mat &desc_train,
                      std::vector<std::vector<cv::DMatch>> &dmatches)
{
  if (maxnquery < desc_query.rows)
  {
    if (descq_d)
      cudaFree(descq_d);
    if (dmatches_d)
      cudaFree(dmatches_d);
    cudaMallocPitch((void **)&descq_d, &pitch, 64, desc_query.rows);
    cudaMemset2D(descq_d, pitch, 0, 64, desc_query.rows);
    cudaMalloc((void **)&dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch));
    if (dmatches_h)
      delete[] dmatches_h;
    dmatches_h = new cv::DMatch[2 * desc_query.rows];
    maxnquery = desc_query.rows;
  }
  if (maxntrain < desc_train.rows)
  {
    if (desct_d)
      cudaFree(descq_d);
    cudaMallocPitch((void **)&desct_d, &pitch, 64, desc_train.rows);
    cudaMemset2DAsync(desct_d, pitch, 0, 64, desc_train.rows);
    maxntrain = desc_train.rows;
  }

  cudaMemcpy2DAsync(descq_d, pitch, desc_query.data, desc_query.cols,
                    desc_query.cols, desc_query.rows, cudaMemcpyHostToDevice);

  cudaMemcpy2DAsync(desct_d, pitch, desc_train.data, desc_train.cols,
                    desc_train.cols, desc_train.rows, cudaMemcpyHostToDevice);

  dim3 block(desc_query.rows);

  MatchDescriptors(desc_query, desc_train, dmatches, pitch,
                   descq_d, desct_d, dmatches_d, dmatches_h);

  cudaMemcpy(dmatches_h, dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch),
             cudaMemcpyDeviceToHost);

  dmatches.clear();
  for (int i = 0; i < desc_query.rows; ++i)
  {
    std::vector<cv::DMatch> tdmatch;
    tdmatch.push_back(dmatches_h[2 * i]);
    tdmatch.push_back(dmatches_h[2 * i + 1]);
    dmatches.push_back(tdmatch);
  }
}

cv::Mat Matcher::bfmatch_(cv::Mat desc_query, cv::Mat desc_train)
{

  std::vector<std::vector<cv::DMatch>> dmatches_vec;
  bfmatch(desc_query, desc_train, dmatches_vec);

  cv::Mat dmatches_mat(dmatches_vec.size(), 8, CV_32FC1);
  for (size_t i = 0; i < dmatches_vec.size(); ++i)
  {
    float *mdata = (float *)&dmatches_mat.data[i * 8 * sizeof(float)];

    mdata[0] = dmatches_vec[i][0].queryIdx;
    mdata[1] = dmatches_vec[i][0].trainIdx;
    mdata[2] = 0.f; //dmatches_vec[i][0].imgIdx;
    mdata[3] = dmatches_vec[i][0].distance;

    mdata[4] = dmatches_vec[i][1].queryIdx;
    mdata[5] = dmatches_vec[i][1].trainIdx;
    mdata[6] = 0.f; //dmatches_vec[i][1].imgIdx;
    mdata[7] = dmatches_vec[i][1].distance;
  }

  return dmatches_mat;
}

Matcher::~Matcher()
{
  if (descq_d)
  {
    cudaFree(descq_d);
  }
  if (desct_d)
  {
    cudaFree(desct_d);
  }
  if (dmatches_d)
  {
    cudaFree(dmatches_d);
  }
  if (dmatches_h)
  {
    delete[] dmatches_h;
  }
}

/* ************************************************************************* */
AKAZE::AKAZE(const AKAZEOptions &options, cudaStream_t &stream_, std::mt19937 &mt_, cv::InputArray mask) : options_(options), stream(stream_), mt(mt_)
{
  ncycles_ = 0;
  reordering_ = true;
  // deviceInit(0);
  // createStream(stream);

  if (!mask.empty())
  {
    const cv::Mat mask_ = mask.getMat();
    int nrMaskPix = cv::countNonZero(mask_);
    if (nrMaskPix)
    {
      maskRatio = static_cast<double>(mask_.size().area()) / static_cast<double>(nrMaskPix);
      maskRatio = maskRatio > 10. ? 10. : maskRatio;
      // std::cout << "Ratio: " << maskRatio << std::endl;
      options_.maxkeypoints = static_cast<int>(static_cast<double>(options_.maxkeypoints) * maskRatio);
      if (!options_.info.empty())
      {
        options_.info += "-" + std::to_string(mask_.rows) + "-" + std::to_string(mask_.cols);
      }
    }
  }

  if (options_.descriptor_size > 0 && options_.descriptor >= MLDB_UPRIGHT)
  {
    generateDescriptorSubsample(descriptorSamples_, descriptorBits_, options_.descriptor_size,
                                options_.descriptor_pattern_size, options_.descriptor_channels, mt);
  }
  allocCudaVar(d_PointCounter, 1, stream);
  allocCudaVar(d_ExtremaIdx, 16, stream);
  allocCudaVar(comp_idx_1, 61 * 8, stream);
  allocCudaVar(comp_idx_2, 61 * 8, stream);

  Allocate_Memory_Evolution();
}

/* ************************************************************************* */
AKAZE::~AKAZE() {
  cudaStreamSynchronize(stream);
  cudaHostUnregister(cuda_buffers.data());
  evolution_.clear();
  FreeBuffers(b_ptrs);
  freeCudaVar(d_PointCounter);
  freeCudaVar(d_ExtremaIdx);
  freeCudaVar(comp_idx_1);
  freeCudaVar(comp_idx_2);
  // DestroyStream(stream);
}

/* ************************************************************************* */
void AKAZE::Allocate_Memory_Evolution()
{
  float rfactor = 0.0;
  int level_height = 0, level_width = 0;

  // Allocate the dimension of the matrices for the evolution
  for (int i = 0; i <= options_.omax - 1; i++) {
    rfactor = 1.0 / pow(2.0f, i);
    level_height = (int)(options_.img_height * rfactor);
    level_width = (int)(options_.img_width * rfactor);

    // Smallest possible octave and allow one scale if the image is small
    if ((level_width < 80 || level_height < 40) && i != 0) {
      options_.omax = i;
      // cout << "max octaves: " << i << endl;
      break;
    }

    for (int j = 0; j < options_.nsublevels; j++) {
      TEvolution step;
      step.size = cv::Size(level_width, level_height);
      step.esigma = options_.soffset *
                    pow(2.0f, (float)(j) / (float)(options_.nsublevels) + i);
      step.sigma_size = fRound(step.esigma);
      step.etime = 0.5 * (step.esigma * step.esigma);
      step.octave = i;
      step.sublevel = j;
      evolution_.emplace_back(move(step));
    }
  }

  // Allocate memory for the number of cycles and time steps
  for (size_t i = 1; i < evolution_.size(); i++) {
    int naux = 0;
    vector<float> tau;
    float ttime = 0.0;
    ttime = evolution_[i].etime - evolution_[i - 1].etime;
    float tmax = 0.25;// * (1 << 2 * evolution_[i].octave);
    naux = fed_tau_by_process_time(ttime, 1, tmax, reordering_, tau);
    nsteps_.push_back(naux);
    tsteps_.push_back(tau);
    ncycles_++;
  }

  // Allocate memory for CUDA buffers
  options_.ncudaimages = 4 * options_.nsublevels;
  b_ptrs = AllocBuffers(evolution_[0].size.width, evolution_[0].size.height, options_.ncudaimages,
                        options_.omax, options_.maxkeypoints, cuda_buffers, cuda_bufferpoints,
                        cuda_points, cuda_ptindices, cuda_desc, cuda_descbuffer, cuda_images, comp_idx_1, comp_idx_2, stream);
}

/* ************************************************************************* */
// std::mutex m_scale_space;
int AKAZE::Create_Nonlinear_Scale_Space(const cv::Mat &img)
{
  double t1 = 0.0, t2 = 0.0;

  if (evolution_.empty()) {
    cerr << "Error generating the nonlinear scale space!!" << endl;
    cerr << "Firstly you need to call AKAZE::Allocate_Memory_Evolution()"
         << endl;
    return -1;
  }

  h_img = cv::cuda::HostMem(img, cv::cuda::HostMem::PAGE_LOCKED);

  t1 = cv::getTickCount();

  CudaImage& Limg = cuda_buffers[0];
  CudaImage& Lt = cuda_buffers[0];
  CudaImage& Lsmooth = cuda_buffers[1];
  CudaImage& Ltemp = cuda_buffers[2];

  Limg.h_data = (float *)h_img.data;
  Limg.Download(h_img.step, stream);

  std::string info, *info_ptr = nullptr;
  if (!options_.info.empty())
  {
    info = options_.info;
    info_ptr = &info;
  }

  ContrastPercentile(Limg, Ltemp, Lsmooth, options_.kcontrast_percentile,
                     options_.kcontrast_nbins, options_.kcontrast, stream, info_ptr);
  
  std::string info1, *info_ptr1 = nullptr;
  if (info_ptr)
  {
    info1 = info + "_0";
    info_ptr1 = &info1;
  }
  LowPass(Limg, Lt, Ltemp, options_.soffset * options_.soffset,
          2 * ceil((options_.soffset - 0.8) / 0.3) + 3, stream, info_ptr1);
  Copy(Lt, Lsmooth, stream, info_ptr1);

  t2 = cv::getTickCount();
  timing_.kcontrast = 1000.0 * (t2 - t1) / cv::getTickFrequency();

  // Now generate the rest of evolution levels
  for (size_t i = 1; i < evolution_.size(); i++) {
    TEvolution& evn = evolution_[i];
    int num = options_.ncudaimages;
    CudaImage& Ltn = cuda_buffers.at(evn.octave * num + 0 + 4 * evn.sublevel);
    CudaImage& Lsmoothn = cuda_buffers.at(evn.octave * num + 1 + 4 * evn.sublevel);
    CudaImage& Lstepn = cuda_buffers.at(evn.octave * num + 2);
    CudaImage& Lflown = cuda_buffers.at(evn.octave * num + 3);

    if (info_ptr)
    {
      info1 = info + "_1";
      info_ptr1 = &info1;
    }

    TEvolution& evo = evolution_[i - 1];
    CudaImage& Ltold = cuda_buffers[evo.octave * num + 0 + 4 * evo.sublevel];
    if (evn.octave > evo.octave)
    {
      HalfSample(Ltold, Ltn, stream, info_ptr);
      options_.kcontrast = options_.kcontrast * 0.75;
    }
    else
      Copy(Ltold, Ltn, stream, info_ptr1);

    LowPass(Ltn, Lsmoothn, Lstepn, 1.0, 5, stream, info_ptr1);
    Flow(Lsmoothn, Lflown, options_.diffusivity, options_.kcontrast, stream, info_ptr);

    for (int j = 0; j < nsteps_[i - 1]; j++)
    {
      std::string info2, *info_ptr2 = nullptr;
      if (info_ptr)
      {
        info2 = info + "_" + std::to_string(j);
        info_ptr2 = &info2;
      }
      NLDStep(Ltn, Lflown, Lstepn, tsteps_[i - 1][j], stream, info_ptr2);
    }
  }

  t2 = cv::getTickCount();
  timing_.scale = 1000.0 * (t2 - t1) / cv::getTickFrequency();
  
  return 0;
}

void kpvec2mat(std::vector<cv::KeyPoint> &kpts, cv::Mat &_mat)
{
  _mat = cv::Mat(kpts.size(), 7, CV_32FC1);
  for (int i = 0; i < (int)kpts.size(); ++i)
  {
    cv::KeyPoint &kp = kpts.at(i);
    _mat.at<float>(i, 0) = kp.pt.x;
    _mat.at<float>(i, 1) = kp.pt.y;
    _mat.at<float>(i, 2) = kp.size;
    _mat.at<float>(i, 3) = kp.angle;
    _mat.at<float>(i, 4) = kp.response;
    _mat.at<float>(i, 5) = static_cast<float>(kp.octave);
    _mat.at<float>(i, 6) = static_cast<float>(kp.class_id);
  }
}

void mat2kpvec(cv::Mat &_mat, std::vector<cv::KeyPoint> &_kpts)
{
  for (int i = 0; i < _mat.rows; ++i)
  {
    cv::Vec<float, 7> v = _mat.at<cv::Vec<float, 7>>(i, 0);
    cv::KeyPoint kp(v[0], v[1], v[2], v[3], v[4], static_cast<int>(v[5]), static_cast<int>(v[6]));
    _kpts.emplace_back(std::move(kp));
  }
}

cv::Mat AKAZE::Feature_Detection_()
{

  std::vector<cv::KeyPoint> kpts;
  this->Feature_Detection(&kpts);

  cv::Mat mat;
  kpvec2mat(kpts, mat);

  return mat;
}

/* ************************************************************************* */
void AKAZE::Feature_Detection(std::vector<cv::KeyPoint> *kpts)
{
  double t1 = 0.0, t2 = 0.0;

  t1 = cv::getTickCount();

  if(evolution_.size() < 2){
    std::cout << "AKAZE: Evolution size to small" << std::endl;
    return;
  }

  std::string info, *info_ptr = nullptr;
  if (!options_.info.empty())
  {
    info = options_.info;
    info_ptr = &info;
  }

  int num = options_.ncudaimages;
  for (size_t i = 0; i < evolution_.size(); i++) {
    TEvolution& ev = evolution_[i];
    CudaImage& Lsmooth = cuda_buffers[ev.octave * num + 1 + 4 * ev.sublevel];
    CudaImage& Lx = cuda_buffers[ev.octave * num + 2 + 4 * ev.sublevel];
    CudaImage& Ly = cuda_buffers[ev.octave * num + 3 + 4 * ev.sublevel];

    std::string info1, *info_ptr1 = nullptr;
    if (info_ptr)
    {
      info1 = info + "_" + std::to_string(i);
      info_ptr1 = &info1;
    }

    float ratio = pow(2.0f, (float)ev.octave);
    int sigma_size_ = fRound(ev.esigma * options_.derivative_factor / ratio);
    HessianDeterminant(Lsmooth, Lx, Ly, sigma_size_, stream, info_ptr1);
  }
  t2 = cv::getTickCount();
  timing_.derivatives = 1000.0 * (t2 - t1) / cv::getTickFrequency();

  int nrThTries = 0;
  const size_t nr_max_evo = evolution_.size();
  bool recomp = false;
  do
  {
    recomp = false;
    ClearPoints(d_PointCounter, stream);
    bool break_next = false;
    for (size_t i = 0; i < evolution_.size(); i++)
    {
      TEvolution& ev = evolution_[i];
      TEvolution& evp = evolution_[(i > 0 && evolution_[i].octave == evolution_[i - 1].octave ? i - 1 : i)];
      TEvolution &evn = evolution_[(i < evolution_.size() - 1 &&
                                            evolution_[i].octave == evolution_[i + 1].octave
                                        ? i + 1
                                        : i)];
      CudaImage& Ldet = cuda_buffers[ev.octave * num + 1 + 4 * ev.sublevel];
      CudaImage& LdetP = cuda_buffers[evp.octave * num + 1 + 4 * evp.sublevel];
      CudaImage& LdetN = cuda_buffers[evn.octave * num + 1 + 4 * evn.sublevel];

      float smax = 1.0f;
      if (options_.descriptor == SURF_UPRIGHT || options_.descriptor == SURF ||
          options_.descriptor == MLDB_UPRIGHT || options_.descriptor == MLDB)
        smax = 10.0 * sqrtf(2.0f);
      else if (options_.descriptor == MSURF_UPRIGHT ||
              options_.descriptor == MSURF)
        smax = 12.0 * sqrtf(2.0f);

      float ratio = pow(2.0f, (float)evolution_[i].octave);
      float size = evolution_[i].esigma * options_.derivative_factor;
      float border = smax * fRound(size / ratio);
      float thresh = std::max(options_.dthreshold, options_.min_dthreshold);

      std::string info_i, *info_i_ptr = nullptr;
      if (!info.empty())
      {
        info_i = info + "_" + std::to_string(i);
        info_i_ptr = &info_i;
      }

      if (!FindExtrema(Ldet, LdetP, LdetN, border, thresh, i, evolution_[i].octave,
                       size, cuda_points, options_.maxkeypoints, d_PointCounter, d_ExtremaIdx, stream, info_i_ptr))
      {
        if(break_next){
          evolution_.resize(i);
          break;
        }
        if (i < nr_max_evo / 2 && nrThTries < 6){
          recomp = true;
          nrThTries++;
          options_.dthreshold *= 1.5;
          break;
        }
        if (i > 1)
        {
          evolution_.resize(i);
          break;
        }
        else
        {
          break_next = true;
        }
      }
      else if (break_next)
      {
        break_next = false;
      }
    }
  } while (recomp);

  FilterExtrema(cuda_points, cuda_bufferpoints, cuda_ptindices, nump, d_PointCounter, d_ExtremaIdx, static_cast<int>(evolution_.size()), stream, info_ptr);
  // std::cout << "NrPts: " << nump << endl;

  if(kpts){
    kpts->clear();
    WaitCuda(stream);
    if(nump){
      FindOrientation(cuda_points, cuda_buffers, cuda_images, nump, stream, info_ptr);
      GetPoints(*kpts, cuda_points, nump, stream);
    }
  }
  WaitCuda(stream);

  double t3 = cv::getTickCount();
  timing_.extrema = 1000.0 * (t3 - t2) / cv::getTickFrequency();
  timing_.detector = 1000.0 * (t3 - t1) / cv::getTickFrequency();
}

/* ************************************************************************* */
/**
 * @brief This method  computes the set of descriptors through the nonlinear
 * scale space
 * @param kpts Vector of detected keypoints
 * @param desc Matrix to store the descriptors
*/
void AKAZE::Compute_Descriptors(std::vector<cv::KeyPoint> &kpts,
                                cv::Mat &desc)
{
  double t1 = 0.0, t2 = 0.0;
  t1 = cv::getTickCount();

  if (evolution_.size() < 2)
  {
    std::cout << "AKAZE: Evolution size to small" << std::endl;
    return;
  }

  if(!nump){
    std::cout << "AKAZE: No keypoints found" << std::endl;
    return;
  }

  std::string info, *info_ptr = nullptr;
  if (!options_.info.empty())
  {
    info = options_.info;
    info_ptr = &info;
  }

  // Allocate memory for the matrix with the descriptors
  if (options_.descriptor < MLDB_UPRIGHT) {
    desc = cv::Mat::zeros(kpts.size(), 64, CV_32FC1);
  } else {
    // We use the full length binary descriptor -> 486 bits
    if (options_.descriptor_size == 0) {
      int t = (6 + 36 + 120) * options_.descriptor_channels;
      desc = cv::Mat::zeros(kpts.size(), ceil(t / 8.), CV_8UC1);
    } else {
      // We use the random bit selection length binary descriptor
      desc = cv::Mat::zeros(kpts.size(), ceil(options_.descriptor_size / 8.),
                            CV_8UC1);
    }
  }

  int pattern_size = options_.descriptor_pattern_size;

  switch (options_.descriptor) {
    case MLDB:
      if (kpts.empty()){
        FindOrientation(cuda_points, cuda_buffers, cuda_images, nump, stream, info_ptr);
        GetPoints(kpts, cuda_points, nump, stream);
      }
      ExtractDescriptors(cuda_points, cuda_images,
                         cuda_desc, cuda_descbuffer, pattern_size, nump, comp_idx_1, comp_idx_2, stream, info_ptr);
      GetDescriptors(desc, cuda_desc, nump, stream);
      break;
    case SURF_UPRIGHT:
    case SURF:
    case MSURF_UPRIGHT:
    case MSURF:
    case MLDB_UPRIGHT:
      cout << "Descriptor not implemented\n";
  }

  t2 = cv::getTickCount();
  timing_.descriptor = 1000.0 * (t2 - t1) / cv::getTickFrequency();

  WaitCuda(stream);
}

/* ************************************************************************* */
void libAKAZECU::generateDescriptorSubsample(cv::Mat &sampleList,
                                             cv::Mat &comparisons, int nbits,
                                             int pattern_size, int nchannels)
{
  std::random_device rd;
  std::mt19937 g(rd());
  libAKAZECU::generateDescriptorSubsample(sampleList, comparisons, nbits, pattern_size, nchannels, g);
}

void libAKAZECU::generateDescriptorSubsample(cv::Mat &sampleList,
                                             cv::Mat &comparisons, int nbits,
                                             int pattern_size, int nchannels, std::mt19937 &mt)
{
  int ssz = 0;
  for (int i = 0; i < 3; i++) {
    int gz = (i + 2) * (i + 2);
    ssz += gz * (gz - 1) / 2;
  }
  ssz *= nchannels;

  CV_Assert(nbits <= ssz &&
            "descriptor size can't be bigger than full descriptor");

  // Since the full descriptor is usually under 10k elements, we pick
  // the selection from the full matrix.  We take as many samples per
  // pick as the number of channels. For every pick, we
  // take the two samples involved and put them in the sampling list

  cv::Mat_<int> fullM(ssz / nchannels, 5);
  for (size_t i = 0, c = 0; i < 3; i++) {
    int gdiv = i + 2;  // grid divisions, per row
    int gsz = gdiv * gdiv;
    int psz = ceil(2. * pattern_size / (float)gdiv);

    for (int j = 0; j < gsz; j++) {
      for (int k = j + 1; k < gsz; k++, c++) {
        fullM(c, 0) = i;
        fullM(c, 1) = psz * (j % gdiv) - pattern_size;
        fullM(c, 2) = psz * (j / gdiv) - pattern_size;
        fullM(c, 3) = psz * (k % gdiv) - pattern_size;
        fullM(c, 4) = psz * (k / gdiv) - pattern_size;
      }
    }
  }

  // srand(1024);
  cv::Mat_<int> comps =
      cv::Mat_<int>(nchannels * ceil(nbits / (float)nchannels), 2);
  comps = 1000;

  // Select some samples. A sample includes all channels
  int count = 0;
  size_t npicks = ceil(nbits / (float)nchannels);
  cv::Mat_<int> samples(29, 3);
  cv::Mat_<int> fullcopy = fullM.clone();
  samples = -1;

  for (size_t i = 0; i < npicks; i++) {
    size_t k = mt() % (static_cast<size_t>(fullM.rows) - i);
    if (i < 6) {
      // Force use of the coarser grid values and comparisons
      k = i;
    }

    bool n = true;

    for (int j = 0; j < count; j++) {
      if (samples(j, 0) == fullcopy(k, 0) && samples(j, 1) == fullcopy(k, 1) &&
          samples(j, 2) == fullcopy(k, 2)) {
        n = false;
        comps(i * nchannels, 0) = nchannels * j;
        comps(i * nchannels + 1, 0) = nchannels * j + 1;
        comps(i * nchannels + 2, 0) = nchannels * j + 2;
        break;
      }
    }

    if (n) {
      samples(count, 0) = fullcopy(k, 0);
      samples(count, 1) = fullcopy(k, 1);
      samples(count, 2) = fullcopy(k, 2);
      comps(i * nchannels, 0) = nchannels * count;
      comps(i * nchannels + 1, 0) = nchannels * count + 1;
      comps(i * nchannels + 2, 0) = nchannels * count + 2;
      count++;
    }

    n = true;
    for (int j = 0; j < count; j++) {
      if (samples(j, 0) == fullcopy(k, 0) && samples(j, 1) == fullcopy(k, 3) &&
          samples(j, 2) == fullcopy(k, 4)) {
        n = false;
        comps(i * nchannels, 1) = nchannels * j;
        comps(i * nchannels + 1, 1) = nchannels * j + 1;
        comps(i * nchannels + 2, 1) = nchannels * j + 2;
        break;
      }
    }

    if (n) {
      samples(count, 0) = fullcopy(k, 0);
      samples(count, 1) = fullcopy(k, 3);
      samples(count, 2) = fullcopy(k, 4);
      comps(i * nchannels, 1) = nchannels * count;
      comps(i * nchannels + 1, 1) = nchannels * count + 1;
      comps(i * nchannels + 2, 1) = nchannels * count + 2;
      count++;
    }

    cv::Mat tmp = fullcopy.row(k);
    fullcopy.row(fullcopy.rows - i - 1).copyTo(tmp);
  }

  sampleList = samples.rowRange(0, count).clone();
  comparisons = comps.rowRange(0, nbits).clone();
}

/* ************************************************************************* */
void libAKAZECU::check_descriptor_limits(int& x, int& y, int width,
                                         int height) {
  if (x < 0) x = 0;

  if (y < 0) y = 0;

  if (x > width - 1) x = width - 1;

  if (y > height - 1) y = height - 1;
}
