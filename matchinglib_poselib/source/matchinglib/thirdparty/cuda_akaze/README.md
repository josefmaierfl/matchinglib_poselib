## README - A-KAZE Features
This is a fork of libAKAZE with modifications to run it on the GPU using CUDA. The working branch is `cpuidentity`

The interface is the same as the original version. Just changing namespace from libAKAZE to libAKAZECU should be enough. Keypoints and descriptors are returned on the CPU for later matching etc. using e.g. OpenCV. We also provide a rudimentary brute force matcher running on the GPU.

For a detailed description, please refer to <https://github.com/pablofdezalc/akaze>

This code was created as a joint effort between

- Mårten Björkman <https://github.com/Celebrandil>
- Alessandro Pieropan <https://github.com/CoffeRobot>


## Optimizations
The code has been optimized with the goal to maintain the same interface as well as to produce the same results as the original code. This means that certain tradeoffs have been necessary, in particular in the way keypoints are filtered. One difference remains though related to finding scale space extrema, which has been reported as an issue here:

<https://github.com/pablofdezalc/akaze/issues/24>

Major optimizations are possible, but this is work in progress. These optimizations will relax the constraint of having results that are identical to the original code.

## Matcher
A not very optimized matcher is also provided. It returns a std::vector\<std::vector\<cv::DMatch\>\> with the two closest matches.

## Benchmarks
The following benchmarks are measured on the img1.pgm in the iguazu dataset provided by the original authors, and are averages over 100 runs. The computer is a 16 core Xeon running at 2.6 GHz with 32 GB of RAM and an Nvidia Titan X (Maxwell). The operating system is Ubuntu 14.04, with CUDA 8.0.

| Operation     | CPU (original) (ms)      | GPU (ms)  |
| ------------- |:------------------------:|:---------:|
| Detection     |            117           |    6.5    |
| Descriptor    |            10            |    0.9    |

## Limitations
- Previous limitations with respect to the number of keypoints are more or less gone. Set the maximum number of keypoints in AKAZEConfig.h. This is done since cuda memory is preallocated..
- The only descriptor available is MLDB, as proposed in the original authors' paper.
- Currently it only works with 4 octaves and 4 sub-levels (default settings).

## Citation
If you use this code as part of your research, please cite the following papers:

CUDA version

1. **Feature Descriptors for Tracking by Detection: a Benchmark**. Alessandro Pieropan, Mårten Björkman, Niklas Bergström and Danica Kragic (arXiv:1607.06178).

Original A-KAZE papers

2. **Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces**. Pablo F. Alcantarilla, J. Nuevo and Adrien Bartoli. _In British Machine Vision Conference (BMVC), Bristol, UK, September 2013_

3. **KAZE Features**. Pablo F. Alcantarilla, Adrien Bartoli and Andrew J. Davison. _In European Conference on Computer Vision (ECCV), Fiorenze, Italy, October 2012_


## CPU Implementations
If the GPU implementation isn't an option for you, have a look at the CPU-versions below

- <https://github.com/pablofdezalc/akaze>, the original code
- <https://github.com/h2suzuki/fast_akaze>, a faster implementation of the original code.


## Python interface
Compiling a python interface is off by default. Set USE_PYTHON to YES in src/CMakeLists.txt to add this functionality.
I'm just a novice python user, so I cannot say much about the quality of this code. Anyway, I put together a python interface using boost::python, and pyboostcvconverter <https://github.com/Algomorph/pyboostcvconverter> and wrote a small test script computing akaze for two images and then matching the descriptors. The matcher returns a (\#keypoints x 8) numpy array corresponding to the two closest matches (see **Matcher** above). The interface resides in the python-directory and has its own cmake project. It is tested on Ubuntu 14.04 with boost 1.61 and python 2.7. It assumes that akaze is installed in /usr/local. **Important:** For it to work you need to copy the libakaze_pybindings.so to /usr/local/python2.7/dist-packages. Run test.py from the build directory, i.e. cd \<AKAZEROOT\>/python/build; python ../test.py. Suggestions on improvements of pull requests are appreciated.


## MATLAB interface
This will presumably not work, but might only require a few modifications. If someone is interested, fork the repository and create a pull-request.


## Contact Info
If you have questions, or are finding this code useful, please let me know!

Niklas Bergström,
email: nbergst@gmail.com
