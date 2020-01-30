# GMS feature matching

## Introduction

Refactored source code of **GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence** based on the implementation of [JiaWang Bian](http://jwbian.net).

[[Publication](http://jwbian.net/Papers/GMS_CVPR17.pdf)] 
[[Code](https://github.com/JiawangBian/GMS-Feature-Matcher)]
[[Licence](https://github.com/JiawangBian/GMS-Feature-Matcher/blob/master/LICENSE)]

## Dependencies
```
opencv-3.1.0 or later with CUDA support (recommended)
opencv-3.0.0 (minimal)
```

## Usage
```
gms-feature-matching <image path> [--imageIdx <image index of start frame>] [--imageOffset <image offset>] [--numFeatures <number of ORB features>] [--imageHeight <image height of scaled image>]
```

## Authors

* Daniel Steininger
* Julia Simon
