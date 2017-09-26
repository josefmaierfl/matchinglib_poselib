# MATCHING- AND POSELIB


## REQUIREMENTS:

* sba: https://redmine.ait.ac.at/sas-svn/thirdpartyroot/trunk/sba-1.6
* nanoflann: https://redmine.ait.ac.at/sas-svn/thirdpartyroot/trunk/nanoflann-1.1.1
* clapack: https://redmine.ait.ac.at/sas-svn/thirdpartyroot/trunk/clapack-3.2.1

example:
```
svn checkout https://redmine.ait.ac.at/sas-svn/thirdpartyroot/trunk thirdpartyroot --depth immediates --username=JungR
cd thirdpartyroot

svn update --set-depth infinity sba-1.6
svn update --set-depth infinity nanoflann-1.1.1
svn update --set-depth infinity clapack-3.2.1
```

### setup your bashrc:

```
export THIRDPARTYROOT=/home/<user>/work/thirdpartyroot
export PATH=$PATH:/home/<user>/bin
```
