###################################################################
#                                                                 #
#    CrackForest V1.0                                             #
#    Limeng Cui (lmcui932-at-163.com)                             #
#                                                                 #
###################################################################

1. Introduction.

CrackForest is a fast road crack detector that achieves excellent accuracy.

Part of the Matlab code is supported on Piotr Dollar’s Structured Edge Detection Toolbox (https://github.com/pdollar/edges).

If you use the Structured Edge Detection Toolbox, we appreciate it if you cite an appropriate subset of the following papers:

TBA.

###################################################################

2. License.

The software is made available for non-commercial research purposes only.

###################################################################

3. Installation.

a) This code is written for the Matlab interpreter (tested with versions R2014b) and requires the Matlab Image Processing Toolbox. 

b) Additionally, Piotr’s Computer Vision Toolbox (version 3.26 or later) is also required. It can be downloaded at:
 http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html.

c) Next, please compile mex code from within Matlab (note: win64/linux64 binaries included):
  mex private/edgesDetectMex.cpp -outdir private [OMPPARAMS]
  mex private/edgesNmsMex.cpp    -outdir private [OMPPARAMS]
  mex private/spDetectMex.cpp    -outdir private [OMPPARAMS]
  mex private/edgeBoxesMex.cpp   -outdir private
Here [OMPPARAMS] are parameters for OpenMP and are OS and compiler dependent.
  Windows:  [OMPPARAMS] = '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  Linux V1: [OMPPARAMS] = '-DUSEOMP' CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
  Linux V2: [OMPPARAMS] = '-DUSEOMP' CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
To compile without OpenMP simply omit [OMPPARAMS]; note that code will be single threaded in this case.

d) Add crack detection code to Matlab path (change to current directory first): 
 >> addpath(pwd); savepath;

e) Finally, optionally download the crack image dataset (necessary for training/evaluation):
TBA.

f) A fully trained crack model for RGB images is available as part of this release.

###################################################################

4. Getting Started.

 - Make sure to carefully follow the installation instructions above.
 - Please see "edgesDemo.m" to run demos and get basic usage information.

###################################################################

5. History.

Version 1.0 (2015/09/28)
 - initial version

###################################################################
