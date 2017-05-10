###################################################################
#                                                                 #
#    CrackForest V1.0                                             #
#    Limeng Cui (lmcui932-at-163.com)                             #
#                                                                 #
###################################################################

1.Introduction.

CrackForest is a fast road crack detector that achieves excellent accuracy.

Part of the Matlab code is supported on Piotr Dollar’s Structured Edge Detection Toolbox (https://github.com/pdollar/edges).

If you use the Structured Edge Detection Toolbox, we appreciate it if you cite an appropriate subset of the following papers:

@article{shi2016automatic,<br />
&nbsp;&nbsp;title={Automatic road crack detection using random structured forests},<br />
&nbsp;&nbsp;author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},<br />
&nbsp;&nbsp;journal={IEEE Transactions on Intelligent Transportation Systems},<br />
&nbsp;&nbsp;volume={17},<br />
&nbsp;&nbsp;number={12},<br />
&nbsp;&nbsp;pages={3434--3445},<br />
&nbsp;&nbsp;year={2016},<br />
&nbsp;&nbsp;publisher={IEEE}<br />
}

@inproceedings{cui2015pavement,<br />
&nbsp;&nbsp;title={Pavement Distress Detection Using Random Decision Forests},<br />
&nbsp;&nbsp;author={Cui, Limeng and Qi, Zhiquan and Chen, Zhensong and Meng, Fan and Shi, Yong},<br />
&nbsp;&nbsp;booktitle={International Conference on Data Science},<br />
&nbsp;&nbsp;pages={95--102},<br />
&nbsp;&nbsp;year={2015},<br />
&nbsp;&nbsp;organization={Springer}<br />
}

###################################################################

2.License.

The software is made available for non-commercial research purposes only.

###################################################################

3.Installation.

a) This code is written for the Matlab interpreter (tested with versions R2014b) and requires the Matlab Image Processing Toolbox. 

b) Additionally, Piotr’s Computer Vision Toolbox (version 3.26 or later) is also required. It can be downloaded at:

&nbsp;https://pdollar.github.io/toolbox/.

c) Next, please compile mex code from within Matlab (note: win64/linux64 binaries included):

&nbsp;&nbsp;mex private/edgesDetectMex.cpp -outdir private [OMPPARAMS]
  
&nbsp;&nbsp;mex private/edgesNmsMex.cpp    -outdir private [OMPPARAMS]
  
&nbsp;&nbsp;mex private/spDetectMex.cpp    -outdir private [OMPPARAMS]
  
&nbsp;&nbsp;mex private/edgeBoxesMex.cpp   -outdir private

Here [OMPPARAMS] are parameters for OpenMP and are OS and compiler dependent.

&nbsp;&nbsp;Windows:  [OMPPARAMS] = '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'

&nbsp;&nbsp;Linux V1: [OMPPARAMS] = '-DUSEOMP' CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"

&nbsp;&nbsp;Linux V2: [OMPPARAMS] = '-DUSEOMP' CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"

To compile without OpenMP simply omit [OMPPARAMS]; note that code will be single threaded in this case.

d) Add crack detection code to Matlab path (change to current directory first): 
 
 \>> addpath(pwd); savepath;

e) Finally, optionally download the crack image dataset (necessary for training/evaluation):

&nbsp;https://github.com/cuilimeng/CrackForest-dataset

f) A fully trained crack model for RGB images is available as part of this release.

###################################################################

4.Getting Started.

 - Make sure to carefully follow the installation instructions above.
 - Please see "edgesDemo.m" to run demos and get basic usage information.

###################################################################

5.History.

Version 1.0 (2015/09/28)
 - initial version

###################################################################
