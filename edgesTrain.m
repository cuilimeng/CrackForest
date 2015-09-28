function model = edgesTrain( varargin )
% Train structured edge detector.
%
% For an introductory tutorial please see edgesDemo.m.
%
% USAGE
%  opts = edgesTrain()
%  model = edgesTrain( opts )
%
% INPUTS
%  opts       - parameters (struct or name/value pairs)
%   (1) model parameters:
%   .imWidth    - [32] width of image patches
%   .gtWidth    - [16] width of ground truth patches
%   (2) tree parameters:
%   .nPos       - [5e5] number of positive patches per tree
%   .nNeg       - [5e5] number of negative patches per tree
%   .nImgs      - [inf] maximum number of images to use for training
%   .nTrees     - [8] number of trees in forest to train
%   .fracFtrs   - [1/4] fraction of features to use to train each tree
%   .minCount   - [1] minimum number of data points to allow split
%   .minChild   - [8] minimum number of data points allowed at child nodes
%   .maxDepth   - [64] maximum depth of tree
%   .discretize - ['pca'] options include 'pca' and 'kmeans'
%   .nSamples   - [256] number of samples for clustering structured labels
%   .nClasses   - [2] number of classes (clusters) for binary splits
%   .split      - ['gini'] options include 'gini', 'entropy' and 'twoing'
%   (3) feature parameters:
%   .nOrients   - [4] number of orientations per gradient scale
%   .grdSmooth  - [0] radius for image gradient smoothing (using convTri)
%   .chnSmooth  - [2] radius for reg channel smoothing (using convTri)
%   .simSmooth  - [8] radius for sim channel smoothing (using convTri)
%   .normRad    - [4] gradient normalization radius (see gradientMag)
%   .shrink     - [2] amount to shrink channels
%   .nCells     - [5] number of self similarity cells
%   .rgbd       - [0] 0:RGB, 1:depth, 2:RBG+depth (for NYU data only)
%   (4) detection parameters (can be altered after training):
%   .stride     - [2] stride at which to compute edges
%   .multiscale - [0] if true run multiscale edge detector
%   .sharpen    - [2] sharpening amount (can only decrease after training)
%   .nTreesEval - [4] number of trees to evaluate per location
%   .nThreads   - [4] number of threads for evaluation of trees
%   .nms        - [0] if true apply non-maximum suppression to edges
%   (5) other parameters:
%   .seed       - [1] seed for random stream (for reproducibility)
%   .useParfor  - [0] if true train trees in parallel (memory intensive)
%   .modelDir   - ['models/'] target directory for storing models
%   .modelFnm   - ['model'] model filename
%   .bsdsDir    - ['BSR/BSDS500/data/'] location of BSDS dataset
%
% OUTPUTS
%  model      - trained structured edge detector w the following fields
%   .opts       - input parameters and constants
%   .thrs       - [nNodes x nTrees] threshold corresponding to each fid
%   .fids       - [nNodes x nTrees] feature ids for each node
%   .child      - [nNodes x nTrees] index of child for each node
%   .count      - [nNodes x nTrees] number of data points at each node
%   .depth      - [nNodes x nTrees] depth of each node
%   .eBins      - data structure for storing all node edge maps
%   .eBnds      - data structure for storing all node edge maps
%
% EXAMPLE
%
% See also edgesDemo, edgesChns, edgesDetect, forestTrain
%
% Structured Edge Detection Toolbox      Version 3.01
% Code written by Piotr Dollar, 2014.
% Licensed under the MSR-LA Full Rights License [see license.txt]

% get default parameters
dfs={'imWidth',32, 'gtWidth',16, 'nPos',5e5, 'nNeg',5e5, 'nImgs',inf, ...
  'nTrees',1, 'fracFtrs',1/4, 'minCount',1, 'minChild',8, ...
  'maxDepth',64, 'discretize','pca', 'nSamples',256, 'nClasses',2, ...
  'split','gini', 'nOrients',4, 'grdSmooth',0, 'chnSmooth',2, ...
  'simSmooth',8, 'normRad',4, 'shrink',2, 'nCells',5, 'rgbd',0, ...
  'stride',2, 'multiscale',0, 'sharpen',2, 'nTreesEval',4, ...
  'nThreads',4, 'nms',0, 'seed',1, 'useParfor',0, 'modelDir','models/', ...
  'modelFnm','model', 'bsdsDir','crack image/'};
opts = getPrmDflt(varargin,dfs,1);%Matlab中使用varargin来实现参数可变的函数，赋初始值
if(nargin==0), model=opts; return; end%如果函数输入参数个数为零，结束

% if forest exists load it and return 如果存在训练好的森林，载入
cd(fileparts(mfilename('fullpath')));
forestDir = [opts.modelDir '/forest/'];
forestFn = [forestDir opts.modelFnm];
% if(exist([forestFn '.mat'], 'file'))%注释掉了，原本应该有
%   load([forestFn '.mat']); return; end

% compute constants and store in opts
nTrees=opts.nTrees; nCells=opts.nCells; shrink=opts.shrink;
opts.nPos=round(opts.nPos); opts.nNeg=round(opts.nNeg);%四舍五入到整数
opts.nTreesEval=min(opts.nTreesEval,nTrees);%number of trees to evaluate per location
opts.stride=max(opts.stride,shrink);%步长
imWidth=opts.imWidth; gtWidth=opts.gtWidth;%图像块的大小
imWidth=round(max(gtWidth,imWidth)/shrink/2)*shrink*2;
opts.imWidth=imWidth; opts.gtWidth=gtWidth;
nChnsGrad=(opts.nOrients+1)*2; nChnsColor=3;
if(opts.rgbd==1), nChnsColor=1; end%0:RGB, 1:depth, 2:RBG+depth (for NYU data only)
if(opts.rgbd==2), nChnsGrad=nChnsGrad*2; nChnsColor=nChnsColor+1; end
nChns = nChnsGrad+nChnsColor; opts.nChns = nChns;%3 color, 2 magnitude and 8 orientation channels
opts.nChnFtrs = imWidth*imWidth*nChns/shrink/shrink;%candidate feature x，从文章来看32*32*13/2/2=3328个
opts.nSimFtrs = (nCells*nCells)*(nCells*nCells-1)/2*nChns;%再用C(5*5,2)个特征
opts.nTotFtrs = opts.nChnFtrs + opts.nSimFtrs; disp(opts);%总计7228个特征 %显示数组

% generate stream for reproducibility of model重复能力
stream=RandStream('mrg32k3a','Seed',opts.seed);

% train nTrees random trees (can be trained with parfor if enough memory)并行的for循环
%用parfor的前提条件就是，循环的每次迭代独立，不相互依赖。
if(opts.useParfor), parfor i=1:nTrees, trainTree(opts,stream,i); end%开并行计算
else for i=1:nTrees, trainTree(opts,stream,i); end; end

% merge trees and save model把模型存起来
model = mergeTrees( opts );
if(~exist(forestDir,'dir')), mkdir(forestDir); end
save([forestFn '.mat'], 'model', '-v7.3');

end

function model = mergeTrees( opts )
% accumulate trees and merge into final model合并树
nTrees=opts.nTrees; gtWidth=opts.gtWidth;
treeFn = [opts.modelDir '/tree/' opts.modelFnm '_tree'];
for i=1:nTrees
  t=load([treeFn int2str2(i,3) '.mat'],'tree'); t=t.tree;%int2str2整型转字符串
  %第i棵树
  if(i==1), trees=t(ones(1,nTrees)); else trees(i)=t; end%将八棵树放入一个1*8的向量
end
nNodes=0; for i=1:nTrees, nNodes=max(nNodes,size(trees(i).fids,1)); end%哪个树所用到的特征最多，也就是结点数
% merge all fields of all trees
model.opts=opts; Z=zeros(nNodes,nTrees,'uint32');%返回类型为uint32的nNodes*nTrees的零数组
model.thrs=zeros(nNodes,nTrees,'single');
model.fids=Z; model.child=Z; model.count=Z; model.depth=Z;
model.segs=zeros(gtWidth,gtWidth,nNodes,nTrees,'uint8');%每个单元定义在ground truth块大小下，结点数是多少，树数量是多少
for i=1:nTrees, tree=trees(i); nNodes1=size(tree.fids,1);%fids第一维长度
  model.fids(1:nNodes1,i) = tree.fids;%把八棵树的值都合起来成为8维数组而已
  model.thrs(1:nNodes1,i) = tree.thrs;
  model.child(1:nNodes1,i) = tree.child;
  model.count(1:nNodes1,i) = tree.count;
  model.depth(1:nNodes1,i) = tree.depth;
  model.segs(:,:,1:nNodes1,i) = tree.hs-1;
end
% remove very small segments (<=5 pixels)
segs=model.segs; nSegs=squeeze(max(max(segs)))+1;%求segs中元素最大值，squeeze删去所有只有一行或一列的维度，剩nNodes和nTrees两维
parfor i=1:nTrees*nNodes, m=nSegs(i);%一列一列遍历，找哪棵树上哪个特征是2
  if(m==1), continue; end; S=segs(:,:,i); del=0;
  for j=1:m, Sj=(S==j-1); if(nnz(Sj)>5), continue; end%number of nonzero elements in S.
    S(Sj)=median(single(S(convTri(single(Sj),1)>0))); del=1; end%每一列返回一个值,为M该列的从大到小排列的中间值. 把一个矩阵中所有元素都变为单精度的. Extremely fast 2D image convolution with a triangle filter.
  if(del), [~,~,S]=unique(S); S=reshape(S-1,gtWidth,gtWidth);%unique函数用来去除矩阵A中重复的元素 %把指定的矩阵改变形状,但是元素个数不变
    segs(:,:,i)=S; nSegs(i)=max(S(:))+1; end%S(:)将矩阵一列一列拼起来转化成列向量
end
model.segs=segs; model.nSegs=nSegs;%存放的是每个ymap里面的最大值
% store compact representations of sparse binary edge patches
nBnds=opts.sharpen+1; eBins=cell(nTrees*nNodes,nBnds);%sharpening amount
eBnds=zeros(nNodes*nTrees,nBnds);
parfor i=1:nTrees*nNodes
  if(model.child(i) || model.nSegs(i)==1), continue; end %#ok<PFBNS>
  E=gradientMag(single(model.segs(:,:,i)))>.01; E0=0;%gradientMag 计算图像每个点的梯度幅值 E是把梯度大于1的和等于0的记录下来(16*16)
  for j=1:nBnds, eBins{i,j}=uint16(find(E & ~E0)'-1); E0=E;% %记录每个叶子的ymap中非零的index（错开一个位置）
    eBnds(i,j)=length(eBins{i,j}); E=convTri(single(E),1)>.01; end%记录每个叶子的ymap中非零的个数 %经过模糊后，把大于0的位置记录下来
end
eBins=eBins'; model.eBins=[eBins{:}]';%eBins记录了3种ymap图的非零索引
eBnds=eBnds'; model.eBnds=uint32([0; cumsum(eBnds(:))]);%eBnds记录了3种ymap图的非零索引的个数大小
end

function trainTree( opts, stream, treeInd )
% Train a single tree in forest model.

% location of ground truth
trnImgDir = [opts.bsdsDir '/images/train/'];
trnDepDir = [opts.bsdsDir '/depth/train/'];
trnGtDir = [opts.bsdsDir '/groundTruth/train/'];
imgIds=dir(trnImgDir); imgIds=imgIds([imgIds.bytes]>0);%显示xxx目录下的文件和文件夹 用.bytes提取文件大小 imgIds中那些文件大小>0的文件
imgIds={imgIds.name}; ext=imgIds{1}(end-2:end);%cell 1的内容，继续用小括号引用其内容，ext是后缀名，如jpg、png等等
nImgs=length(imgIds); for i=1:nImgs, imgIds{i}=imgIds{i}(1:end-4); end%做训练的图片数nImgs

% extract commonly used options
imWidth=opts.imWidth; imRadius=imWidth/2;
gtWidth=opts.gtWidth; gtRadius=gtWidth/2;
nChns=opts.nChns; nTotFtrs=opts.nTotFtrs; rgbd=opts.rgbd;%通道数、特征数、颜色数
nPos=opts.nPos; nNeg=opts.nNeg; shrink=opts.shrink;

% finalize setup
treeDir = [opts.modelDir '/tree/'];
treeFn = [treeDir opts.modelFnm '_tree'];
if(exist([treeFn int2str2(treeInd,3) '.mat'],'file'))
  fprintf('Reusing tree %d of %d\n',treeInd,opts.nTrees); return; end
fprintf('\n-------------------------------------------\n');
fprintf('Training tree %d of %d\n',treeInd,opts.nTrees); tStart=clock;

% set global stream to stream with given substream (will undo at end)
streamOrig = RandStream.getGlobalStream();
set(stream,'Substream',treeInd);%训练第几棵树
RandStream.setGlobalStream( stream );

% collect positive and negative patches and compute features
fids=sort(randperm(nTotFtrs,round(nTotFtrs*opts.fracFtrs)));%randperm(n)随机打乱一个数字序列 总特征数 四舍五入总特征数*fraction of features to use to train each tree
% p = randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n inclusive.
k = nPos+nNeg; nImgs=min(nImgs,opts.nImgs);%做训练的图片数nImgs
ftrs = zeros(k,length(fids),'single');
labels = zeros(gtWidth,gtWidth,k,'uint8'); k = 0;
tid = ticStatus('Collecting data',30,1);% Used to display the progress of a long process.
for i = 1:nImgs%对每个做训练的图片
  % get image and compute channels
  gt=load([trnGtDir imgIds{i} '.mat']); gt=gt.groundTruth;%Ground Truth，是标记的，不是图片
  I=imread([trnImgDir imgIds{i} '.' ext]); siz=size(I);%读取原图 返回一个行向量，该行向量的第一个元素是矩阵的行数，第二个元素是矩阵的列数。
  if(rgbd), D=single(imread([trnDepDir imgIds{i} '.png']))/1e4; end%0:RGB, 1:depth, 2:RBG+depth (for NYU data only)
  if(rgbd==1), I=D; elseif(rgbd==2), I=cat(3,single(I)/255,D); end%rgbd默认是0，无视这三行
  p=zeros(1,4); p([2 4])=mod(4-mod(siz(1:2),4),4);%p的第二个和第四个元素
  if(any(p)), I=imPad(I,p,'symmetric'); end%检测矩阵中是否有非零元素，如果有，则返回1，否则，返回0。 填充图像或填充数组。 图像大小通过围绕边界进行镜像反射来扩展。
  [chnsReg,chnsSim] = edgesChns(I,opts);% Compute features for structured edge detection.
  % 出来regular output channels和self-similarity output channels大小均为为原图长宽减半的矩阵
  % sample positive and negative locations
  nGt=length(gt); xy=[]; k1=0; B=false(siz(1),siz(2));%B为和原图大小一样全为0的矩阵
  B(shrink:shrink:end,shrink:shrink:end)=1;%shrink为2，B偶数行的偶数元素为1
  B([1:imRadius end-imRadius:end],:)=0;%imRadius为17，把B前shrink行和末shrink行置为0
  B(:,[1:imRadius end-imRadius:end])=0;%把B前shrink列和末shrink列置为0
  for j=1:nGt%对每个图像的每个标注的ground truth
    %M=gt{j}.Boundaries;
    M=gt(j).Boundaries;%运行自己标注的数据时
    M(bwdist(M)<gtRadius)=1;%每个值为0的像素点到非零的像素点的距离
    %直观上看把标注的边缘周围一定距离的点都置为边缘，边缘“粗”了
    [y,x]=find(M.*B); k2=min(length(y),ceil(nPos/nImgs/nGt));%find找到非零元素，返回行列坐标，.*表示矩阵中元素与元素相乘，这两个矩阵的维数必需相同。ceil取整。 %k2是每幅图像需要采集的样本点个数
    rp=randperm(length(y),k2); y=y(rp); x=x(rp);%把y和x都随机只留下k2个点
    xy=[xy; x y ones(k2,1)*j]; k1=k1+k2; %#ok<AGROW> %正类点的x一列，y一列，ground truth序号一列，存在xy矩阵中
    [y,x]=find(~M.*B); k2=min(length(y),ceil(nNeg/nImgs/nGt));
    rp=randperm(length(y),k2); y=y(rp); x=x(rp);
    xy=[xy; x y ones(k2,1)*j]; k1=k1+k2; %#ok<AGROW>%正类点的x和负类点的x一列，正类点的y和负类点的y一列，ground truth序号一列，存在xy矩阵中
  end
  if(k1>size(ftrs,1)-k), k1=size(ftrs,1)-k; xy=xy(1:k1,:); end%保证不超出总个数
  % crop patches and ground truth labels
  psReg=zeros(imWidth/shrink,imWidth/shrink,nChns,k1,'single');%每个点都对应着16*16*13的一个特征。k1是样本点的个数。
  lbls=zeros(gtWidth,gtWidth,k1,'uint8');%表示具体通道的一个样本点。
  psSim=psReg; ri=imRadius/shrink; rg=gtRadius;
  for j=1:k1, xy1=xy(j,:); xy2=xy1/shrink;%k1为样本个数
    psReg(:,:,:,j)=chnsReg(xy2(2)-ri+1:xy2(2)+ri,xy2(1)-ri+1:xy2(1)+ri,:);%把选定像素点对应的特征取出来存放在psReg和psSim中 
    psSim(:,:,:,j)=chnsSim(xy2(2)-ri+1:xy2(2)+ri,xy2(1)-ri+1:xy2(1)+ri,:);
    %t=gt{xy1(3)}.Segmentation(xy1(2)-rg+1:xy1(2)+rg,xy1(1)-rg+1:xy1(1)+rg);
    t=gt(xy1(3)).Segmentation(xy1(2)-rg+1:xy1(2)+rg,xy1(1)-rg+1:xy1(1)+rg);%运行自己标注的数据时 %取出的原始分割图像
    if(all(t(:)==t(1))), lbls(:,:,j)=1; else [~,~,t]=unique(t);
      lbls(:,:,j)=reshape(t,gtWidth,gtWidth); end%把局部块儿的分割图放进来(16*16)
  end
  if(0), figure(1); montage2(squeeze(psReg(:,:,1,:))); drawnow; end% Used to display collections of images and videos.
  if(0), figure(2); montage2(lbls(:,:,:)); drawnow; end
  % compute features and store
  ftrs1=[reshape(psReg,[],k1)' stComputeSimFtrs(psSim,opts)];%所有的特征连起来，文章中提到一共7228个
  ftrs(k+1:k+k1,:)=ftrs1(:,fids); labels(:,:,k+1:k+k1)=lbls;
  k=k+k1; if(k==size(ftrs,1)), tocStatus(tid,1); break; end
  tocStatus(tid,i/nImgs);
end
if(k<size(ftrs,1)), ftrs=ftrs(1:k,:); labels=labels(:,:,1:k); end

% train structured edge classifier (random decision tree)
pTree=struct('minCount',opts.minCount, 'minChild',opts.minChild, ...
  'maxDepth',opts.maxDepth, 'H',opts.nClasses, 'split',opts.split);
t=labels; labels=cell(k,1); for i=1:k, labels{i}=t(:,:,i); end%把label做个变形，一共6000*1个cell，每个cell中是16*16 unit8
pTree.discretize=@(hs,H) discretize(hs,H,opts.nSamples,opts.discretize);%每个节点都需要做一次discretize
tree=forestTrain(ftrs,labels,pTree);
tree.hs=cell2array(tree.hs);%dollar toolbox中一个函数
tree.fids(tree.child>0) = fids(tree.fids(tree.child>0)+1)-1;
if(~exist(treeDir,'dir')), mkdir(treeDir); end
save([treeFn int2str2(treeInd,3) '.mat'],'tree'); e=etime(clock,tStart);
fprintf('Training of tree %d complete (time=%.1fs).\n',treeInd,e);
RandStream.setGlobalStream( streamOrig );

end

function ftrs = stComputeSimFtrs( chns, opts )
% Compute self-similarity features (order must be compatible w mex file).
w=opts.imWidth/opts.shrink; n=opts.nCells; if(n==0), ftrs=[]; return; end
nSimFtrs=opts.nSimFtrs; nChns=opts.nChns; m=size(chns,4);%chns第四维有多少值
inds=round(w/n/2); inds=round((1:n)*(w+2*inds-1)/(n+1)-inds+1);
chns=reshape(chns(inds,inds,:,:),n*n,nChns,m);
ftrs=zeros(nSimFtrs/nChns,nChns,m,'single');
k=0; for i=1:n*n-1, k1=n*n-i; i1=ones(1,k1)*i;
  ftrs(k+1:k+k1,:,:)=chns(i1,:,:)-chns((1:k1)+i,:,:); k=k+k1; end%本来ftrs只有(1:k,:,:)有数
ftrs = reshape(ftrs,nSimFtrs,m)';
end

% ----------------------------------------------------------------------- %
% 2015/01/05 cuilimeng
% 改动：给discretize函数加了tmpsegs变量
% ----------------------------------------------------------------------- %
function [hs,segs,tmpsegs] = discretize( segs, nClasses, nSamples, type )
% Convert a set of segmentations into a set of labels in [1,nClasses].
tmpsegs=segs;%增加的tmpsegs变量
persistent cache;
w=size(segs{1},1); assert(size(segs{1},2)==w);%persistent定义静态变量 %assert 由于要求对参数的保护，需要对输入参数或处理过程中的一些状态进行判断，判断程序能否/是否需要继续执行。
if(~isempty(cache) && cache{1}==w), [~,is1,is2]=deal(cache{:}); else %deal(X) 将单个输入数据赋值给所有输出参数
  % compute all possible lookup inds for w x w patches
  is=1:w^4; is1=floor((is-1)/w/w); is2=is-is1*w*w; is1=is1+1;%floor 向地板方向取整
  kp=is2>is1; is1=is1(kp); is2=is2(kp); cache={w,is1,is2};%kp是一个和is1、is2一样大的数组，每位判断is1是否大于is2
end%其实这段只是遍历了1~256间，一个数大于另一个数的组合，只要文件不删，不用重新算；其实就是16*16中任意两个像素点
% compute n binary codes zs of length nSamples
nSamples=min(nSamples,length(is1)); kp=randperm(length(is1),nSamples);%随机取256个俩像素的组合（nSamples）
n=length(segs); is1=is1(kp); is2=is2(kp); zs=false(n,nSamples);%n是采样点的个数，zs为全logic 0矩阵
for i=1:n, zs(i,:)=segs{i}(is1)==segs{i}(is2); end%第i个组合中的两个像素点属不属于同一个seg，属于算1，不属于算0
zs=bsxfun(@minus,zs,sum(zs,1)/n); zs=zs(:,any(zs,1));%n为正负类点总个数 %sum(zs,1)按列求和 %sum(zs,1)/n算每列平均有多少属于相同seg的组合，一个double的数，将sum(zs,1)/n复制6000行，用zs去减它，相当于zs变成了每个值对每列平均值的偏差 %any：当向量中的元素有非零元素时返回值为1，any(A, 1)表示矩阵A的列向量判断
if(isempty(zs)), hs=ones(n,1,'uint32'); segs=segs{1}; return; end%没用，把源代码中
% find most representative segs (closest to mean)
[~,ind]=min(sum(zs.*zs,2)); segs=segs{ind};%zs.*zs表示za矩阵每个元素求平方；sum(x,2)表示矩阵x的横向相加，求每行的和，结果是列向量。 %取到了偏差最小的那个seg，也就是每个元素最接近平均值，该seg最具有代表性
% apply PCA to reduce dimensionality of zs
U=pca(zs'); d=min(5,size(U,2)); zs=zs*U(:,1:d);%把zs转置，有256(nSample)*6000，算法将256降维成5 %可理解为zs对每个主要维度的权重
% discretize zs by clustering or discretizing pca dimensions
d=min(d,floor(log2(nClasses))); hs=zeros(n,1);%由于要把这些划分分为两类，d=1；hs为6000*1的零矩阵
for i=1:d, hs=hs+(zs(:,i)<0)*2^(i-1); end%每回加上zs的第i列的小于零元素乘以2^(i-1) %由于d=1，只算i=1，于是把zs第一维小于零的元素的行数在hs中置为1（也就是对第一个主成份负相关的样本点？）
[~,~,hs]=unique(hs); hs=uint32(hs);%unique(hs)出来[0 1]，hs变为原来向量在[0 1]中的位置；其实就是一个把原来向量全部+1的高大上方式；能将原来的seg分成两个类别，输出了类标号~~
if(strcmpi(type,'kmeans'))
  nClasses1=max(hs); C=zs(1:nClasses1,:);
  for i=1:nClasses1, C(i,:)=mean(zs(hs==i,:),1); end
  hs=uint32(kmeans2(zs,nClasses,'C0',C,'nIter',1));
end
% optionally display different types of hs
%for i=1:2, figure(i); montage2(cell2array(segs(hs==i))); end
%figure(3); imshow(seg);
end
