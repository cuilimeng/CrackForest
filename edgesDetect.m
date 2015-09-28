function [E,O,inds,segs] = edgesDetect( I, model )
% Detect edges in image.
%
% For an introductory tutorial please see edgesDemo.m.
%
% The following model params may be altered prior to detecting edges:
%  prm = stride, sharpen, multiscale, nTreesEval, nThreads, nms
% Simply alter model.opts.prm. For example, set model.opts.nms=1 to enable
% non-maximum suppression. See edgesTrain for parameter details.
%
% USAGE
%  [E,O,inds,segs] = edgesDetect( I, model )
%
% INPUTS
%  I          - [h x w x 3] color input image
%  model      - structured edge model trained with edgesTrain
%
% OUTPUTS
%  E          - [h x w] edge probability map
%  O          - [h x w] coarse edge normal orientation (0=left, pi/2=up)
%  inds       - [h/s x w/s x nTreesEval] leaf node indices
%  segs       - [g x g x h/s x w/s x nTreesEval] local segmentations
%
% EXAMPLE
%
% See also edgesDemo, edgesTrain, edgesChns
%
% Structured Edge Detection Toolbox      Version 3.01
% Code written by Piotr Dollar, 2014.
% Licensed under the MSR-LA Full Rights License [see license.txt]

% get parameters
opts=model.opts; opts.nTreesEval=min(opts.nTreesEval,opts.nTrees);
if(~isfield(opts,'sharpen')), opts.sharpen=0; end
if(~isfield(model,'segs')), model.segs=[]; model.nSegs=[]; end
opts.stride=max(opts.stride,opts.shrink); model.opts=opts;

if( opts.multiscale )
  % if multiscale run edgesDetect multiple times
  ss=2.^(-1:1); k=length(ss); inds=cell(1,k); segs=inds;
  siz=size(I); model.opts.multiscale=0; model.opts.nms=0; E=0;
  for i=1:k, s=ss(i); I1=imResample(I,s);
    if(nargout<4), [E1,~,inds{i}]=edgesDetect(I1,model);
    else [E1,~,inds{i},segs{i}]=edgesDetect(I1,model); end
    E=E+imResample(E1,siz(1:2));
  end; E=E/k; model.opts=opts;
  
else
  % pad image, making divisible by 4
  siz=size(I); r=opts.imWidth/2; p=[r r r r];
  p([2 4])=p([2 4])+mod(4-mod(siz(1:2)+2*r,4),4);
  if(0), figure(1); imshow(I); end
  I = imPad(I,p,'symmetric');
  if(0), figure(2); imshow(I); end
  
  % compute features and apply forest to image
  [chnsReg,chnsSim] = edgesChns( I, opts );
  s=opts.sharpen; if(s), I=convTri(rgbConvert(I,'rgb'),1); end%将图片归一化
  if(nargout<4), [E,inds] = edgesDetectMex(model,I,chnsReg,chnsSim);%在edgesDetect前已经算好特征了
  else [E,inds,segs] = edgesDetectMex(model,I,chnsReg,chnsSim); end%此时E是一个每个元素都大于1的矩阵
  % --------------------------------------------------------------------- %
  % 2015/01/19 cuilimeng
  % 功能：将每棵树的叶结点上的划分输出
  % 第一个是输出叶结点上最有代表性的划分，第二个是输出叶结点上所有划分的均值。
  y=length(find(inds==1));
  for i=2:1:length(model.fids(:, 1)),
      if(max(max(model.segs(:,:,i,1)))==0), y=[y 0];
      else y=[y length(find(inds==i))]; end
  end;
  [~, tmp]=max(y);%tmp存的是出现次数最多的ymap的序号
  [b,c]=sort(y(:),'descend');%b是出现次数，c是ymap序号
%   y_all=evalin('base','y_all');
%   y_all=y_all+y;
%   y_all=y;
%   assignin('base','y_all',y_all);
  MyColorMap=1.0/255*[
    1,135,250
    254,84,209
    206,92,249
    101,80,245
    40,241,125
    255,128,45
    255,206,52
    254,62,39
    131,241,56
    222,243,64
    
    1,135,250
    254,84,209
    206,92,249
    101,80,245
    40,241,125
    255,128,45
    255,206,52
    254,62,39
    131,241,56
    222,243,64
    
    1,135,250
    254,84,209
    206,92,249
    101,80,245
    40,241,125
    255,128,45
    255,206,52
    254,62,39
    131,241,56
    222,243,64];
  if(0), figure('NumberTitle', 'off', 'Name', '直方图1'); bar(y);
       %for i=1:10, text(i, b(i), ['(',num2str(c(i)),')']); end;%c为ymap编号，只标前10个
       axis on;
       axis([0,26443,0,120]);
       xlabel('Token Number'); ylabel('Occurrence');
       print('-depsc', ['010_histogram_1', '.eps']);
  end
  if(0), figure('NumberTitle', 'off', 'Name', '出现最频繁的10个token');
      for i=1:10,
          subplot(2,5,i);
          imshow(model.segs(:,:,c(i),1)*255);
          print('-depsc', ['010_most_frequent_tokens', '.eps']);
      end;
  end
  % --------------------------------------------------------------------- %
  
  % normalize and finalize edge maps
  t=opts.stride^2/opts.gtWidth^2/opts.nTreesEval; r=opts.gtWidth/2;
  if(s==0), t=t*2; elseif(s==1), t=t*1.8; else t=t*1.66; end
  E=E(1+r:siz(1)+r,1+r:siz(2)+r,:)*t; E=convTri(E,1);
end

% compute approximate orientation O from edges E
if( opts.nms==-1 ), O=[]; elseif( nargout>1 || opts.nms )
  [Ox,Oy]=gradient2(convTri(E,4));
  [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
  O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
end

% perform nms
if( opts.nms>0 ), E=edgesNmsMex(E,O,1,5,1.01,opts.nThreads); end

end
