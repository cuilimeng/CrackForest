function edgesDemo
% Demo for Structured Edge Detector (please see readme.txt first).

%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
% opts.nPos=3e3; opts.nNeg=3e3;%正负样本各取3000个
opts.useParfor=0;                 % parallelize if sufficient memory
opts.nTrees = 8;%八棵树
%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=1;          % for top speed set nTreesEval=1 从4改成1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval( model, 'show',1, 'name','' ); end

%% detect edge and visualize results
picNo = '051';
I = imread(['crack image/images/train/', picNo, '.jpg']);
tic, [E,O,inds,segs]=edgesDetect(I,model); toc
[edge_inst,edge_label] = detection(E,inds,I,picNo,model,opts);
picNo

end

function [edge_inst,edge_label] = detection(E,inds,I,picNo,model,opts,edge_inst,edge_label)
if(1), figure('NumberTitle', 'off', 'Name', '原图'); imshow(I); print('-depsc', [num2str(picNo), '_original.eps']); end;
if(0), figure('NumberTitle', 'off', 'Name', '边缘检测结果'); imshow(1-E); print('-depsc', [num2str(picNo), '_edgedetection.eps']); end;

G=load('crack image/groundTruth/train/051.mat');
g=G.groundTruth.Boundaries();
box on; imshow(1-g); print('-depsc', [picNo, '.eps']);

SE1 = strel('rectangle',[4 4]);
BW1=1-E;
BW2=imerode(BW1, SE1); if(0), figure(5); imshow(BW2); end%腐蚀
SE2 = strel('rectangle',[4 4]);
BW3=imdilate(BW2, SE2); if(0), figure(6); imshow(BW3); end%膨胀
BW3(BW3<0.9)=0; BW3(BW3>0.9)=1; if(0), figure(7); imshow(BW3); print('-depsc', [num2str(picNo), '_腐蚀膨胀+阈值.eps']);end%腐蚀膨胀处理后的二值图
BW1(BW1<0.9)=0; BW1(BW1>0.9)=1; if(0), figure(8); imshow(BW1); print('-depsc', [num2str(picNo), '_阈值.eps']);end%边缘检测出来的二值图

MyColorMap=1.0/255*[
    255,255,255
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

[L, num] = bwlabel(1-BW1);%BW1是边缘检测出来的二值图
[r,c] = size(L);
stats = regionprops(L);
Ar = cat(1, stats.Area);

tmpL1 = zeros(r,c);
[p, ind] = sort(Ar, 'descend');
for i=1:length(ind),
    if(Ar(ind(i))>=10), tmpL1(L==ind(i))=i+1; end;%i+1只不过是为了从第二个颜色开始标颜色
end;
if(0), figure('NumberTitle', 'off', 'Name', '二值图'); image(tmpL1); colormap(MyColorMap); axis off; print('-depsc', [num2str(picNo), '_阈值.eps']); end;%加了ColorMap

[L, num] = bwlabel(1-BW3);%BW3是膨胀腐蚀处理后的二值图
[r,c] = size(L);
stats = regionprops(L);
Ar = cat(1, stats.Area);

tmpL1 = zeros(r,c);
[p, ind] = sort(Ar, 'descend');
for i=1:length(ind),
    if(Ar(ind(i))>=10), tmpL1(L==ind(i))=i+1; end;%i+1只不过是为了从第二个颜色开始标颜色
end;
if(0), figure('NumberTitle', 'off', 'Name', '腐蚀膨胀处理后的二值图'); image(tmpL1); colormap(MyColorMap); axis off; print('-depsc', [num2str(picNo), '_erode_dilate.eps']); end;%加了ColorMap，腐蚀膨胀处理后的图

tmpL1 = zeros(r,c);
[p, ind] = sort(Ar, 'descend');
for i=1:length(ind),
    if(Ar(ind(i))>=1000), tmpL1(L==ind(i))=i+1; end;%i+1只不过是为了从第二个颜色开始标颜色
end;
if(1), figure('NumberTitle', 'off', 'Name', '腐蚀膨胀处理后的二值图'); image(tmpL1); colormap(MyColorMap); axis off; print('-depsc', [num2str(picNo), '_crack.eps']); end;%加了ColorMap，腐蚀膨胀处理后的图
% ----------------------------------------------------------------------- %
% 2015/02/06 cuilimeng
% 功能：识别出图中是裂纹的区域，现在tmpL1是最终识别出来的裂纹
%提取出数据库中的groundtruth
trnImgDir = [opts.bsdsDir '/images/train/']; trnGtDir = [opts.bsdsDir '/groundTruth/train/'];
imgIds=dir(trnImgDir); imgIds=imgIds([imgIds.bytes]>0);
imgIds={imgIds.name}; ext=imgIds{1}(end-2:end);
nImgs=length(imgIds); for i=1:nImgs, imgIds{i}=imgIds{i}(1:end-4); end
gt=load([trnGtDir imgIds{i} '.mat']); gt=gt.groundTruth;
nGt=length(gt);%每个图片groundtruth的个数
for i=1:length(ind),
    %创建label和inst矩阵，label是区域是否是裂纹，inst是该区域的token分布
    if(~exist('edge_label', 'var')), edge_label=zeros(1, 1); else edge_label=[edge_label; zeros(1, 1)]; end%在edge_label增加一行（多少个ind(裂纹区域数)就有多少行），每行列数为1，值为＋1或－1，区分是否为真的裂纹
    if(~exist('edge_inst', 'var')), edge_inst=zeros(1, length(model.count)); else edge_inst=[edge_inst; zeros(1, length(model.count))]; end%在edge_inst增加一行，多少个ind(裂纹区域数)就有多少行，每行列数为token数，值为token的权重（出现次数）
    tmp=zeros(r, c);
    tmp(L==i)=1;%提取出图中属于同一裂纹区域的像素点
    %与groundtruth对比，只要这块裂纹能覆盖到groundtruth的数据点，那么这块裂纹属于真
    flag = 0;%标记变量，看这块区域是不是包含groundtruth中的点
    if(nGt==1),%运行自己标注的数据时
        M=double(gt(1).Boundaries);
        for k=1:numel(tmp), if(tmp(k)==M(k)&&tmp(k)==1), flag=1; break; end; end;
    end
    %接下来计算覆盖到的、在groundtruth中属于裂纹的数据点，在inds的token标号，这样就可以求得这块裂纹的token分布，要用tmp算
    [r_edge_inst,~] = size(edge_inst);
    if flag, edge_label(r_edge_inst) = 1; else edge_label(r_edge_inst) = -1; end
    [r_tmp,c_tmp]=size(tmp);
    for k=1:r_tmp,
        for l=1:c_tmp, edge_inst(r_edge_inst,inds(floor((k+1)/2),floor((l+1)/2)))=edge_inst(r_edge_inst,inds(floor((k+1)/2),floor((l+1)/2)))+1; end;
    end
end
end
% ----------------------------------------------------------------------- %