data1=load('C:\mcode\edges\BSR\BSDS500\data\groundTruth\train\100.mat');
% a=data1.groundTruth{1, 5}.Boundaries~=0;
% [i,j]=find(a);
% [height,width]=size(a);
% plot(j,i,'.')
% axis([0 width 0 height])
% axis equal
% set(gca,'YDir','reverse')%对Y方向反转
imshow(data1.groundTruth.Boundaries)

% data2=load('C:\Users\lenovo\Desktop\公路裂纹\标注结果\0004.seg');
% [height,width]=size(data2);
% A=data2;
% i=1;
% while i<height;
%     if A(i,3)>=0 && A(i,3)<=10;
%         A(i,:)=[];
%     else
%         i=i+1;
%     end
%     [height,width]=size(A);
% end
% x1=A(:,2);
% y1=A(:,3);
% B=data2;
% i=1;
% while i<height;
%     if B(i,4)>=471 && B(i,4)<=481 || B(i,4)>=311 && B(i,4)<=321;
%         B(i,:)=[];
%     else
%         i=i+1;
%     end
%     [height,width]=size(B);
% end
% x2=B(:,2);
% y2=B(:,4);
% plot(y1,x1,'.',y2,x2,'.')
% axis([0 481 0 481])
% axis equal
% set(gca,'YDir','reverse')%对Y方向反转
