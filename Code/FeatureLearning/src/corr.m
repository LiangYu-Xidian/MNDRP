close all;
clear;
%导入数据
exp=load('1111.txt');


exp_nor=[];

%for i=2:size(exp,1)
%	exp_nor = [exp_nor; zscore(exp(i,:))];
%end

[corr,p_value]=corrcoef(exp_nor');
[row,column]=find(p_value<0.01);
index=size(row,1);
corr_pearson=zeros(m,n);
for i=1:index
	aa=row(i,1);
	bb=column(i,1);
    if(corr(aa,bb)>0)
        corr_pearson(aa,bb)=corr(aa,bb);
    end
end

save corr corr
save corr_pearson corr_pearson
save p_value p_value
