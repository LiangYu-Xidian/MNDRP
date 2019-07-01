close all;
clear;
chem=load('network\\FinalCellEXP.csv');
[corr,p_value]=corrcoef(chem');
[row,column]=find(p_value<0.05);
index=size(row,1);
corr_pearson=zeros(size(chem,1));
for i=1:index
	aa=row(i,1);
	bb=column(i,1);
   % if(corr(aa,bb)>0)
     corr_pearson(aa,bb)=corr(aa,bb);
    %end
end

save corr corr
save corr_pearson corr_pearson
save p_value p_value
dlmwrite(['Sim_CellExp.txt'], corr_pearson, '\t');
