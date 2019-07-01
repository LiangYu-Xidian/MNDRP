Nets = {'drugTarget'};
	tic
	%inputID = char(strcat('../data/', Nets(i), '.csv'));
	M = load('network\\CCLE_24drugTargetMatrix.csv');
	Sim = 1 - pdist(M, 'jaccard');%jaccard
	Sim = squareform(Sim);
	Sim = Sim + eye(size(M,1));
	Sim(isnan(Sim)) = 0;
	outputID = char(strcat('network/Sim_', Nets(1), '.txt'));
	dlmwrite(outputID, Sim, '\t');
	toc

