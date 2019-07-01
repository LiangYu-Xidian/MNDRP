
maxiter = 20;
restartProb = 0.50;
dim_cell = [10,15,50,100,500];%dimension of cellLine feature




%drugNets = {'Sim_mat_drug_drug', 'Sim_mat_drug_disease', 'Sim_mat_drug_se', 'Sim_mat_Drugs'};
%proteinNets = {'Sim_protein_protein', 'Sim_mat_protein_disease', 'Sim_mat_Proteins'};

%drugNets = {'Sim_219DrugTarget',  'Similarity_Matrix_Drugs'};

%four networks
%cellLineNets = {'CNA', 'EXP', 'mss', 'Pathway'};

%three networks
%cellLineNets = {'CNA', 'EXP', 'mss'};%cem
% cellLineNets = {'CNA', 'EXP', 'Pathway'};%cep
%cellLineNets = {'CNA', 'mss', 'Pathway'};%cmp
%cellLineNets = {'EXP', 'mss', 'Pathway'};%emp

%two networks
%cellLineNets = {'CNA', 'EXP'};%ce
%cellLineNets = {'CNA',  'mss'};%cm
%cellLineNets = {'CNA', 'Pathway'};%cp
%cellLineNets = {'EXP', 'mss'};%em
%cellLineNets = { 'EXP', 'Pathway'};%ep
%cellLineNets = { 'mss', 'Pathway'};%mp

%one network
%cellLineNets = {'CNA'};
%cellLineNets = { 'EXP'};
%cellLineNets = { 'mss'};
cellLineNets = {'corr_urogenital_system'};


for j=1:size(dim_cell,2)
    tic
    Y = DCA(cellLineNets, dim_cell(1,j), restartProb, maxiter);
    toc
	dlmwrite(['feature/cellLine_vector_d', num2str(dim_cell(1,j)), '.txt'], Y, '\t');
end


