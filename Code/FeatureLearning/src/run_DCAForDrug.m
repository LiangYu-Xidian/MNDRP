maxiter = 20;
restartProb = 0.50;
dim_drug = [10,15,50,100];%dimension of drug feature




%drugNets = {'Sim_mat_drug_drug', 'Sim_mat_drug_disease', 'Sim_mat_drug_se', 'Sim_mat_Drugs'};
%proteinNets = {'Sim_protein_protein', 'Sim_mat_protein_disease', 'Sim_mat_Proteins'};

%drugNets = { 'Sim_219DrugTarget','Similarity_Matrix_Drugs'};
%drugNets = { 'Similarity_Matrix_Drugs'};
drugNets = { 'Sim_DrugChem','Sim_drugTarget'};



for i=1:size(dim_drug,2)
    tic
    X = DCA(drugNets, dim_drug(1,i), restartProb, maxiter);
    toc
	dlmwrite(['feature/drug_vector_d', num2str(dim_drug(1,i)), '.txt'], X, '\t');
	
end



