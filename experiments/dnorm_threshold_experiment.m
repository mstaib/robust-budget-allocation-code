PARALLEL=0; % for pMATLAB

eps = 0.001;
prob_edge = 1;

y_constraint_fracs = [0.2 0.9 2 4];
S = 6;
T = 2;

seed = 2;
randn('state',seed);
rand('state',seed);

max_num_trials = 5;
confidence = 0.95;

[~, problem_confidence_func] = random_bipartite_problem_from_trials(S, T, prob_edge, max_num_trials);
problem = problem_confidence_func(confidence);
problem.alpha = alpha;

x_constraints = (linspace(0.8,0.01,80)*nnz(problem.var_edges));
k = 1
for ii=1:length(x_constraints)
    for y_constraint_frac=y_constraint_fracs
	    test_param.x_constraint = x_constraints(ii);
	    test_param.y_constraint = y_constraint_frac*T;
	    test_param.eps = eps;
	    test_param.verbose = 1;
	    test_param.maxiters = 200;
	    test_param.problem = problem;
	    test_param.uncertainty_set_type = UncertaintySetType.Dnorm;
	    test_param.adversary_algorithm = AdversaryAlgorithm.SubmodularMinimization;
	    
	    test_params(k) = test_param;
   	    k = k+1;
    end
end

n = length(test_params);
if PARALLEL
    myIndices = global_ind(zeros(n,1,map([Np 1],'c',0:Np-1)));
    myIndices
else
    myIndices = 1:n;
end

run_all_params(test_params(myIndices));