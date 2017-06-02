clear all

eps = 0.0001;
prob_edge = 1;

y_constraint_frac = 0.2;
S = 6;
T = 2;

seed = 2;
randn('state',seed);
rand('state',seed);

max_num_trials = 5;
confidence = 0.95;

[~, problem_confidence_func] = random_bipartite_problem_from_trials(S, T, prob_edge, max_num_trials);
problem = problem_confidence_func(confidence);

x_constraints = (linspace(0.8,0.01,80)*nnz(problem.var_edges));
for ii=1:length(x_constraints)
    test_param.x_constraint = x_constraints(ii);
    test_param.y_constraint = y_constraint_frac*T;
    test_param.eps = eps;
    test_param.verbose = 1;
    test_param.maxiters = 200;
    test_param.problem = problem;
    test_param.uncertainty_set_type = UncertaintySetType.Dnorm;
    test_param.adversary_algorithm = AdversaryAlgorithm.SubmodularMinimization;
    
    test_params(ii) = test_param;
end

run_all_params(test_params);