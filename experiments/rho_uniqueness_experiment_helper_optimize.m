%% setup the tests
clear all

eps = 0.001;
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

% now that the problem is fixed, create test_param structs for each test we
% wish to run

% Dnorm uncertainty, for small and large budgets
x_constraint = 0.2*nnz(problem.var_edges);
test_param.x_constraint = x_constraint;
test_param.y_constraint = y_constraint_frac*T;
test_param.eps = eps;
test_param.verbose = 1;
test_param.maxiters = 200;
test_param.problem = problem;
test_param.uncertainty_set_type = UncertaintySetType.Dnorm;
test_param.adversary_algorithm = AdversaryAlgorithm.SubmodularMinimization;
test_params(1) = test_param;

test_param.y_constraint = 2*T;
test_params(2) = test_param;

% Ellipsoidal uncertainty, for small and large budgets
test_param.x_constraint = 2*nnz(problem.var_edges);
test_param.y_constraint = y_constraint_frac*T;
test_param.uncertainty_set_type = UncertaintySetType.Ellipsoidal;
test_params(3) = test_param;

test_param.y_constraint = 2*T;
test_params(4) = test_param;


%% do the experiment runs
for ii=1:length(test_params)
    [ summary, out_struct_max ] = test_robust_influence_maximization_instance(test_params(ii));
    summaries(ii) = summary;
    out_structs(ii) = out_struct_max;
end