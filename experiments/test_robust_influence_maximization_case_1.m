clear;

S = 2;
T = 1;

x_median = zeros(S,T);
x_median(1,1) = 0.5;
x_median(2,1) = 0.8;

x_upper = zeros(S,T);
x_upper(1,1) = 1;
x_upper(2,1) = 0.8001;

x_lower = zeros(S,T);
x_lower(1,1) = 0;
x_lower(2,1) = 0.79;

problem.x_lower = x_lower;
problem.x_upper = x_upper;
problem.x_median = x_median;
problem.x_centers_empirical = x_median;
problem.real_edges = true(size(x_upper));
problem.var_edges = problem.real_edges; % all edges are vars
problem.S = S;
problem.T = T;

test_param.x_constraint = 1;
test_param.y_constraint = 1;
test_param.eps = 0.0001;
test_param.verbose = 1;
test_param.maxiters = 200;
test_param.problem = problem;
test_param.uncertainty_set_type = UncertaintySetType.Dnorm;
test_param.adversary_algorithm = AdversaryAlgorithm.SubmodularMinimization;

run_all_params(test_param);