clear all
seed = 1;
randn('state',seed);
rand('state',seed);

data_frac = 0.2;%0.2;
num_edges = 1000000;
problem_confidence_func = get_yahoo_problem_func(data_frac);

y_constraint_frac = 0.2;
x_constraint_frac = 0.2;%20;
eps = 0.01;

confidences = [0.95];%fliplr([0.99 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2])

for ii=1:length(confidences)
    confidence = confidences(ii);
    
    
    problem = problem_confidence_func(confidence);
    
    var_edges_tic = tic;
    var_edges = choose_var_edges(problem, eps, y_constraint_frac*problem.S, num_edges);
    var_edges_time = toc(var_edges_tic);
    fprintf('Choosing edges took %0.1f seconds\n', var_edges_time);
    
    % subsample which x's we will allow robustness on
    %x_var_indices = randsample(find(problem.real_edges), 60);
    %var_edges = false(size(problem.real_edges));
    %var_edges(x_var_indices) = true;
    problem.var_edges = var_edges;
    
    num_edge_vars = nnz(problem.var_edges);
    
    test_param.x_constraint = x_constraint_frac*num_edge_vars;
    test_param.y_constraint = y_constraint_frac*problem.S;
    test_param.eps = eps;
    test_param.verbose = 2;
    test_param.maxiters = 200;
    test_param.problem = problem;
    test_param.uncertainty_set_type = UncertaintySetType.Dnorm;
    test_param.adversary_algorithm = AdversaryAlgorithm.SubmodularMinimization;
    
    test_params(ii) = test_param;
end
    
run_all_params(test_params);