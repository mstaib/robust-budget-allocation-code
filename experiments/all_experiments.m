clear all

eps = 0.001;
prob_edge = 1;

y_constraint_fracs = logspace(log10(0.2), log10(10), 8);
S = 6;
T = 2;

seed = 2;
randn('state',seed);
rand('state',seed);

max_num_trials = 5;
confidence = 0.95;

[~, problem_confidence_func] = random_bipartite_problem_from_trials(S, T, prob_edge, max_num_trials);
problem = problem_confidence_func(confidence);

kk = 1;
for ust=[UncertaintySetType.Dnorm, ...
         UncertaintySetType.Ellipsoidal]

     if ust == UncertaintySetType.Dnorm
         x_constraints = linspace(0.01,0.8,80)*nnz(problem.var_edges);
     else
         x_constraints = linspace(0,7,80)*nnz(problem.var_edges)^2;
     end
     
    for ii=1:length(x_constraints)
        for y_constraint_frac=y_constraint_fracs
            test_param.x_constraint = x_constraints(ii);
            test_param.y_constraint = y_constraint_frac*T;
            test_param.eps = eps;
            test_param.verbose = 1;
            test_param.maxiters = 200;
            test_param.problem = problem;
            test_param.uncertainty_set_type = UncertaintySetType.Dnorm;


            for aa=[AdversaryAlgorithm.SubmodularMinimization, ...
                    AdversaryAlgorithm.SubsampledSubmodularMinimization, ...
                    AdversaryAlgorithm.ProjectedGradient]

                test_param.adversary_algorithm = aa;
                
                test_params(kk) = test_param;
                kk = kk + 1;
            end

        end
    end
end

run_all_params(test_params);