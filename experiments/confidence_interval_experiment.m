clear all

num_advertisers = 1;
alpha = ones(1,num_advertisers);
eps = 0.001;
prob_edge = 1;

y_constraint_frac = 0.3;
x_constraint = 2;
S = 6;
T = 2;

seed = 2;
randn('state',seed);
rand('state',seed);

for max_num_trials = [5 10 15 20 25 30]
    [~, problem_confidence_func] = random_bipartite_problem_from_trials(S, T, num_advertisers, prob_edge, max_num_trials);

    for confidence = [0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99]
        problem = problem_confidence_func(confidence);
        problem.alpha = alpha;

        test_param.x_constraint = x_constraint;
        test_param.y_constraint = y_constraint_frac*T;
        test_param.eps = eps;
        test_param.verbose = 1;
        test_param.maxiters = 200;
        test_param.problem = problem;
        test_param.uncertainty_set_type = UncertaintySetType.Dnorm;

        try
            [ summary, out_struct_max ] = test_robust_influence_maximization_instance(test_param);
        catch ME
            fprintf('Caught exception for case confidence=%0.2f, skipping\n', confidence);
            fprintf('Exception message: %s\n', ME.message);
            continue
        end

        if  summary.rel_oblivious_gap > 0.05
            fprintf('Found nontrivial robustness with gap %0.3f, relative gap %0.3f\n', ...
                summary.oblivious_gap, summary.rel_oblivious_gap);
        end

        filename = sprintf('%d_customers_%d_channels_%d_advertisers_%0.1f_yconstraint_%0.2f_confidence_%d_maxnumtrials', T, S, num_advertisers, y_constraint_frac, confidence, max_num_trials);
        filepath = strcat('confidence_test_data/', filename, '.mat');

        fprintf('Finished run, saving to %s\n', filepath);

        save(filepath, 'out_struct_max', 'test_param', 'summary');
    end
end