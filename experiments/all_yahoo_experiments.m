PARALLEL=0;
addpath /home/gridsan/mstaib/research/submodular-networked-control/submodular-saddle-point/submodular_multi_online
addpath /home/gridsan/mstaib/research/submodular-networked-control/submodular-saddle-point/submodular_multi_online/continuous_submodular_minimization

eps = 0.01;
prob_edge = 1;

y_constraint_fracs = logspace(log10(0.2), log10(10), 8);

seed = 2;
randn('state',seed);
rand('state',seed);

max_num_trials = 5;
confidence = 0.95;

data_frac = 0.5;
problem_confidence_func = get_yahoo_problem_func(data_frac);
problem = problem_confidence_func(confidence);
problem.var_edges = problem.real_edges;

kk = 1;
for ust=[UncertaintySetType.Dnorm, ...
         UncertaintySetType.Ellipsoidal]

     if ust == UncertaintySetType.Dnorm
         x_constraints = linspace(0.01,0.8,8)*nnz(problem.var_edges);
     else
         x_constraints = linspace(0,7,8)*nnz(problem.var_edges)^2;
     end
     
    for ii=1:length(x_constraints)
        for y_constraint_frac=y_constraint_fracs
            test_param.x_constraint = x_constraints(ii);
            test_param.y_constraint = y_constraint_frac*problem.S;
            test_param.eps = eps;
            test_param.verbose = 2;
            test_param.maxiters = 2;
            test_param.problem = problem;
            test_param.uncertainty_set_type = ust;


            for aa=[AdversaryAlgorithm.SubmodularMinimization]
                    %AdversaryAlgorithm.SubsampledSubmodularMinimization]
                    %AdversaryAlgorithm.ProjectedGradient]

                test_param.adversary_algorithm = aa;
                
                test_params(kk) = test_param;
                kk = kk + 1;
            end

        end
    end
end

n = length(test_params);
if PARALLEL
    myIndices = global_ind(zeros(n,1,map([Np 1],'c',0:Np-1)));
    myIndices
else
    myIndices = 1;
end
run_all_params(test_params(myIndices));
