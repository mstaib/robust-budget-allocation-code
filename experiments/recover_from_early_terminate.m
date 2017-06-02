from_beta_dist = isfield(test_param.problem, 'alphas');

problem = test_param.problem;
    
if test_param.uncertainty_set_type == UncertaintySetType.Ellipsoidal
    problem.x_upper = ones(size(problem.x_upper));
end

param_F = submodular_fct_influence_adversary_build_param_from_problem(problem, test_param.eps);


%% input setup for minimization problem
param_min.param_F = param_F;
param_min.maxiter = 50;
param_min.k_vec = param_F.k_vec; %param_F.k_mat(param_F.var_indices);
param_min.x_lower = param_F.x_lower; %param_F.x_lower(param_F.var_indices);
param_min.x_upper = param_F.x_upper; %param_F.x_upper(param_F.var_indices);
param_min.constraint = test_param.x_constraint;

% we should get eps/2 error from each of discretization and the actual
% minimization, param_F
param_min.early_terminate_gap = test_param.eps/100;%1e-3;
param_min.verbose = false;

%% input setup for maximization problem
param_max.maxiters = test_param.maxiters;
param_max.simplex_constraint = test_param.y_constraint;
param_max.early_terminate_gap = test_param.eps;%1e-4;

if isfield(test_param, 'verbose')
    param_max.verbose = (test_param.verbose >= 1);
    param_min.verbose = (test_param.verbose >= 2);
else
    param_max.verbose = false;
end

R_cell = cell(size(param_min.k_vec));
if test_param.uncertainty_set_type == UncertaintySetType.Ellipsoidal
    assert(from_beta_dist, 'Ellipsoidal uncertainty set only supported if the problem comes from a beta distribution\n');

    alphas = test_param.problem.alphas(param_F.var_edges_inx);
    betas = test_param.problem.betas(param_F.var_edges_inx);
    variance = alphas .* betas ./ ( (alphas+betas).^2 .* (alphas+betas+1) );

    for ii=1:length(param_min.k_vec)
        R_cell{ii} = @(t) (t - param_F.x_lower_vec(ii)).^2 / variance(ii).^2;
    end
else % UncertaintySetType.Dnorm
    for ii=1:length(param_min.k_vec)
        R_cell{ii} = @(t) (t - param_F.x_lower_vec(ii)) / (param_F.x_upper_vec(ii) - param_F.x_lower_vec(ii));
        %R_cell{ii} = @(t) abs(t - param_F.x_lower_vec(ii));
    end
end






param_F = param_min.param_F;
%% fill in the output of influence_maximize_robust
xmin = interpolate(xmin_unscaled, param_F.x_lower_vec, param_F.x_upper_vec, param_min.k_vec);

% how badly would we have done assuming no adversarial perturbations?
[ y_mat_oblivious, Fval, out, opts ] = solve_influence_fixed_x(zeros(size(xmin_unscaled)), y_mat_hist{end}, param_F, test_param.y_constraint);

% we need to optimize over the xs for this particular y
F = @(x,param_F) submodular_fct_influence_adversary_vec_wrapper(x,y_mat_oblivious,param_F);

%% store all we want
out_struct_max.Fval_hist = Fval_hist;
out_struct_max.dual_bound = dual_bound;
out_struct_max.dual_gap = dual_gap;
out_struct_max.x_dual_certificate_unscaled = xmin_ave_unscaled;
out_struct_max.stepsize = stepsize;
out_struct_max.iter_time = iter_time;
out_struct_max.y_mat_hist = y_mat_hist;
out_struct_max.xmin = xmin;
out_struct_max.xmin_unscaled = xmin_unscaled;
% out_struct_max.xmin_oblivious_unscaled = xmin_oblivious_unscaled;
% out_struct_max.Fmin_oblivious = Fmin_oblivious;
out_struct_max.y_mat_oblivious = y_mat_oblivious;


%% now that we've recovered the output of influence_maximize_robust, let's fill in the diagnostics


if from_beta_dist
    y_expect = maximize_expected_influence_beta(test_param.problem, test_param.y_constraint);
else
    y_expect = NaN; %filler
end

% summarize the run
x_constraint = test_param.x_constraint;
%summary = post_processing(out_struct_max, y_expect, test_param.problem, R_cell, test_param.x_constraint, from_beta_dist, param_min);
y_robust = out_struct_max.y_mat_hist{end};
y_nom = out_struct_max.y_mat_oblivious;

[expected_y_nom, nominal_y_nom, adversarial_y_nom] = evaluate_budget_quality(y_nom, problem, R_cell, x_constraint, from_beta_dist, param_min);
[expected_y_robust, nominal_y_robust, adversarial_y_robust] = evaluate_budget_quality(y_robust, problem, R_cell, x_constraint, from_beta_dist, param_min);

if from_beta_dist
    [expected_y_expect, nominal_y_expect, adversarial_y_expect] = evaluate_budget_quality(y_expect, problem, R_cell, x_constraint, from_beta_dist, param_min);
else
    expected_y_expect = NaN;
    nominal_y_expect = NaN;
    adversarial_y_expect = NaN;
end

summary.y_expect = y_expect;
summary.y_nom = y_nom;
summary.y_robust = y_robust;

summary.expected_y_expect = expected_y_expect;
summary.expected_y_nom = expected_y_nom;
summary.expected_y_robust = expected_y_robust;

summary.nominal_y_expect = nominal_y_expect;
summary.nominal_y_nom = nominal_y_nom;
summary.nominal_y_robust = nominal_y_robust;

summary.adversarial_y_expect = adversarial_y_expect;
summary.adversarial_y_nom = adversarial_y_nom;
summary.adversarial_y_robust = adversarial_y_robust;
% take in account possible errors due to not running alg long enough
summary.adversarial_y_robust_upper_bound = out_struct_max.dual_bound(end);

summary.oblivious_gap = summary.adversarial_y_robust - summary.adversarial_y_nom;
summary.rel_oblivious_gap = summary.oblivious_gap / summary.adversarial_y_nom;


% we now should have summary and out_struct_max as usual