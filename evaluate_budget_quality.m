function [ expect_val, nom_val, adversarial_val ] = evaluate_budget_quality( y, problem, R_cell, oracle_R, x_constraint, from_beta_dist, param_min )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

if from_beta_dist
    expect_val = expected_influence_beta_oracle(y, problem);
else
    expect_val = NaN;
end

if nargin < 6 % didn't supply param_min
    param_min = prepare_param_min(problem, x_constraint);
end

F = @(x,param_F) submodular_fct_influence_adversary_vec_wrapper(x, y, param_F);

param_min.param_F.y_mat = y;
[x_adversary,Fmin,out_struct] = minimize_submodular_constrained(F, R_cell, oracle_R, param_min);

nom_val = F(zeros(size(x_adversary)), param_min.param_F);
adversarial_val = F(x_adversary, param_min.param_F);

end

function [param_min] = prepare_param_min(problem, x_constraint)
    eps = 0.01;
    param_F = submodular_fct_influence_adversary_build_param_from_problem(problem, eps);

    param_min.param_F = param_F;
    param_min.maxiter = 100;
    param_min.k_vec = param_F.k_mat(param_F.var_edges);
    param_min.x_lower = param_F.x_lower(param_F.var_edges);
    param_min.x_upper = param_F.x_upper(param_F.var_edges);
    param_min.constraint = x_constraint; %this is in terms of the continuous formulation

    % we should get eps/2 error from each of discretization and the actual
    % minimization
    param_min.early_terminate_gap = eps/20;%1e-3;
    param_min.verbose = false;
end