function [w,f,Fmin] = greedy_algorithm(rho,F,param_F)
% greedy algorithm (assumes all variables have the same cardinality)

% make sure rho is n x 1
rho = rho(:);
n = length(rho);

[is, js] = build_increment_ordering_mex(rho);

rho_lengths = cellfun(@length, rho);

% now go through all elements
xold = zeros(n,1);
H0 = F(xold,param_F);

% F_vals_all(1) is just F(xold,param_F)
F_vals_all = submodular_fct_influence_adversary_vec_wrapper_update_based(xold, is, js, param_F);

[w, Fmin] = build_w_from_F_vals_mex(F_vals_all, is, js, rho_lengths);

% f is affine in the rho; the variable f is the actual function value, and
% the vector w is a subgradient
f = cell_innerprod_mex(w, rho) + H0;
end