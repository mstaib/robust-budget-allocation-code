function [ k_mat ] = discretize_bipartite_problem( eps, problem )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% eps: desired global suboptimality
% prob: conatins x_upper, x_median (replace with mean?), and real_edges

% we will always want to increase all the xs, so why even both discretizing
% the lower half of the space?

gaps = problem.x_upper - problem.x_median;%.x_lower;
k_vals = round(4 * gaps/eps) + 1;
k_mat = problem.real_edges.*k_vals;

end