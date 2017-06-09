function [ y_expect ] = maximize_expected_influence_beta(problem, y_constraint)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

smoothF = @(y_mat) expected_influence_beta_oracle(y_mat,problem);
opts = tfocs;
opts.maxmin = -1;
opts.printEvery = 1;
warning('off');
[y, out, opts] = tfocs(smoothF, [], proj_simplex(y_constraint,false,false), zeros(problem.S,1), opts);
warning('on');

y_expect = y;


end

