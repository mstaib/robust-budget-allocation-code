function [ y_mat, Fval, out, opts ] = solve_influence_fixed_x( x_unscaled, y_mat_init, param_F, C )
%SOLVE_INFLUENCE_FIXED Summary of this function goes here
%   Solves the bipartite influence maximization problem for y, for fixed
%   failure probabilities x in discrete ("x_unscaled") form

warning('off');

opts = tfocs;
opts.printEvery = 0;
opts.maxmin = -1; %do maximization

% TODO: there appears to be a bug in TFOCS's implementation of proj_simplex
% so that it doesn't always work when we use inequality constraints instead
% of equality constraints. Hence we are using equality constraints for now
[ y_mat, out, opts ] = tfocs( @(y) smoothF(y, x_unscaled, param_F), ...
    [], ... %affineF
    proj_simplex(C, false, false), ... %projectorF %@(x,t) proj_feas(x), ... %projectorF
    y_mat_init, ... %x0
    opts ); %opts
warning('on');

% compute this separately so we don't use the output from TFOCS in out.f
% (which will often return infinity because we get a slighty infeasible
% solution)
Fval = smoothF(y_mat, x_unscaled, param_F);

% TFOCS functions
function [ fy, gy ] = smoothF(y, x_unscaled, param_F)
    [fy, gy] = submodular_fct_influence_adversary_y_grad_vec_wrapper(x_unscaled, y, param_F);
%     fy = submodular_fct_influence_adversary_vec_wrapper(x_unscaled, y, param_F);
%     gy = submodular_fct_influence_adversary_y_grad_vec_wrapper(x_unscaled, y, param_F);
end

end

