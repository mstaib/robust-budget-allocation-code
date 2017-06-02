function [ x_unscaled ] = discretize( x, x_lower, x_upper, k_vec )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% TODO: more explicitly enforce (or convert) column/row vectors
eps_vec = (x_upper - x_lower) ./ (k_vec - 1);
approx_k = (x(:) - x_lower(:)) ./ eps_vec(:);

x_unscaled = round(approx_k);
x_unscaled = max(0, x_unscaled);
x_unscaled = min(k_vec(:) - 1, x_unscaled(:));

end

