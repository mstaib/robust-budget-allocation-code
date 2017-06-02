function x_scaled = interpolate( x, x_lower, x_upper, k_vec )
%INTERPOLATE Summary of this function goes here
%   Detailed explanation goes here

x_scaled = x(:) ./ (k_vec(:) - 1) ...
    .* (x_upper(:) - x_lower(:)) ...
    + x_lower(:);

% when the discretization level is just one point, default to the lower
% value
k_vec_leq_1_inx = find(k_vec <= 1);
x_scaled(k_vec_leq_1_inx) = x_lower(k_vec_leq_1_inx);

end