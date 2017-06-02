function x_grad = submodular_fct_influence_adversary_x_grad_sparse(log_x_scaled,y_mat_diag)
% submodular/concave adversarial influence maximization function,

% x_scaled_gpu is an S x T x n matrix already scaled to continuous values and with
% holes replaced with x_centers. (Here n is the number of points on which
% to evaluate the function)

% y_mat has (s,i)-th element equal to y_i(s) (i indexes the advertisers
% 1:k)

% the output val_gpu is an n x 1 vector on the GPU


%log_x_scaled = spfun(@log, x_scaled);

log_prod = y_mat_diag * log_x_scaled;
log_prod_summed = sum(log_prod);%sum along S

y_mat = diag(y_mat_diag);
log_neg_grad = log_prod_summed + log(y_mat) - log_x_scaled;

x_grad = -exp(log_neg_grad);
end